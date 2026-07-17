package com.manga.translate

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.pdf.PdfDocument
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.ParcelFileDescriptor
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import kotlin.math.roundToInt

internal object PdfImageCodec {
    private const val IMPORT_RENDER_SCALE = 2f
    private const val IMPORT_FILE_NAME_WIDTH = 4
    private const val PDF_JPEG_QUALITY = 100

    fun renderPdfToImages(
        contentResolver: ContentResolver,
        uri: Uri,
        outputDir: File
    ): Int {
        val descriptor = contentResolver.openFileDescriptor(uri, "r") ?: return 0
        return descriptor.use { pfd ->
            PdfRenderer(pfd).use { renderer ->
                var imported = 0
                for (index in 0 until renderer.pageCount) {
                    val page = renderer.openPage(index)
                    try {
                        val bitmap = Bitmap.createBitmap(
                            (page.width * IMPORT_RENDER_SCALE).roundToInt().coerceAtLeast(1),
                            (page.height * IMPORT_RENDER_SCALE).roundToInt().coerceAtLeast(1),
                            Bitmap.Config.ARGB_8888
                        )
                        try {
                            bitmap.eraseColor(0xFFFFFFFF.toInt())
                            val matrix = Matrix().apply {
                                setScale(IMPORT_RENDER_SCALE, IMPORT_RENDER_SCALE)
                            }
                            page.render(bitmap, null, matrix, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                            val fileName = "${(index + 1).toString().padStart(IMPORT_FILE_NAME_WIDTH, '0')}.png"
                            FileOutputStream(File(outputDir, fileName)).use { output ->
                                bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
                            }
                            imported += 1
                        } finally {
                            bitmap.recycle()
                        }
                    } finally {
                        page.close()
                    }
                }
                imported
            }
        }
    }

    fun writeImagesToPdf(
        images: List<File>,
        outputStream: OutputStream
    ): Boolean {
        if (images.isEmpty()) return false
        val document = PdfDocument()
        return try {
            images.forEachIndexed { index, image ->
                val bitmap = android.graphics.BitmapFactory.decodeFile(image.absolutePath) ?: return false
                try {
                    val pageInfo = PdfDocument.PageInfo.Builder(
                        bitmap.width.coerceAtLeast(1),
                        bitmap.height.coerceAtLeast(1),
                        index + 1
                    ).create()
                    val page = document.startPage(pageInfo)
                    try {
                        drawBitmapFit(page.canvas, bitmap)
                    } finally {
                        document.finishPage(page)
                    }
                } finally {
                    bitmap.recycle()
                }
            }
            document.writeTo(outputStream)
            true
        } catch (e: Exception) {
            AppLogger.log("PdfImageCodec", "Write PDF failed", e)
            false
        } finally {
            document.close()
        }
    }

    fun writeImagesToPdfWithOutlines(
        chapterOutlines: List<Pair<String, Int>>,
        images: List<File>,
        outputStream: OutputStream
    ): Boolean {
        if (images.isEmpty()) return false
        return try {
            streamPdfWithOutlines(chapterOutlines, images, outputStream)
            true
        } catch (e: Exception) {
            AppLogger.log("PdfImageCodec", "Write PDF with outlines failed", e)
            false
        }
    }

    private fun streamPdfWithOutlines(
        chapterOutlines: List<Pair<String, Int>>,
        images: List<File>,
        out: OutputStream
    ) {
        val n = images.size
        val m = chapterOutlines.size

        val maxObjectNumber = 3 + 3 * n + m
        val offsets = LongArray(maxObjectNumber + 1) { -1L }
        var byteCount = 0L

        val counter = object : OutputStream() {
            override fun write(b: Int) { out.write(b); byteCount++ }
            override fun write(b: ByteArray) { out.write(b); byteCount += b.size }
            override fun write(b: ByteArray, off: Int, len: Int) {
                out.write(b, off, len)
                byteCount += len
            }
        }

        fun writeln(s: String) {
            val bytes = "$s\n".toByteArray()
            counter.write(bytes)
        }

        fun beginObj(objNum: Int) {
            offsets[objNum] = byteCount
            counter.write("$objNum 0 obj\n".toByteArray())
        }

        fun endObj() {
            counter.write("\nendobj\n".toByteArray())
        }

        fun indirectRef(objNum: Int): String = "$objNum 0 R"

        fun pdfString(s: String): String {
            val sb = StringBuilder("<FEFF")
            for (c in s) {
                val code = c.code
                if (code <= 0xFFFF) {
                    sb.append(String.format("%04X", code))
                } else {
                    val high = 0xD800 + ((code - 0x10000) shr 10)
                    val low = 0xDC00 + ((code - 0x10000) and 0x3FF)
                    sb.append(String.format("%04X%04X", high, low))
                }
            }
            sb.append(">")
            return sb.toString()
        }

        // Object layout (object numbers are deterministic):
        // 1: Catalog
        // 2: Pages root
        // 3: Outlines root
        // 4..3+n: Page objects
        // 4+n..3+2n: Content streams
        // 4+2n..3+3n: Image XObjects
        // 4+3n..3+3n+m: Outline items

        val pageObjBase = 4
        val contentObjBase = 4 + n
        val imageObjBase = 4 + 2 * n
        val outlineObjBase = 4 + 3 * n

        writeln("%PDF-1.4")
        writeln("%âãÏÓ")

        // 1: Catalog
        beginObj(1)
        writeln("<< /Type /Catalog /Pages ${indirectRef(2)} /Outlines ${indirectRef(3)} >>")
        endObj()

        // 2: Pages root
        beginObj(2)
        val kids = (0 until n).joinToString(" ") { indirectRef(pageObjBase + it) }
        writeln("<< /Type /Pages /Kids [$kids] /Count $n >>")
        endObj()

        // 3: Outlines root
        beginObj(3)
        val firstOutlineRef = if (m > 0) "/First ${indirectRef(outlineObjBase)}" else ""
        val lastOutlineRef = if (m > 0) "/Last ${indirectRef(outlineObjBase + m - 1)}" else ""
        writeln("<< /Type /Outlines $firstOutlineRef $lastOutlineRef >>")
        endObj()

        // Pages — process one image at a time to keep memory low
        for (i in 0 until n) {
            val imageFile = images[i]
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(imageFile.absolutePath, bounds)
            val w = bounds.outWidth
            val h = bounds.outHeight
            if (w <= 0 || h <= 0) {
                throw IllegalStateException("Cannot decode image: ${imageFile.name}")
            }

            val jpegBytes = encodeImageAsJpegBytes(imageFile)
                ?: throw IllegalStateException("Cannot encode image as JPEG: ${imageFile.name}")

            // Image XObject
            beginObj(imageObjBase + i)
            writeln("<< /Type /XObject /Subtype /Image /Width $w /Height $h " +
                "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode " +
                "/Length ${jpegBytes.size} >>")
            writeln("stream")
            counter.write(jpegBytes)
            writeln("")
            writeln("endstream")
            endObj()

            // Content stream
            beginObj(contentObjBase + i)
            val content = "q $w 0 0 $h 0 0 cm /Im0 Do Q"
            val contentBytes = content.toByteArray()
            writeln("<< /Length ${contentBytes.size} >>")
            writeln("stream")
            counter.write(contentBytes)
            writeln("")
            writeln("endstream")
            endObj()

            // Page object
            beginObj(pageObjBase + i)
            writeln("<< /Type /Page /Parent ${indirectRef(2)} " +
                "/MediaBox [0 0 $w $h] " +
                "/Contents ${indirectRef(contentObjBase + i)} " +
                "/Resources << /XObject << /Im0 ${indirectRef(imageObjBase + i)} >> >> >>")
            endObj()
        }

        // Outline items
        for (i in 0 until m) {
            val (chapterName, firstPage) = chapterOutlines[i]
            val objNum = outlineObjBase + i
            val nextRef = if (i < m - 1) "/Next ${indirectRef(objNum + 1)}" else ""
            val prevRef = if (i > 0) "/Prev ${indirectRef(objNum - 1)}" else ""
            val destPage = firstPage.coerceAtMost(n - 1)

            beginObj(objNum)
            writeln("<< /Title ${pdfString(chapterName)} " +
                "/Parent ${indirectRef(3)} " +
                "/Dest [${indirectRef(pageObjBase + destPage)} /Fit] " +
                "$nextRef $prevRef >>")
            endObj()
        }

        val startxref = byteCount

        // Cross-reference table
        writeln("xref")
        writeln("0 ${maxObjectNumber + 1}")
        writeln("0000000000 65535 f ")
        for (objNum in 1..maxObjectNumber) {
            val offset = offsets[objNum]
            if (offset < 0) {
                throw IllegalStateException("Missing PDF object offset: $objNum")
            }
            writeln(String.format("%010d 00000 n ", offset))
        }

        // Trailer
        writeln("trailer")
        writeln("<< /Size ${maxObjectNumber + 1} /Root ${indirectRef(1)} >>")
        writeln("startxref")
        writeln("$startxref")
        counter.write("%%EOF".toByteArray())
    }

    private fun encodeImageAsJpegBytes(imageFile: File): ByteArray? {
        val ext = imageFile.name.substringAfterLast('.', "").lowercase()
        if (ext == "jpg" || ext == "jpeg") {
            return runCatching { imageFile.readBytes() }.getOrNull()
        }
        val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath) ?: return null
        return try {
            val jpegStream = ByteArrayOutputStream()
            if (!bitmap.compress(Bitmap.CompressFormat.JPEG, PDF_JPEG_QUALITY, jpegStream)) {
                null
            } else {
                jpegStream.toByteArray()
            }
        } finally {
            bitmap.recycle()
        }
    }

    private fun drawBitmapFit(canvas: Canvas, bitmap: Bitmap) {
        val src = Rect(0, 0, bitmap.width, bitmap.height)
        val dst = Rect(0, 0, canvas.width, canvas.height)
        canvas.drawColor(0xFFFFFFFF.toInt())
        canvas.drawBitmap(bitmap, src, dst, Paint(Paint.FILTER_BITMAP_FLAG))
    }
}

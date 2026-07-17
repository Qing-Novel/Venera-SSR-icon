package com.manga.translate

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapRegionDecoder
import android.graphics.Color
import android.graphics.Rect
import android.graphics.RectF
import android.os.Build
import java.io.File
import kotlin.math.max
import kotlin.math.roundToInt

internal object BubbleColorSampler {

    private const val SAMPLE_STEP = 4
    private const val FILE_SAMPLE_MAX_EDGE = 96

    fun sampleBackgroundColor(
        bitmap: Bitmap?,
        left: Float,
        top: Float,
        right: Float,
        bottom: Float
    ): Int? {
        val src = bitmap ?: return null
        val bitmapWidth = src.width
        val bitmapHeight = src.height
        if (bitmapWidth <= 0 || bitmapHeight <= 0) return null

        val leftPx = left.toInt().coerceIn(0, bitmapWidth - 1)
        val topPx = top.toInt().coerceIn(0, bitmapHeight - 1)
        val rightPx = right.toInt().coerceIn(leftPx + 1, bitmapWidth)
        val bottomPx = bottom.toInt().coerceIn(topPx + 1, bitmapHeight)

        val regionWidth = rightPx - leftPx
        val regionHeight = bottomPx - topPx
        if (regionWidth <= 0 || regionHeight <= 0) return null

        val stepX = SAMPLE_STEP.coerceAtLeast(1)
        val stepY = SAMPLE_STEP.coerceAtLeast(1)

        val tempBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
            src.config == Bitmap.Config.HARDWARE
        ) {
            src.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            null
        }
        val samplingBitmap = tempBitmap ?: src
        try {
            return averagePixels(
                samplingBitmap = samplingBitmap,
                leftPx = leftPx,
                topPx = topPx,
                rightPx = rightPx,
                bottomPx = bottomPx,
                stepX = stepX,
                stepY = stepY
            )
        } finally {
            tempBitmap?.recycle()
        }
    }

    fun sampleBackgroundColor(bitmap: Bitmap?, rect: RectF): Int? {
        return sampleBackgroundColor(bitmap, rect.left, rect.top, rect.right, rect.bottom)
    }

    /**
     * Samples bubble background from a full in-memory bitmap when available,
     * otherwise decodes only the target region from [imageFile] (tiled / long pages).
     *
     * Coordinates are in source-image space (same as bubble rects).
     */
    fun sampleBackgroundColor(
        bitmap: Bitmap?,
        imageFile: File?,
        sourceWidth: Int,
        sourceHeight: Int,
        left: Float,
        top: Float,
        right: Float,
        bottom: Float
    ): Int? {
        if (bitmap != null && sourceWidth > 0 && sourceHeight > 0) {
            val sampleScaleX = bitmap.width.toFloat() / sourceWidth.toFloat()
            val sampleScaleY = bitmap.height.toFloat() / sourceHeight.toFloat()
            sampleBackgroundColor(
                bitmap,
                left * sampleScaleX,
                top * sampleScaleY,
                right * sampleScaleX,
                bottom * sampleScaleY
            )?.let { return it }
        } else if (bitmap != null) {
            sampleBackgroundColor(bitmap, left, top, right, bottom)?.let { return it }
        }
        return sampleBackgroundColorFromFile(imageFile, sourceWidth, sourceHeight, left, top, right, bottom)
    }

    fun sampleBackgroundColorFromFile(
        imageFile: File?,
        sourceWidth: Int,
        sourceHeight: Int,
        left: Float,
        top: Float,
        right: Float,
        bottom: Float
    ): Int? {
        val file = imageFile ?: return null
        if (!file.isFile) return null
        if (ImageFileSupport.isAvifFile(file.name)) return null

        val boundsWidth: Int
        val boundsHeight: Int
        if (sourceWidth > 0 && sourceHeight > 0) {
            boundsWidth = sourceWidth
            boundsHeight = sourceHeight
        } else {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(file.absolutePath, bounds)
            boundsWidth = bounds.outWidth
            boundsHeight = bounds.outHeight
        }
        if (boundsWidth <= 0 || boundsHeight <= 0) return null

        val leftPx = left.roundToInt().coerceIn(0, boundsWidth - 1)
        val topPx = top.roundToInt().coerceIn(0, boundsHeight - 1)
        val rightPx = right.roundToInt().coerceIn(leftPx + 1, boundsWidth)
        val bottomPx = bottom.roundToInt().coerceIn(topPx + 1, boundsHeight)
        if (rightPx <= leftPx || bottomPx <= topPx) return null

        val region = Rect(leftPx, topPx, rightPx, bottomPx)
        val regionWidth = region.width()
        val regionHeight = region.height()
        val maxEdge = max(regionWidth, regionHeight)
        val sampleSize = if (maxEdge <= FILE_SAMPLE_MAX_EDGE) {
            1
        } else {
            var sample = 1
            while (maxEdge / (sample * 2) >= FILE_SAMPLE_MAX_EDGE) {
                sample *= 2
            }
            sample
        }

        val decoder = runCatching { createBitmapRegionDecoder(file) }.getOrNull() ?: return null
        return try {
            val options = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val cropped = decoder.decodeRegion(region, options) ?: return null
            try {
                sampleBackgroundColor(cropped, 0f, 0f, cropped.width.toFloat(), cropped.height.toFloat())
            } finally {
                cropped.recycle()
            }
        } finally {
            decoder.recycle()
        }
    }

    private fun averagePixels(
        samplingBitmap: Bitmap,
        leftPx: Int,
        topPx: Int,
        rightPx: Int,
        bottomPx: Int,
        stepX: Int,
        stepY: Int
    ): Int? {
        var r = 0L
        var g = 0L
        var b = 0L
        var count = 0

        val yEnd = bottomPx - 1
        val xEnd = rightPx - 1
        var y = topPx
        while (y <= yEnd) {
            var x = leftPx
            while (x <= xEnd) {
                val pixel = samplingBitmap.getPixel(x, y)
                r += Color.red(pixel)
                g += Color.green(pixel)
                b += Color.blue(pixel)
                count++
                x += stepX
            }
            y += stepY
        }

        if (count == 0) return null
        return Color.rgb((r / count).toInt(), (g / count).toInt(), (b / count).toInt())
    }

    private fun createBitmapRegionDecoder(imageFile: File): BitmapRegionDecoder {
        val path = imageFile.absolutePath
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            BitmapRegionDecoder.newInstance(path)
        } else {
            @Suppress("DEPRECATION")
            BitmapRegionDecoder.newInstance(path, false)
        }
    }
}

package com.manga.translate

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import androidx.core.graphics.createBitmap
import kotlin.math.max
import kotlin.math.min

internal object BubbleShapePaths {
    private val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }

    fun buildPath(
        outPath: Path,
        bubble: BubbleTranslation,
        sourceWidth: Int,
        sourceHeight: Int,
        originX: Float,
        originY: Float,
        scaleX: Float,
        scaleY: Float,
        offsetX: Float = 0f,
        offsetY: Float = 0f,
        shrinkPercent: Int = 0,
        fallbackCornerRadius: Float = 6f
    ) {
        outPath.reset()
        val contour = bubble.maskContour
        if (sourceWidth > 0 && sourceHeight > 0 && contour != null && contour.size >= 6) {
            outPath.moveTo(
                originX + (contour[0] * sourceWidth + offsetX) * scaleX,
                originY + (contour[1] * sourceHeight + offsetY) * scaleY
            )
            var index = 2
            while (index + 1 < contour.size) {
                outPath.lineTo(
                    originX + (contour[index] * sourceWidth + offsetX) * scaleX,
                    originY + (contour[index + 1] * sourceHeight + offsetY) * scaleY
                )
                index += 2
            }
            outPath.close()
        } else {
            outPath.addRoundRect(
                originX + (bubble.rect.left + offsetX) * scaleX,
                originY + (bubble.rect.top + offsetY) * scaleY,
                originX + (bubble.rect.right + offsetX) * scaleX,
                originY + (bubble.rect.bottom + offsetY) * scaleY,
                fallbackCornerRadius,
                fallbackCornerRadius,
                Path.Direction.CW
            )
        }
        applyShrink(outPath, shrinkPercent)
    }

    fun translateMaskContour(
        contour: FloatArray?,
        deltaX: Float,
        deltaY: Float,
        sourceWidth: Int,
        sourceHeight: Int
    ): FloatArray? {
        if (contour == null || contour.size < 2 || sourceWidth <= 0 || sourceHeight <= 0) {
            return contour?.copyOf()
        }
        val normalizedDeltaX = deltaX / sourceWidth.toFloat()
        val normalizedDeltaY = deltaY / sourceHeight.toFloat()
        return FloatArray(contour.size) { index ->
            contour[index] + if (index % 2 == 0) normalizedDeltaX else normalizedDeltaY
        }
    }

    fun insetTextBounds(path: Path, outRect: RectF) {
        val pathBounds = RectF()
        path.computeBounds(pathBounds, true)
        outRect.set(pathBounds)
        if (pathBounds.width() <= 0f || pathBounds.height() <= 0f) return

        val pad = (minOf(pathBounds.width(), pathBounds.height()) * 0.08f).coerceAtLeast(6f)
        outRect.inset(pad, pad)
        if (outRect.width() <= 0f || outRect.height() <= 0f) {
            outRect.set(pathBounds)
            return
        }

        val safeRect = cachedSafeTextRect(path, pathBounds, pad)
        if (safeRect != null && safeRect.width() > 0f && safeRect.height() > 0f) {
            outRect.set(safeRect)
        }
    }

    private data class SafeRectCacheKey(
        val left: Int,
        val top: Int,
        val right: Int,
        val bottom: Int,
        val pad: Float
    )

    private val safeRectCache = androidx.collection.LruCache<SafeRectCacheKey, RectF>(128)

    fun cachedSafeTextRect(
        path: Path,
        pathBounds: RectF,
        fallbackPad: Float
    ): RectF? {
        val key = SafeRectCacheKey(
            left = pathBounds.left.toInt(),
            top = pathBounds.top.toInt(),
            right = pathBounds.right.toInt(),
            bottom = pathBounds.bottom.toInt(),
            pad = fallbackPad
        )
        safeRectCache.get(key)?.let { cached ->
            if (cached.width() > 0f && cached.height() > 0f) {
                return RectF(cached)
            }
        }
        val computed = estimateSafeTextRect(path, pathBounds, fallbackPad) ?: return null
        if (computed.width() > 0f && computed.height() > 0f) {
            safeRectCache.put(key, RectF(computed))
        }
        return computed
    }

    fun clearSafeRectCache() {
        safeRectCache.evictAll()
    }

    private fun applyShrink(path: Path, shrinkPercent: Int) {
        val normalizedPercent = shrinkPercent.coerceIn(0, 95)
        if (normalizedPercent <= 0) return
        val tempBounds = RectF()
        path.computeBounds(tempBounds, true)
        if (tempBounds.width() <= 0f || tempBounds.height() <= 0f) return
        val scale = (100f - normalizedPercent) / 100f
        val tempMatrix = Matrix()
        tempMatrix.setScale(scale, scale, tempBounds.centerX(), tempBounds.centerY())
        path.transform(tempMatrix)
    }

    private fun estimateSafeTextRect(path: Path, pathBounds: RectF, fallbackPad: Float): RectF? {
        val maxMaskSize = 96
        val maskWidth = pathBounds.width().toInt().coerceIn(16, maxMaskSize)
        val maskHeight = pathBounds.height().toInt().coerceIn(16, maxMaskSize)
        if (maskWidth <= 1 || maskHeight <= 1) return null

        val maskBitmap = createBitmap(maskWidth, maskHeight)
        val maskCanvas = Canvas(maskBitmap)
        val maskPath = Path(path)
        val matrix = Matrix().apply {
            postTranslate(-pathBounds.left, -pathBounds.top)
            postScale(
                (maskWidth - 1).toFloat() / pathBounds.width().coerceAtLeast(1f),
                (maskHeight - 1).toFloat() / pathBounds.height().coerceAtLeast(1f)
            )
        }
        maskPath.transform(matrix)
        maskCanvas.drawColor(Color.TRANSPARENT)
        maskCanvas.drawPath(maskPath, maskPaint)

        val pixels = IntArray(maskWidth * maskHeight)
        maskBitmap.getPixels(pixels, 0, maskWidth, 0, 0, maskWidth, maskHeight)
        maskBitmap.recycle()

        val filled = BooleanArray(maskWidth * maskHeight)
        for (y in 0 until maskHeight) {
            val rowOffset = y * maskWidth
            for (x in 0 until maskWidth) {
                filled[rowOffset + x] = (pixels[rowOffset + x] ushr 24) >= 224
            }
        }

        val safeMaskRect = findLargestFilledRect(filled, maskWidth, maskHeight) ?: return null
        val left = safeMaskRect.left
        val top = safeMaskRect.top
        val right = safeMaskRect.right
        val bottom = safeMaskRect.bottom

        val widthScale = pathBounds.width() / maskWidth.toFloat()
        val heightScale = pathBounds.height() / maskHeight.toFloat()
        val extraPadX = min(fallbackPad, pathBounds.width() * 0.12f)
        val extraPadY = min(fallbackPad, pathBounds.height() * 0.12f)
        val safeRect = RectF(
            pathBounds.left + left * widthScale + extraPadX * 0.35f,
            pathBounds.top + top * heightScale + extraPadY * 0.35f,
            pathBounds.left + right * widthScale - extraPadX * 0.35f,
            pathBounds.top + bottom * heightScale - extraPadY * 0.35f
        )
        return if (safeRect.width() > pathBounds.width() * 0.18f &&
            safeRect.height() > pathBounds.height() * 0.18f
        ) {
            safeRect
        } else {
            null
        }
    }

    private fun findLargestFilledRect(filled: BooleanArray, width: Int, height: Int): MaskRect? {
        val heights = IntArray(width)
        val stack = IntArray(width + 1)
        var best: MaskRect? = null
        var bestScore = 0f
        for (y in 0 until height) {
            for (x in 0 until width) {
                heights[x] = if (filled[y * width + x]) heights[x] + 1 else 0
            }
            var stackSize = 0
            var x = 0
            while (x <= width) {
                val currentHeight = if (x == width) 0 else heights[x]
                if (stackSize == 0 || currentHeight >= heights[stack[stackSize - 1]]) {
                    stack[stackSize++] = x
                    x += 1
                } else {
                    val topIndex = stack[--stackSize]
                    val rectHeight = heights[topIndex]
                    if (rectHeight <= 0) continue
                    val rectRight = x
                    val rectLeft = if (stackSize == 0) 0 else stack[stackSize - 1] + 1
                    val rectWidth = rectRight - rectLeft
                    val score = scoreTextRect(rectWidth, rectHeight)
                    if (score > bestScore) {
                        bestScore = score
                        best = MaskRect(
                            left = rectLeft,
                            top = y - rectHeight + 1,
                            right = rectRight,
                            bottom = y + 1
                        )
                    }
                }
            }
        }
        return best
    }

    private fun scoreTextRect(width: Int, height: Int): Float {
        val minSide = min(width, height).toFloat()
        val maxSide = max(width, height).toFloat().coerceAtLeast(1f)
        val balance = (minSide / maxSide).coerceIn(0.35f, 1f)
        return width * height * balance
    }

    private data class MaskRect(
        val left: Int,
        val top: Int,
        val right: Int,
        val bottom: Int
    )
}

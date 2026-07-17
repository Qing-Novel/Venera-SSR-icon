package com.manga.translate

import android.graphics.Matrix
import android.graphics.Path
import android.graphics.RectF
import android.text.StaticLayout
import kotlin.math.max
import kotlin.math.sqrt

internal object BubbleTextScaling {
    fun layoutFits(layout: StaticLayout, maxWidth: Int, maxHeight: Int): Boolean {
        if (layout.height > maxHeight) return false
        for (line in 0 until layout.lineCount) {
            if (layout.getLineWidth(line) > maxWidth + 0.5f) {
                return false
            }
        }
        return true
    }

    fun scalePathAroundCenter(path: Path, scaleX: Float, scaleY: Float) {
        if (scaleX == 1f && scaleY == 1f) return
        val bounds = RectF()
        path.computeBounds(bounds, true)
        if (bounds.width() <= 0f || bounds.height() <= 0f) return
        val matrix = Matrix()
        matrix.setScale(scaleX, scaleY, bounds.centerX(), bounds.centerY())
        path.transform(matrix)
    }

    /**
     * Ensures the bubble path provides at least [minAreaPerCharSp] area per character
     * in the text-safe region. If the current area is insufficient, the path is scaled
     * around its center in one shot.
     *
     * @return the text-safe RectF inside the (possibly expanded) path
     */
    fun resolveAreaAdjustedTextRect(
        text: String,
        path: Path,
        minAreaPerCharSp: Float,
        density: Float
    ): RectF {
        val textRect = RectF()
        BubbleShapePaths.insetTextBounds(path, textRect)
        if (textRect.width() <= 0f || textRect.height() <= 0f) return textRect

        val charCount = text.trim().length.coerceAtLeast(1)
        val areaPx = textRect.width() * textRect.height()
        val areaSp2 = areaPx / (density * density)
        val areaPerCharSp = areaSp2 / charCount.toFloat()

        if (areaPerCharSp < minAreaPerCharSp) {
            val targetAreaPx = charCount * minAreaPerCharSp * density * density
            val areaScale = (targetAreaPx / areaPx).coerceAtMost(9f)
            val linearScale = sqrt(areaScale).coerceIn(1f, 3f)
            scalePathAroundCenter(path, linearScale, linearScale)
            BubbleShapePaths.insetTextBounds(path, textRect)
        }
        return textRect
    }

    fun findDefaultHorizontalTextSize(
        text: String,
        maxWidth: Int,
        maxHeight: Int,
        minTextSizePx: Float,
        buildLayout: (String, Int, Float) -> StaticLayout,
        layoutFits: (StaticLayout, Int, Int) -> Boolean
    ): Float {
        val minLayout = buildLayout(text, maxWidth, minTextSizePx)
        if (!layoutFits(minLayout, maxWidth, maxHeight)) {
            return minTextSizePx
        }
        var bestSize = minTextSizePx
        var low = minTextSizePx
        var high = maxOf(low, maxWidth.toFloat(), maxHeight.toFloat())
        val stepPx = 0.5f
        while (high - low > stepPx) {
            val mid = (low + high) / 2f
            val candidate = buildLayout(text, maxWidth, mid)
            if (layoutFits(candidate, maxWidth, maxHeight)) {
                bestSize = mid
                low = mid
            } else {
                high = mid
            }
        }
        return bestSize
    }
}

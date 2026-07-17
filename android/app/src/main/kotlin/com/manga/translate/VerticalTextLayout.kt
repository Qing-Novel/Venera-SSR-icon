package com.manga.translate

import android.graphics.Paint
import android.text.TextPaint

internal data class VerticalTextLayout(
    val columnWidth: Float,
    val lineHeight: Float,
    val maxRows: Int,
    val columns: Int,
    val totalWidth: Float,
    val totalHeight: Float,
    val fontMetrics: Paint.FontMetrics,
    val fits: Boolean
)

internal object VerticalTextLayoutCalculator {
    fun build(
        textPaint: TextPaint,
        text: String,
        maxWidth: Int,
        maxHeight: Int,
        textSize: Float
    ): VerticalTextLayout {
        textPaint.textSize = textSize
        val fontMetrics = textPaint.fontMetrics
        val lineHeight = (fontMetrics.descent - fontMetrics.ascent).coerceAtLeast(1f)
        val maxRows = (maxHeight / lineHeight).toInt().coerceAtLeast(1)
        val charCount = text.count { it != '\n' }.coerceAtLeast(1)
        var maxCharWidth = 0f
        for (ch in text) {
            if (ch == '\n') continue
            val width = textPaint.measureText(ch.toString())
            if (width > maxCharWidth) {
                maxCharWidth = width
            }
        }
        if (maxCharWidth <= 0f) {
            maxCharWidth = textPaint.measureText("国")
        }
        maxCharWidth = maxCharWidth.coerceAtLeast(1f)
        val columns = ((charCount + maxRows - 1) / maxRows).coerceAtLeast(1)
        val totalWidth = columns * maxCharWidth
        val totalHeight = maxRows * lineHeight
        val fits = totalWidth <= maxWidth && totalHeight <= maxHeight
        return VerticalTextLayout(
            columnWidth = maxCharWidth,
            lineHeight = lineHeight,
            maxRows = maxRows,
            columns = columns,
            totalWidth = totalWidth,
            totalHeight = totalHeight,
            fontMetrics = fontMetrics,
            fits = fits
        )
    }
}

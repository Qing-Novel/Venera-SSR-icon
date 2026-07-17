package com.manga.translate

import android.graphics.Color
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

internal object BubbleTextColorResolver {

    fun resolveContrastingTextColor(
        backgroundColor: Int,
        darkTextColor: Int,
        lightTextColor: Int = Color.WHITE
    ): Int {
        val opaqueBackground = Color.rgb(
            Color.red(backgroundColor),
            Color.green(backgroundColor),
            Color.blue(backgroundColor)
        )
        val darkContrast = contrastRatio(darkTextColor, opaqueBackground)
        val lightContrast = contrastRatio(lightTextColor, opaqueBackground)
        return if (lightContrast > darkContrast) lightTextColor else darkTextColor
    }

    private fun contrastRatio(foreground: Int, background: Int): Double {
        val foregroundLuminance = relativeLuminance(foreground)
        val backgroundLuminance = relativeLuminance(background)
        val lighter = max(foregroundLuminance, backgroundLuminance)
        val darker = min(foregroundLuminance, backgroundLuminance)
        return (lighter + 0.05) / (darker + 0.05)
    }

    private fun relativeLuminance(color: Int): Double {
        val r = linearize(Color.red(color) / 255.0)
        val g = linearize(Color.green(color) / 255.0)
        val b = linearize(Color.blue(color) / 255.0)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    private fun linearize(component: Double): Double {
        return if (component <= 0.03928) {
            component / 12.92
        } else {
            ((component + 0.055) / 1.055).pow(2.4)
        }
    }
}

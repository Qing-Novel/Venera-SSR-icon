package com.manga.translate

import android.graphics.Bitmap
import android.graphics.RectF

interface OcrEngine {
    fun recognize(bitmap: Bitmap): String

    fun recognizeWithScore(bitmap: Bitmap, rect: RectF? = null): OcrEngineResult {
        return OcrEngineResult(recognize(bitmap), 1.0f)
    }

    data class OcrEngineResult(
        val text: String,
        val score: Float
    )
}

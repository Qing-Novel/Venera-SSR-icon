package com.manga.translate

import android.content.Context

class KoreanOcr(
    context: Context,
    threadProfile: OnnxThreadProfile = OnnxThreadProfile.LIGHT,
    settingsStore: SettingsStore = SettingsStore(context.applicationContext)
) : PaddleOcrBase(
    context = context,
    modelAssetName = MODEL_ASSET,
    logTag = LOG_TAG,
    threadProfile = threadProfile,
    settingsStore = settingsStore
) {
    override fun trimLowConfidenceTail(tokens: List<OcrToken>): List<OcrToken> {
        if (tokens.isEmpty()) return tokens
        var endExclusive = tokens.size
        while (endExclusive > 0) {
            val token = tokens[endExclusive - 1]
            if (!shouldTrimTailToken(token)) break
            endExclusive--
        }
        return if (endExclusive == tokens.size) tokens else tokens.subList(0, endExclusive)
    }

    override fun shouldTrimTailToken(token: OcrToken): Boolean {
        if (token.score >= LOW_CONFIDENCE_TAIL_SCORE) return false
        return token.text.isNotBlank() && token.text.all(::isSuspiciousTailChar)
    }

    override fun isSuspiciousTailChar(char: Char): Boolean {
        if (char in '가'..'힣') return false
        if (char in 'ㄱ'..'ㅎ' || char in 'ㅏ'..'ㅣ') return false
        if (char.isLetterOrDigit()) return false
        return !char.isWhitespace()
    }

    override fun getDefaultCharset(): List<String> {
        val chars = mutableListOf("blank")

        // Last-resort fallback only. Keep it compact to avoid pathological CTC decode cost
        // when both session metadata and asset metadata extraction are unavailable.
        for (cp in 0x3131..0x318E) {
            chars.add(String(Character.toChars(cp)))
        }

        for (cp in 0x1100..0x11FF) {
            chars.add(String(Character.toChars(cp)))
        }

        for (i in '0'..'9') {
            chars.add(i.toString())
        }

        for (i in 'A'..'Z') {
            chars.add(i.toString())
        }

        for (i in 'a'..'z') {
            chars.add(i.toString())
        }

        val punctuation = "!\"#\$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        for (c in punctuation) {
            chars.add(c.toString())
        }

        AppLogger.log(LOG_TAG, "Using default charset with ${chars.size} characters")
        return chars
    }

    companion object {
        private const val MODEL_ASSET = "models/ocr/korean_PP-OCRv3_rec_infer.onnx"
        private const val LOG_TAG = "KoreanOcr"
        private const val LOW_CONFIDENCE_TAIL_SCORE = 0.65f
    }
}

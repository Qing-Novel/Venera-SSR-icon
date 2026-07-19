package com.manga.translate

import android.content.Context

class PPOcrV6SmallRec(
    context: Context,
    threadProfile: OnnxThreadProfile = OnnxThreadProfile.LIGHT,
    settingsStore: SettingsStore = SettingsStore(context.applicationContext)
) : PaddleOcrBase(
    context = context,
    modelAssetName = MODEL_ASSET,
    logTag = LOG_TAG,
    threadProfile = threadProfile,
    settingsStore = settingsStore,
    dictAssetName = DICT_ASSET,
    useXnnpack = false
) {
    // NOTE: This model is the PP-OCRv6 *multilingual* recogniser (output vocab = 18710).
    // The charset is loaded from the bundled dictionary asset, NOT from getDefaultCharset()
    // (which only returns a Latin subset and would mis-map every non-Latin index to garbage).
    // The asset below is PaddleOCR's ppocrv6_dict.txt (18708 glyphs + blank + space = 18710),
    // which is what the multilingual model was trained against. Swapping in a Chinese-only
    // dict (6622 glyphs) was the root cause of English/French/... bubbles being recognised as
    // garbled Chinese characters.
    override fun getDefaultCharset(): List<String> = emptyList()

    companion object {
        private const val MODEL_ASSET = "models/ocr/PP-OCRv6_small_rec.onnx"
        private const val DICT_ASSET = "models/ocr/ppocr_keys_v6_multilingual.txt"
        private const val LOG_TAG = "PPOcrV6SmallRec"
    }
}

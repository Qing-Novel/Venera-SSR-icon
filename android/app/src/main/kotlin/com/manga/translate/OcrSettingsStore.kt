package com.manga.translate

internal class OcrSettingsStore(
    private val storage: SettingsStoreStorage
) {
    fun loadOcrApiSettings(): OcrApiSettings {
        val useLocal = storage.prefs.getBoolean(SettingsStore.KEY_OCR_USE_LOCAL, true)
        val url = storage.prefs.getString(
            SettingsStore.KEY_OCR_API_URL,
            SettingsStore.DEFAULT_OCR_API_URL
        ) ?: SettingsStore.DEFAULT_OCR_API_URL
        val key = storage.prefs.getString(SettingsStore.KEY_OCR_API_KEY, "") ?: ""
        val model = storage.prefs.getString(
            SettingsStore.KEY_OCR_MODEL_NAME,
            SettingsStore.DEFAULT_OCR_MODEL_NAME
        ) ?: SettingsStore.DEFAULT_OCR_MODEL_NAME
        val timeoutSeconds = storage.prefs.getInt(
            SettingsStore.KEY_OCR_API_TIMEOUT_SECONDS,
            SettingsStore.DEFAULT_OCR_API_TIMEOUT_SECONDS
        ).coerceIn(
            SettingsStore.MIN_OCR_API_TIMEOUT_SECONDS,
            SettingsStore.MAX_OCR_API_TIMEOUT_SECONDS
        )
        return OcrApiSettings(
            useLocalOcr = useLocal,
            japaneseLocalOcrEngine = JapaneseLocalOcrEngine.fromPref(
                storage.prefs.getString(SettingsStore.KEY_JAPANESE_LOCAL_OCR_ENGINE, null)
            ),
            apiUrl = url,
            apiKey = key,
            modelName = model,
            timeoutSeconds = timeoutSeconds,
            apiOcrConcurrencyLimit = storage.prefs.getInt(
                SettingsStore.KEY_OCR_API_CONCURRENCY,
                SettingsStore.DEFAULT_OCR_API_CONCURRENCY
            ).coerceIn(
                SettingsStore.MIN_OCR_API_CONCURRENCY,
                SettingsStore.MAX_OCR_API_CONCURRENCY
            ),
            localOcrConcurrencyLimit = storage.prefs.getInt(
                SettingsStore.KEY_LOCAL_OCR_CONCURRENCY,
                SettingsStore.DEFAULT_LOCAL_OCR_CONCURRENCY
            ).coerceIn(
                SettingsStore.MIN_LOCAL_OCR_CONCURRENCY,
                SettingsStore.MAX_LOCAL_OCR_CONCURRENCY
            ),
            ocrApiFormat = OcrApiFormat.fromPref(
                storage.prefs.getString(SettingsStore.KEY_OCR_API_FORMAT, null)
            ),
            secretKey = storage.prefs.getString(SettingsStore.KEY_OCR_SECRET_KEY, "") ?: ""
        )
    }

    fun saveOcrApiSettings(settings: OcrApiSettings) {
        val normalizedTimeout = settings.timeoutSeconds.coerceIn(
            SettingsStore.MIN_OCR_API_TIMEOUT_SECONDS,
            SettingsStore.MAX_OCR_API_TIMEOUT_SECONDS
        )
        val normalizedConcurrency = settings.apiOcrConcurrencyLimit.coerceIn(
            SettingsStore.MIN_OCR_API_CONCURRENCY,
            SettingsStore.MAX_OCR_API_CONCURRENCY
        )
        val normalizedLocalConcurrency = settings.localOcrConcurrencyLimit.coerceIn(
            SettingsStore.MIN_LOCAL_OCR_CONCURRENCY,
            SettingsStore.MAX_LOCAL_OCR_CONCURRENCY
        )
        storage.editSettings(
            setOf(
                SettingsStore.KEY_OCR_USE_LOCAL,
                SettingsStore.KEY_JAPANESE_LOCAL_OCR_ENGINE,
                SettingsStore.KEY_OCR_API_URL,
                SettingsStore.KEY_OCR_API_KEY,
                SettingsStore.KEY_OCR_MODEL_NAME,
                SettingsStore.KEY_OCR_API_TIMEOUT_SECONDS,
                SettingsStore.KEY_OCR_API_CONCURRENCY,
                SettingsStore.KEY_LOCAL_OCR_CONCURRENCY,
                SettingsStore.KEY_OCR_API_FORMAT,
                SettingsStore.KEY_OCR_SECRET_KEY
            )
        ) {
            putBoolean(SettingsStore.KEY_OCR_USE_LOCAL, settings.useLocalOcr)
                .putString(
                    SettingsStore.KEY_JAPANESE_LOCAL_OCR_ENGINE,
                    settings.japaneseLocalOcrEngine.prefValue
                )
                .putString(SettingsStore.KEY_OCR_API_URL, settings.apiUrl)
                .putString(SettingsStore.KEY_OCR_API_KEY, settings.apiKey)
                .putString(SettingsStore.KEY_OCR_MODEL_NAME, settings.modelName)
                .putInt(SettingsStore.KEY_OCR_API_TIMEOUT_SECONDS, normalizedTimeout)
                .putInt(SettingsStore.KEY_OCR_API_CONCURRENCY, normalizedConcurrency)
                .putInt(SettingsStore.KEY_LOCAL_OCR_CONCURRENCY, normalizedLocalConcurrency)
                .putString(
                    SettingsStore.KEY_OCR_API_FORMAT,
                    settings.ocrApiFormat.prefValue
                )
                .putString(SettingsStore.KEY_OCR_SECRET_KEY, settings.secretKey)
        }
    }
}

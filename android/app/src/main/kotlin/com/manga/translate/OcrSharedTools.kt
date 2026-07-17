package com.manga.translate

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import java.text.Normalizer
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.channels.Channel

class OcrEngineRegistry(
    context: Context,
    private val settingsStore: SettingsStore = SettingsStore(context.applicationContext)
) {
    private val appContext = context.applicationContext
    private var mangaOcrMobile: MangaOcrMobile? = null
    private var mangaOcrMobileInitFailed = false
    private var ppOcrV6SmallRec: PPOcrV6SmallRec? = null
    private var koreanOcr: KoreanOcr? = null
    private var englishLineDetector: EnglishLineDetector? = null

    @Volatile private var jaPoolClosed = false
    private var jaPool: Channel<MangaOcrMobile>? = null
    private val jaActiveBorrows = AtomicInteger(0)

    @Synchronized
    fun getMangaOcrMobile(logTag: String): MangaOcrMobile? {
        if (mangaOcrMobile != null) return mangaOcrMobile
        if (mangaOcrMobileInitFailed) return null
        return try {
            MangaOcrMobile(appContext, settingsStore = settingsStore).also {
                mangaOcrMobile = it
            }
        } catch (e: Exception) {
            mangaOcrMobileInitFailed = true
            AppLogger.log(logTag, "Failed to init mobile OCR", e)
            null
        }
    }

    @Synchronized
    fun getPpOcrV6SmallRec(logTag: String): PPOcrV6SmallRec? {
        if (ppOcrV6SmallRec != null) return ppOcrV6SmallRec
        return try {
            PPOcrV6SmallRec(appContext, settingsStore = settingsStore).also { ppOcrV6SmallRec = it }
        } catch (e: Exception) {
            AppLogger.log(logTag, "Failed to init PP-OCRv6_small_rec", e)
            null
        }
    }

    @Synchronized
    fun getKoreanOcr(logTag: String): KoreanOcr? {
        if (koreanOcr != null) return koreanOcr
        return try {
            KoreanOcr(appContext, settingsStore = settingsStore).also { koreanOcr = it }
        } catch (e: Exception) {
            AppLogger.log(logTag, "Failed to init Korean OCR", e)
            null
        }
    }

    @Synchronized
    fun getEnglishLineDetector(logTag: String): EnglishLineDetector? {
        if (englishLineDetector != null) return englishLineDetector
        return try {
            EnglishLineDetector(appContext, settingsStore = settingsStore).also {
                englishLineDetector = it
            }
        } catch (e: Exception) {
            AppLogger.log(logTag, "Failed to init English line detector", e)
            null
        }
    }

    @OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
    @Synchronized
    fun ensureJaPool(logTag: String, concurrencyOverride: Int = 0): Channel<MangaOcrMobile>? {
        if (jaPoolClosed) return null
        jaPool?.let { return it }
        val concurrency = LocalOcrConcurrency.resolve(concurrencyOverride)
        if (concurrency <= 1) return null
        val channel = Channel<MangaOcrMobile>(capacity = concurrency)
        repeat(concurrency) {
            try {
                channel.trySend(MangaOcrMobile(appContext, settingsStore = settingsStore, numThreads = 1))
            } catch (e: Exception) {
                AppLogger.log(logTag, "Failed to create JA pool instance", e)
            }
        }
        if (channel.isEmpty) return null
        jaPool = channel
        return channel
    }

    suspend fun borrowJa(logTag: String): MangaOcrMobile? {
        val pool = synchronized(this) { if (!jaPoolClosed) jaPool else null } ?: return null
        val engine = try {
            pool.receive()
        } catch (_: Exception) {
            null
        } ?: return null
        if (jaPoolClosed) {
            engine.close()
            return null
        }
        jaActiveBorrows.incrementAndGet()
        return engine
    }

    fun returnJa(engine: MangaOcrMobile) {
        jaActiveBorrows.decrementAndGet()
        val pool = synchronized(this) { jaPool }
        if (jaPoolClosed || pool == null) {
            engine.close()
        } else {
            val sent = pool.trySend(engine)
            if (sent.isFailure) {
                engine.close()
            }
        }
    }

    @Synchronized
    fun releaseLoadedEngines() {
        val hadLoadedEngines = mangaOcrMobile != null ||
            ppOcrV6SmallRec != null ||
            koreanOcr != null ||
            englishLineDetector != null ||
            jaPool != null
        mangaOcrMobile = null
        ppOcrV6SmallRec = null
        koreanOcr = null
        englishLineDetector = null
        jaPoolClosed = true
        val pool = jaPool
        jaPool = null
        if (pool != null) {
            // drain idle instances; instances still borrowed will be closed on returnJa
            var engine = pool.tryReceive().getOrNull()
            while (engine != null) {
                engine.close()
                engine = pool.tryReceive().getOrNull()
            }
            pool.close()
        }
        if (hadLoadedEngines) {
            AppLogger.log("OcrEngineRegistry", "Released loaded OCR engine references")
        }
    }
}

class BubbleTextRecognizer(
    private val llmClient: LlmGateway,
    private val engineRegistry: OcrEngineRegistry,
    private val settingsStore: SettingsStore
) {
    fun getLocalOcrEngine(
        language: TranslationLanguage,
        logTag: String
    ): OcrEngine? {
        return when (language) {
            TranslationLanguage.JA_TO_ZH -> when (
                settingsStore.loadOcrApiSettings().japaneseLocalOcrEngine
            ) {
                JapaneseLocalOcrEngine.MANGA_OCR_MOBILE -> engineRegistry.getMangaOcrMobile(logTag)
            }
            TranslationLanguage.EN_TO_ZH,
            TranslationLanguage.ZH_HANS_TO_TARGET,
            TranslationLanguage.ZH_HANT_TO_TARGET,
            TranslationLanguage.CHN_ENG_TO_ZH -> engineRegistry.getPpOcrV6SmallRec(logTag)
            TranslationLanguage.KO_TO_ZH -> engineRegistry.getKoreanOcr(logTag)
            TranslationLanguage.FR_TO_ZH,
            TranslationLanguage.ES_TO_ZH,
            TranslationLanguage.PT_TO_ZH,
            TranslationLanguage.DE_TO_ZH,
            TranslationLanguage.IT_TO_ZH -> engineRegistry.getPpOcrV6SmallRec(logTag)
            TranslationLanguage.RU_TO_ZH -> null
        }
    }

    fun detectRecognizedLines(
        source: Bitmap,
        language: TranslationLanguage,
        logTag: String
    ): List<EnglishLine> {
        val lineDetector = engineRegistry.getEnglishLineDetector(logTag) ?: return emptyList()
        val lineRects = lineDetector.detectLines(source)
        return when (language) {
            TranslationLanguage.EN_TO_ZH,
            TranslationLanguage.FR_TO_ZH,
            TranslationLanguage.ES_TO_ZH,
            TranslationLanguage.PT_TO_ZH,
            TranslationLanguage.DE_TO_ZH,
            TranslationLanguage.IT_TO_ZH -> {
                val engine = engineRegistry.getPpOcrV6SmallRec(logTag) ?: return emptyList()
                recognizeEnglishLines(source, lineRects, engine)
            }

            TranslationLanguage.KO_TO_ZH -> {
                val engine = engineRegistry.getKoreanOcr(logTag) ?: return emptyList()
                recognizeKoreanLines(source, lineRects, engine)
            }

            TranslationLanguage.JA_TO_ZH,
            TranslationLanguage.ZH_HANS_TO_TARGET,
            TranslationLanguage.ZH_HANT_TO_TARGET,
            TranslationLanguage.CHN_ENG_TO_ZH,
            TranslationLanguage.RU_TO_ZH -> emptyList()
        }
    }

    suspend fun recognizeRegion(
        source: Bitmap,
        rect: RectF,
        language: TranslationLanguage,
        useLocalOcr: Boolean,
        logTag: String
    ): OcrRecognitionResult {
        val crop = cropBitmap(source, rect)?.let { PipelineBitmapDecoder.scaleDownIfNeeded(it) }
            ?: return OcrRecognitionResult.Success("")
        return try {
            recognizeCrop(crop, language, useLocalOcr, logTag)
        } finally {
            crop.recycleSafely()
        }
    }

    internal suspend fun recognizeRegion(
        cropSource: BitmapCropSource,
        rect: RectF,
        language: TranslationLanguage,
        useLocalOcr: Boolean,
        logTag: String
    ): OcrRecognitionResult {
        val clamped = PipelineBitmapDecoder.clampRect(rect, cropSource.width, cropSource.height)
            ?: return OcrRecognitionResult.Success("")
        val crop = cropSource.decodeRegion(clamped) ?: return OcrRecognitionResult.Success("")
        return try {
            recognizeCrop(crop, language, useLocalOcr, logTag)
        } finally {
            crop.recycleSafely()
        }
    }

    suspend fun recognizeCrop(
        crop: Bitmap,
        language: TranslationLanguage,
        useLocalOcr: Boolean,
        logTag: String
    ): OcrRecognitionResult {
        val resolvedUseLocalOcr = useLocalOcr && language.supportsLocalOcr()
        val rawText = if (!resolvedUseLocalOcr) {
            try {
                llmClient.recognizeImageText(crop, language)?.trim().orEmpty()
            } catch (e: Exception) {
                AppLogger.log(logTag, "API OCR failed", e)
                return OcrRecognitionResult.Failure(e)
            }
        } else when (language) {
            TranslationLanguage.JA_TO_ZH -> {
                when (settingsStore.loadOcrApiSettings().japaneseLocalOcrEngine) {
                    JapaneseLocalOcrEngine.MANGA_OCR_MOBILE -> {
                        val engine = engineRegistry.getMangaOcrMobile(logTag)
                            ?: return OcrRecognitionResult.Failure(
                                IllegalStateException("Japanese OCR engine unavailable")
                            )
                        engine.recognize(crop).trim()
                    }
                }
            }

            TranslationLanguage.EN_TO_ZH,
            TranslationLanguage.FR_TO_ZH,
            TranslationLanguage.ES_TO_ZH,
            TranslationLanguage.PT_TO_ZH,
            TranslationLanguage.DE_TO_ZH,
            TranslationLanguage.IT_TO_ZH -> {
                val engine = engineRegistry.getPpOcrV6SmallRec(logTag)
                    ?: return OcrRecognitionResult.Failure(
                        IllegalStateException("PP-OCRv6_small_rec engine unavailable")
                    )
                val lineDetector = engineRegistry.getEnglishLineDetector(logTag)
                val lineRects = lineDetector?.detectLines(crop).orEmpty()
                val lines = recognizeEnglishLines(crop, lineRects, engine)
                if (lines.isEmpty()) {
                    engine.recognize(crop).trim()
                } else {
                    lines.joinToString("\n") { it.text }
                }
            }

            TranslationLanguage.ZH_HANS_TO_TARGET,
            TranslationLanguage.ZH_HANT_TO_TARGET,
            TranslationLanguage.CHN_ENG_TO_ZH -> {
                val engine = engineRegistry.getPpOcrV6SmallRec(logTag)
                    ?: return OcrRecognitionResult.Failure(
                        IllegalStateException("PP-OCRv6_small_rec engine unavailable")
                    )
                engine.recognize(crop).trim()
            }

            TranslationLanguage.KO_TO_ZH -> {
                val engine = engineRegistry.getKoreanOcr(logTag)
                    ?: return OcrRecognitionResult.Failure(
                        IllegalStateException("Korean OCR engine unavailable")
                    )
                val lineDetector = engineRegistry.getEnglishLineDetector(logTag)
                val lineRects = lineDetector?.detectLines(crop).orEmpty()
                val lines = recognizeKoreanLines(crop, lineRects, engine)
                if (lines.isEmpty()) {
                    val decoded = engine.recognizeWithScore(crop)
                    decoded.text.trim().takeIf {
                        decoded.score >= DEFAULT_KO_MIN_LINE_SCORE && it.isNotBlank()
                    }.orEmpty()
                } else {
                    lines.joinToString("\n") { it.text }
                }
            }

            TranslationLanguage.RU_TO_ZH -> return OcrRecognitionResult.Failure(
                IllegalStateException("Local OCR unsupported for ${language.name}")
            )
        }
        return OcrRecognitionResult.Success(OcrTextSanitizer.sanitize(rawText, language, logTag))
    }

    internal suspend fun sanitizeJaCrop(
        engine: MangaOcrMobile,
        crop: Bitmap,
        language: TranslationLanguage,
        logTag: String
    ): String {
        val rawText = engine.recognize(crop).trim()
        return OcrTextSanitizer.sanitize(rawText, language, logTag)
    }
}

const val DEFAULT_EN_MIN_LINE_SCORE = 0.5f

data class EnglishLine(
    val rect: RectF,
    val text: String
)

fun normalizeOcrText(text: String, language: TranslationLanguage): String {
    val sanitized = OcrTextSanitizer.sanitize(text, language)
    if (!language.usesLatinOcr() && language != TranslationLanguage.KO_TO_ZH) {
        return sanitized
    }
    return sanitized.replace('\r', ' ')
        .replace('\n', ' ')
        .replace(Regex("\\s+"), " ")
        .trim()
}

object OcrTextSanitizer {
    fun sanitize(
        text: String,
        language: TranslationLanguage,
        logTag: String? = null
    ): String {
        if (text.isBlank()) return ""
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
        val cleaned = removeInvisibleNoise(normalized)
            .replace(Regex("[ \\t\\x0B\\f]+"), " ")
            .replace(Regex(" *\\n+ *"), "\n")
            .trim()
        if (cleaned.isBlank()) return ""
        if (!containsText(cleaned)) {
            logTag?.let {
                AppLogger.log(it, "Drop OCR non-text bubble language=${language.name}, text=${cleaned.take(80)}")
            }
            return ""
        }
        return cleaned
    }

    private fun removeInvisibleNoise(text: String): String {
        val builder = StringBuilder(text.length)
        var index = 0
        while (index < text.length) {
            val codePoint = text.codePointAt(index)
            index += Character.charCount(codePoint)
            when {
                codePoint == '\r'.code || codePoint == '\n'.code -> builder.append('\n')
                codePoint == '\t'.code -> builder.append(' ')
                shouldDropCodePoint(codePoint) -> Unit
                else -> builder.appendCodePoint(codePoint)
            }
        }
        return builder.toString()
    }

    private fun shouldDropCodePoint(codePoint: Int): Boolean {
        return when (Character.getType(codePoint)) {
            Character.CONTROL.toInt(),
            Character.FORMAT.toInt(),
            Character.SURROGATE.toInt(),
            Character.PRIVATE_USE.toInt(),
            Character.UNASSIGNED.toInt() -> true
            else -> false
        }
    }

    private fun containsText(text: String): Boolean {
        var index = 0
        while (index < text.length) {
            val codePoint = text.codePointAt(index)
            index += Character.charCount(codePoint)
            if (isTextCodePoint(codePoint)) return true
        }
        return false
    }

    private fun isTextCodePoint(codePoint: Int): Boolean {
        return when (Character.getType(codePoint)) {
            Character.UPPERCASE_LETTER.toInt(),
            Character.LOWERCASE_LETTER.toInt(),
            Character.TITLECASE_LETTER.toInt(),
            Character.MODIFIER_LETTER.toInt(),
            Character.OTHER_LETTER.toInt() -> true
            else -> false
        }
    }
}

fun cropBitmap(source: Bitmap, rect: RectF): Bitmap? {
    val left = rect.left.toInt().coerceIn(0, source.width - 1)
    val top = rect.top.toInt().coerceIn(0, source.height - 1)
    val right = rect.right.toInt().coerceIn(1, source.width)
    val bottom = rect.bottom.toInt().coerceIn(1, source.height)
    val width = right - left
    val height = bottom - top
    if (width <= 0 || height <= 0) return null
    return Bitmap.createBitmap(source, left, top, width, height)
}

fun Bitmap?.recycleSafely() {
    if (this != null && !isRecycled) {
        recycle()
    }
}

inline fun <T> withBitmapCrop(
    source: Bitmap,
    rect: RectF,
    block: (Bitmap) -> T
): T? {
    val crop = cropBitmap(source, rect) ?: return null
    return try {
        block(crop)
    } finally {
        crop.recycleSafely()
    }
}

fun recognizeEnglishLines(
    source: Bitmap,
    lineRects: List<RectF>,
    ocrEngine: OcrEngine,
    minLineScore: Float = DEFAULT_EN_MIN_LINE_SCORE
): List<EnglishLine> {
    if (lineRects.isEmpty()) return emptyList()
    val results = ArrayList<EnglishLine>(lineRects.size)
    for (rect in lineRects) {
        val decoded = ocrEngine.recognizeWithScore(source, rect)
        val text = decoded.text.trim()
        if (decoded.score >= minLineScore && text.isNotBlank()) {
            results.add(EnglishLine(rect, text))
        }
    }
    return results
}

fun recognizeKoreanLines(
    source: Bitmap,
    lineRects: List<RectF>,
    ocrEngine: OcrEngine,
    minLineScore: Float = DEFAULT_KO_MIN_LINE_SCORE
): List<EnglishLine> {
    if (lineRects.isEmpty()) return emptyList()
    val results = ArrayList<EnglishLine>(lineRects.size)
    for (rect in lineRects) {
        val decoded = ocrEngine.recognizeWithScore(source, rect)
        val text = decoded.text.trim()
        if (decoded.score >= minLineScore && text.isNotBlank()) {
            results.add(EnglishLine(rect, text))
        }
    }
    return results
}

const val DEFAULT_KO_MIN_LINE_SCORE = 0.65f

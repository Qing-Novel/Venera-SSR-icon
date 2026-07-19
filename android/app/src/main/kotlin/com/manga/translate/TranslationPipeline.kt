package com.manga.translate
import com.github.kiastr.venera_ssr.R

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import java.io.File

internal class TranslationPipeline(
    context: Context,
    private val llmClient: LlmGateway = LlmClient(context.applicationContext),
    private val settingsStore: SettingsStore = SettingsStore(context.applicationContext),
    private val store: TranslationStore = TranslationStore(),
    private val ocrStore: OcrStore = OcrStore(),
    private val ocrEngineRegistry: OcrEngineRegistry =
        OcrEngineRegistry(context.applicationContext, settingsStore),
    private val bubbleTextRecognizer: BubbleTextRecognizer =
        BubbleTextRecognizer(llmClient, ocrEngineRegistry, settingsStore),
    private val pageRegionDetector: PageRegionDetector =
        PageRegionDetector(context.applicationContext, settingsStore),
    private var marianMtEngine: MarianMtEngine? = null,
    private val textBubbleTranslationCoordinator: TextBubbleTranslationCoordinator
) {
    private val appContext = context.applicationContext
    init {
        initMarianMtEngine()
        // Construct the coordinator AFTER marianMtEngine is initialised, otherwise
        // it captures the still-null field and the local (MarianMT) engine is
        // never used — every translation would wrongly fall through to the LLM
        // branch (which then fails on the missing prompts/llm_prompts.json asset).
        textBubbleTranslationCoordinator = TextBubbleTranslationCoordinator(
            llmClient = llmClient,
            localTranslationEngine = marianMtEngine
        )
    }

    private fun initMarianMtEngine() {
        val modelDir = settingsStore.loadLocalTranslationModelDir()
        if (modelDir.isNotBlank()) {
            marianMtEngine = MarianMtEngine(appContext, File(modelDir))
        }
    }


    suspend fun translateImage(
        imageFile: File,
        glossary: MutableMap<String, String>,
        forceOcr: Boolean,
        language: TranslationLanguage = TranslationLanguage.JA_TO_ZH,
        providerContext: PageTranslationProviderContext? = null,
        onProgress: (String) -> Unit
        ): TranslationResult? = withContext(Dispatchers.Default) {
        if (marianMtEngine == null) {
            initMarianMtEngine()
        }

        val resolvedApiSettings = providerContext?.apiSettings
        if (marianMtEngine?.isAvailable() != true && !llmClient.isConfigured(resolvedApiSettings)) {
            onProgress(appContext.getString(R.string.missing_api_settings))
            AppLogger.log("Pipeline", "Missing API settings")
            return@withContext null
        }
        val page = ocrImage(imageFile, forceOcr, language, onProgress) ?: return@withContext null
        translateStandardPage(
            page = page,
            imageFile = imageFile,
            glossary = glossary,
            language = language,
            providerContext = providerContext,
            onProgress = onProgress
        )
    }

    suspend fun translateStandardPage(
        page: PageOcrResult,
        imageFile: File,
        glossary: MutableMap<String, String>,
        language: TranslationLanguage = TranslationLanguage.JA_TO_ZH,
        providerContext: PageTranslationProviderContext? = null,
        onProgress: (String) -> Unit
    ): TranslationResult? = withContext(Dispatchers.Default) {
        val resolvedApiSettings = providerContext?.apiSettings
        val metadata = buildTranslationMetadata(
            imageFile = imageFile,
            language = language,
            mode = TranslationMetadata.MODE_STANDARD,
            promptAsset = STANDARD_PROMPT_ASSET,
            ocrCacheMode = page.cacheMode,
            providerContext = providerContext
        )
        AppLogger.log("Pipeline", "Translate image ${imageFile.name}")
        val ocrPage = page.withRecognizedTextBubblesOnly("Pipeline")
        val translatable = ocrPage.bubbles
        if (translatable.isEmpty()) {
            val emptyTranslations = ocrPage.bubbles.map {
                BubbleTranslation.pending(it.id, it.rect, "", it.source, it.maskContour)
            }
            return@withContext TranslationResult(
                imageFile.name,
                ocrPage.width,
                ocrPage.height,
                emptyTranslations,
                metadata.copy(status = PageTranslationStatus.SUCCESS)
            )
        }
        onProgress(appContext.getString(R.string.translating_bubbles))
        val promptAsset = STANDARD_PROMPT_ASSET
        val translatedBatch = try {
            val translated = executeWithModelResponseRetries("Pipeline") {
                textBubbleTranslationCoordinator.translateBubbles(
                    bubbles = translatable.map {
                        BubbleTranslation.pending(
                            id = it.id,
                            rect = it.rect,
                            originalText = it.text,
                            source = it.source,
                            maskContour = it.maskContour
                        )
                    },
                    glossary = glossary,
                    promptAsset = promptAsset,
                    apiSettings = resolvedApiSettings,
                    language = language,
                    logTag = "Pipeline",
                    translationMode = "standard"
                )
            } ?: return@withContext null
            if (translated.glossaryUsed.isNotEmpty()) {
                glossary.putAll(translated.glossaryUsed)
            }
            translated
        } catch (e: LlmResponseException) {
            throw e.withPageName(imageFile.name)
        }
        val translationMap = translatedBatch.bubbles.associateBy { it.id }
        val bubbles = ocrPage.bubbles.filterNot { it.id in translatedBatch.removedBubbleIds }.map { bubble ->
            translationMap[bubble.id] ?: BubbleTranslation.pending(
                id = bubble.id,
                rect = bubble.rect,
                originalText = bubble.text,
                source = bubble.source,
                maskContour = bubble.maskContour
            )
        }
        AppLogger.log("Pipeline", "Translation finished for ${imageFile.name}")
        val resultBase = TranslationResult(imageFile.name, ocrPage.width, ocrPage.height, bubbles, metadata)
        resultBase.copy(metadata = metadata.copy(status = resultBase.deriveStatus()))
    }

    suspend fun ocrImage(
        imageFile: File,
        forceOcr: Boolean,
        language: TranslationLanguage = TranslationLanguage.JA_TO_ZH,
        onProgress: (String) -> Unit
    ): PageOcrResult? = withContext(Dispatchers.Default) {
        val ocrSettings = settingsStore.loadOcrApiSettings()
        val resolvedLanguage = TranslationLanguage.resolveForOcr(language, ocrSettings.useLocalOcr)
        val effectiveUseLocalOcr = ocrSettings.useLocalOcr && resolvedLanguage.supportsLocalOcr()
        val isBaiduFullPage = !effectiveUseLocalOcr && ocrSettings.ocrApiFormat == OcrApiFormat.BAIDU_AI
        val cacheMode = if (isBaiduFullPage) {
            buildBaiduFullPageOcrCacheMode(imageFile)
        } else {
            buildOcrCacheMode(imageFile, effectiveUseLocalOcr, resolvedLanguage)
        }
        val expectedMetadata = if (isBaiduFullPage) {
            buildBaiduFullPageOcrMetadata(imageFile, language, cacheMode)
        } else {
            buildOcrMetadata(imageFile, language, ocrSettings, cacheMode)
        }
        if (!forceOcr) {
            val cached = ocrStore.load(imageFile, expectedMetadata = expectedMetadata)
            if (cached != null) {
                AppLogger.log("Pipeline", "Reuse OCR for ${imageFile.name}")
                return@withContext cached
            }
        }
        val useLocalOcr = effectiveUseLocalOcr
        val ocrEngine: OcrEngine? = if (useLocalOcr) {
            bubbleTextRecognizer.getLocalOcrEngine(resolvedLanguage, "Pipeline")
        } else {
            null
        }
        if (!useLocalOcr && !llmClient.isOcrConfigured()) {
            onProgress(appContext.getString(R.string.missing_ocr_api_settings))
            AppLogger.log("Pipeline", "Missing OCR API settings")
            return@withContext null
        }
        if (useLocalOcr && ocrEngine == null) {
            return@withContext null
        }
        PipelineBitmapDecoder.openCropSource(imageFile)?.use { cropSource ->
            onProgress(appContext.getString(R.string.detecting_bubbles))
            val pageRegions = pageRegionDetector.detect(
                cropSource = cropSource,
                pageWidth = cropSource.width,
                pageHeight = cropSource.height,
                logTag = "Pipeline"
            ) ?: return@withContext null
            val regions = pageRegions.regions
            AppLogger.log("Pipeline", "Detected ${regions.size} regions in ${imageFile.name}")
            if (regions.isEmpty()) {
                val emptyResult = PageOcrResult(
                    imageFile,
                    pageRegions.width,
                    pageRegions.height,
                    emptyList(),
                    cacheMode,
                    expectedMetadata
                )
                ocrStore.save(imageFile, emptyResult)
                return@withContext emptyResult
            }
            val bubbles: List<OcrBubble> = if (isBaiduFullPage) {
                recognizeFullPageBaiduAndMatch(
                    cropSource = cropSource,
                    pageRegions = pageRegions,
                    language = resolvedLanguage
                )
            } else {
                onProgress(
                    appContext.getString(R.string.floating_progress_recognizing, regions.size)
                )
                recognizeBubblesIndividually(
                    cropSource = cropSource,
                    regions = regions,
                    language = resolvedLanguage,
                    useLocalOcr = useLocalOcr,
                    ocrSettings = ocrSettings,
                    ocrEngine = ocrEngine
                )
            }
            val mergedBubbles = RectGeometryDeduplicator.mergeShortTextDetectorOcrBubbles(
                bubbles = bubbles,
                imageWidth = pageRegions.width,
                imageHeight = pageRegions.height,
                maxMergedHeight = if (shouldUseLongImageTiling(pageRegions.width, pageRegions.height)) {
                    longImageMaxRegionHeight(pageRegions.width, pageRegions.height)
                } else {
                    null
                }
            )
            if (mergedBubbles.size < bubbles.size) {
                AppLogger.log(
                    "Pipeline",
                    "Merged short text detector OCR bubbles: ${bubbles.size} -> ${mergedBubbles.size}"
                )
            }
            val result = PageOcrResult(
                imageFile,
                pageRegions.width,
                pageRegions.height,
                mergedBubbles,
                cacheMode,
                expectedMetadata
            )
            ocrStore.save(imageFile, result)
            result
        } ?: run {
            AppLogger.log("Pipeline", "Failed to open crop source for ${imageFile.name}")
            null
        }
    }

    suspend fun translateFullPage(
        page: PageOcrResult,
        glossary: Map<String, String>,
        promptAsset: String,
        language: TranslationLanguage = TranslationLanguage.JA_TO_ZH,
        providerContext: PageTranslationProviderContext? = null,
        onProgress: (String) -> Unit
    ): TranslationResult? = withContext(Dispatchers.Default) {
        val metadata = buildTranslationMetadata(
            imageFile = page.imageFile,
            language = language,
            mode = TranslationMetadata.MODE_FULL_PAGE,
            promptAsset = promptAsset,
            ocrCacheMode = page.cacheMode,
            providerContext = providerContext
        )
        val ocrPage = page.withRecognizedTextBubblesOnly("Pipeline")
        val translatable = ocrPage.bubbles
        if (translatable.isEmpty()) {
            val emptyTranslations = ocrPage.bubbles.map {
                BubbleTranslation.pending(it.id, it.rect, "", it.source, it.maskContour)
            }
            return@withContext TranslationResult(
                ocrPage.imageFile.name,
                ocrPage.width,
                ocrPage.height,
                emptyTranslations,
                metadata.copy(status = PageTranslationStatus.SUCCESS)
            )
        }
        onProgress(appContext.getString(R.string.translating_bubbles))
        val translatedBatch = try {
            val translated = executeWithModelResponseRetries("Pipeline") {
                textBubbleTranslationCoordinator.translateBubbles(
                    bubbles = translatable.map {
                        BubbleTranslation.pending(
                            id = it.id,
                            rect = it.rect,
                            originalText = it.text,
                            source = it.source,
                            maskContour = it.maskContour
                        )
                    },
                    glossary = glossary,
                    promptAsset = promptAsset,
                    apiSettings = providerContext?.apiSettings,
                    language = language,
                    logTag = "Pipeline",
                    translationMode = "full_page"
                )
            } ?: return@withContext null
            translated
        } catch (e: LlmResponseException) {
            throw e.withPageName(ocrPage.imageFile.name)
        }
        val translationMap = translatedBatch.bubbles.associateBy { it.id }
        val bubbles = ocrPage.bubbles.filterNot { it.id in translatedBatch.removedBubbleIds }.map { bubble ->
            translationMap[bubble.id] ?: BubbleTranslation.pending(
                id = bubble.id,
                rect = bubble.rect,
                originalText = bubble.text,
                source = bubble.source,
                maskContour = bubble.maskContour
            )
        }
        val resultBase = TranslationResult(ocrPage.imageFile.name, ocrPage.width, ocrPage.height, bubbles, metadata)
        resultBase.copy(metadata = metadata.copy(status = resultBase.deriveStatus()))
    }


    fun hasValidTranslation(
        imageFile: File,
        fullTranslate: Boolean,
        useVlDirectTranslate: Boolean,
        language: TranslationLanguage
    ): Boolean {
        val translation = loadValidTranslation(
            imageFile = imageFile,
            fullTranslate = fullTranslate,
            useVlDirectTranslate = useVlDirectTranslate,
            language = language
        ) ?: return false
        if (translation.metadata.isManual()) return true
        return translation.metadata.status == PageTranslationStatus.SUCCESS
    }

    fun loadValidTranslation(
        imageFile: File,
        fullTranslate: Boolean,
        useVlDirectTranslate: Boolean,
        language: TranslationLanguage
    ): TranslationResult? {
        val expected = buildExpectedTranslationMetadata(
            imageFile = imageFile,
            fullTranslate = fullTranslate,
            useVlDirectTranslate = useVlDirectTranslate,
            language = language
        )
        return store.load(imageFile, expectedMetadata = expected)
    }

    fun loadAnyTranslation(imageFile: File): TranslationResult? {
        return store.load(imageFile)
    }

    fun saveResult(imageFile: File, result: TranslationResult): File {
        val saved = store.save(imageFile, result)
        if (result.metadata.status == PageTranslationStatus.SUCCESS) {
            val ocrFile = ocrStore.ocrFileFor(imageFile)
            if (ocrFile.exists()) {
                ocrFile.delete()
            }
        }
        return saved
    }

    suspend fun buildBlankTranslationResult(
        imageFile: File,
        forceOcr: Boolean,
        language: TranslationLanguage = TranslationLanguage.JA_TO_ZH
    ): TranslationResult? = withContext(Dispatchers.Default) {
        val page = ocrImage(imageFile, forceOcr, language) { } ?: return@withContext null
        buildBlankTranslationResult(
            page = page,
            mode = TranslationMetadata.MODE_STANDARD,
            promptAsset = STANDARD_PROMPT_ASSET,
            language = language
        )
    }

    fun buildBlankTranslationResult(
        page: PageOcrResult,
        mode: String,
        promptAsset: String,
        language: TranslationLanguage = TranslationLanguage.JA_TO_ZH
    ): TranslationResult {
        val metadata = buildTranslationMetadata(
            imageFile = page.imageFile,
            language = language,
            mode = mode,
            promptAsset = promptAsset,
            ocrCacheMode = page.cacheMode,
            providerContext = null
        )
        val ocrPage = page.withRecognizedTextBubblesOnly("Pipeline")
        val bubbles = ocrPage.bubbles.map { bubble ->
            BubbleTranslation.pending(bubble.id, bubble.rect, "", bubble.source, bubble.maskContour)
        }
        return TranslationResult(
            imageName = ocrPage.imageFile.name,
            width = ocrPage.width,
            height = ocrPage.height,
            bubbles = bubbles,
            metadata = metadata
        )
    }

    fun translationFileFor(imageFile: File): File {
        return store.translationFileFor(imageFile)
    }

    fun releaseLoadedModels() {
        pageRegionDetector.releaseLoadedDetectors()
    }


    private suspend fun recognizeBubblesIndividually(
        cropSource: BitmapCropSource,
        regions: List<PageRegion>,
        language: TranslationLanguage,
        useLocalOcr: Boolean,
        ocrSettings: OcrApiSettings,
        ocrEngine: OcrEngine?
    ): List<OcrBubble> {
        val bubbles = ArrayList<OcrBubble>(regions.size)
        val isJaLocal = useLocalOcr && language == TranslationLanguage.JA_TO_ZH
        val jaLocalConcurrency = if (isJaLocal) LocalOcrConcurrency.resolve(ocrSettings.localOcrConcurrencyLimit) else 1
        if (isJaLocal && jaLocalConcurrency > 1) {
            ocrEngineRegistry.ensureJaPool("Pipeline", ocrSettings.localOcrConcurrencyLimit)
            val results = coroutineScope {
                regions.map { region ->
                    async(Dispatchers.Default) {
                        val clamped = PipelineBitmapDecoder.clampRect(
                            region.rect, cropSource.width, cropSource.height
                        ) ?: return@async null
                        val crop = cropSource.decodeRegion(clamped) ?: return@async null
                        val engine = ocrEngineRegistry.borrowJa("Pipeline")
                        val text = if (engine != null) {
                            try {
                                bubbleTextRecognizer.sanitizeJaCrop(engine, crop, language, "Pipeline")
                            } catch (e: CancellationException) {
                                throw e
                            } catch (e: Throwable) {
                                AppLogger.log("Pipeline", "JA pool OCR threw for region", e)
                                ""
                            } finally {
                                ocrEngineRegistry.returnJa(engine)
                                crop.recycleSafely()
                            }
                        } else {
                            try {
                                bubbleTextRecognizer
                                    .recognizeCrop(crop, language, useLocalOcr = true, logTag = "Pipeline")
                                    .textOrEmpty()
                            } finally {
                                crop.recycleSafely()
                            }
                        }
                        if (text.isBlank()) null
                        else OcrBubble(
                            id = region.id,
                            rect = region.rect,
                            text = text,
                            source = region.source,
                            maskContour = region.maskContour
                        )
                    }
                }.awaitAll()
            }
            results.filterNotNullTo(bubbles)
        } else if (useLocalOcr || ocrSettings.apiOcrConcurrencyLimit <= 1) {
            for (region in regions) {
                val text = recognizeRegionFromSource(
                    cropSource = cropSource,
                    rect = region.rect,
                    language = language,
                    useLocalOcr = useLocalOcr,
                    logTag = "Pipeline"
                )
                if (text.isBlank()) continue
                bubbles.add(
                    OcrBubble(
                        id = region.id,
                        rect = region.rect,
                        text = text,
                        source = region.source,
                        maskContour = region.maskContour
                    )
                )
            }
        } else {
            val semaphore = Semaphore(ocrSettings.apiOcrConcurrencyLimit)
            val results = coroutineScope {
                regions.map { region ->
                    async(Dispatchers.IO) {
                        semaphore.withPermit {
                            val text = recognizeRegionFromSource(
                                cropSource = cropSource,
                                rect = region.rect,
                                language = language,
                                useLocalOcr = false,
                                logTag = "Pipeline"
                            )
                            if (text.isBlank()) null
                            else OcrBubble(
                                id = region.id,
                                rect = region.rect,
                                text = text,
                                source = region.source,
                                maskContour = region.maskContour
                            )
                        }
                    }
                }.awaitAll()
            }
            results.filterNotNullTo(bubbles)
        }
        return bubbles
    }

    private suspend fun recognizeFullPageBaiduAndMatch(
        cropSource: BitmapCropSource,
        pageRegions: PageRegionDetectionResult,
        language: TranslationLanguage
    ): List<OcrBubble> {
        val fullBitmap = cropSource.decodeRegion(
            RectF(0f, 0f, pageRegions.width.toFloat(), pageRegions.height.toFloat()),
            maxEdge = BAIDU_FULL_PAGE_MAX_EDGE
        ) ?: run {
            AppLogger.log("Pipeline", "Baidu full-page OCR: failed to decode full page image")
            return emptyList()
        }
        return try {
            val baiduWords = llmClient.recognizeFullPageWithBaidu(fullBitmap, language)
            if (baiduWords.isNullOrEmpty()) {
                AppLogger.log("Pipeline", "Baidu full-page OCR returned no words, falling back to empty")
                pageRegions.regions.map { region ->
                    OcrBubble(
                        id = region.id,
                        rect = region.rect,
                        text = "",
                        source = region.source,
                        maskContour = region.maskContour
                    )
                }
            } else {
                AppLogger.log(
                    "Pipeline",
                    "Baidu full-page OCR recognized ${baiduWords.size} words for ${pageRegions.regions.size} regions"
                )
                matchBaiduWordsToRegions(baiduWords, pageRegions.regions)
            }
        } catch (e: Exception) {
            AppLogger.log("Pipeline", "Baidu full-page OCR failed, falling back to empty", e)
            pageRegions.regions.map { region ->
                OcrBubble(
                    id = region.id,
                    rect = region.rect,
                    text = "",
                    source = region.source,
                    maskContour = region.maskContour
                )
            }
        } finally {
            fullBitmap.recycleSafely()
        }
    }

    private fun matchBaiduWordsToRegions(
        words: List<BaiduOcrWord>,
        regions: List<PageRegion>
    ): List<OcrBubble> {
        val regionTexts = Array(regions.size) { StringBuilder() }
        for (word in words) {
            val wordLoc = word.location ?: continue
            val cx = wordLoc.centerX()
            val cy = wordLoc.centerY()
            var bestRegionIndex = -1
            for (i in regions.indices) {
                if (regions[i].rect.contains(cx, cy)) {
                    bestRegionIndex = i
                    break
                }
            }
            if (bestRegionIndex < 0) {
                var bestIou = 0f
                for (i in regions.indices) {
                    val iou = rectIoU(wordLoc, regions[i].rect)
                    if (iou > bestIou) {
                        bestIou = iou
                        bestRegionIndex = i
                    }
                }
                if (bestIou < BAIDU_WORD_REGION_IOU_THRESHOLD) {
                    bestRegionIndex = -1
                }
            }
            if (bestRegionIndex >= 0) {
                if (regionTexts[bestRegionIndex].isNotEmpty()) {
                    regionTexts[bestRegionIndex].append('\n')
                }
                regionTexts[bestRegionIndex].append(word.words)
            }
        }
        return regions.mapIndexed { index, region ->
            val text = regionTexts[index].toString()
            OcrBubble(
                id = region.id,
                rect = region.rect,
                text = text,
                source = region.source,
                maskContour = region.maskContour
            )
        }
    }

    private suspend fun recognizeRegionFromSource(
        cropSource: BitmapCropSource,
        rect: RectF,
        language: TranslationLanguage,
        useLocalOcr: Boolean,
        logTag: String
    ): String {
        val clamped = PipelineBitmapDecoder.clampRect(rect, cropSource.width, cropSource.height) ?: return ""
        val crop = cropSource.decodeRegion(clamped) ?: return ""
        return try {
            when (val result = bubbleTextRecognizer.recognizeCrop(crop, language, useLocalOcr, logTag)) {
                is OcrRecognitionResult.Success -> result.text
                is OcrRecognitionResult.Failure -> {
                    AppLogger.log(logTag, "OCR failed for region", result.error)
                    ""
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            AppLogger.log(logTag, "OCR threw for region", e)
            ""
        } finally {
            crop.recycleSafely()
        }
    }


    companion object {
        private const val STANDARD_PROMPT_ASSET = "prompts/llm_prompts.json"
        private const val FULL_TRANS_PROMPT_ASSET = "prompts/llm_prompts_FullTrans.json"
        private const val VL_PROMPT_ASSET = "prompts/vl_bubble_prompts.json"
        private const val MODEL_RESPONSE_SILENT_RETRY_COUNT = 3
        private const val BAIDU_FULL_PAGE_MAX_EDGE = 4096
        private const val BAIDU_WORD_REGION_IOU_THRESHOLD = 0.15f
        private const val BAIDU_FULL_PAGE_CACHE_MODE = "baidu_fullpage"
    }

    private fun buildExpectedTranslationMetadata(
        imageFile: File,
        fullTranslate: Boolean,
        useVlDirectTranslate: Boolean,
        language: TranslationLanguage
    ): TranslationMetadata {
        val baseMetadata = when {
            useVlDirectTranslate -> buildTranslationMetadata(
                imageFile = imageFile,
                language = language,
                mode = TranslationMetadata.MODE_VL_DIRECT,
                promptAsset = VL_PROMPT_ASSET,
                ocrCacheMode = "",
                providerContext = null
            )
            fullTranslate -> buildTranslationMetadata(
                imageFile = imageFile,
                language = language,
                mode = TranslationMetadata.MODE_FULL_PAGE,
                promptAsset = FULL_TRANS_PROMPT_ASSET,
                ocrCacheMode = buildOcrCacheMode(
                    imageFile,
                    settingsStore.loadOcrApiSettings().useLocalOcr,
                    language
                ),
                providerContext = null
            )
            else -> buildTranslationMetadata(
                imageFile = imageFile,
                language = language,
                mode = TranslationMetadata.MODE_STANDARD,
                promptAsset = STANDARD_PROMPT_ASSET,
                ocrCacheMode = buildOcrCacheMode(
                    imageFile,
                    settingsStore.loadOcrApiSettings().useLocalOcr,
                    language
                ),
                providerContext = null
            )
        }
        return baseMetadata
    }

    private fun buildTranslationMetadata(
        imageFile: File,
        language: TranslationLanguage,
        mode: String,
        promptAsset: String,
        ocrCacheMode: String,
        providerContext: PageTranslationProviderContext?
    ): TranslationMetadata {
        val apiSettings = providerContext?.apiSettings ?: settingsStore.load()
        return TranslationMetadata(
            sourceLastModified = imageFile.lastModified(),
            sourceFileSize = imageFile.length(),
            mode = mode,
            language = language.name,
            promptAsset = PromptAssetResolver.resolve(appContext, promptAsset),
            apiFormat = apiSettings.apiFormat.prefValue,
            ocrCacheMode = ocrCacheMode
        )
    }

    private fun buildOcrMetadata(
        imageFile: File,
        language: TranslationLanguage,
        ocrSettings: OcrApiSettings,
        cacheMode: String
    ): OcrMetadata {
        val effectiveUseLocalOcr = ocrSettings.useLocalOcr && language.supportsLocalOcr()
        val engineModel = if (effectiveUseLocalOcr) {
            "local:$cacheMode"
        } else {
            val customParamsFingerprint = settingsStore.loadCustomRequestParameters()
                .asSequence()
                .filter { it.enabled && it.targetProviderId == OCR_PROVIDER_ID }
                .map {
                    buildString {
                        append(it.key.trim())
                        append('=')
                        append(it.value.trim())
                    }
                }
                .sorted()
                .joinToString("&")
            if (customParamsFingerprint.isBlank()) {
                "api:${ocrSettings.modelName}"
            } else {
                "api:${ocrSettings.modelName}?$customParamsFingerprint"
            }
        }
        return OcrMetadata(
            sourceLastModified = imageFile.lastModified(),
            sourceFileSize = imageFile.length(),
            cacheMode = cacheMode,
            language = language.name,
            engineModel = engineModel
        )
    }

    private fun buildOcrCacheMode(
        imageFile: File,
        useLocalOcr: Boolean,
        language: TranslationLanguage
    ): String {
        val baseMode = if (!useLocalOcr) {
            "api"
        } else {
            val ocrSettings = settingsStore.loadOcrApiSettings()
            when (language) {
                TranslationLanguage.JA_TO_ZH -> when (ocrSettings.japaneseLocalOcrEngine) {
                    JapaneseLocalOcrEngine.MANGA_OCR_MOBILE -> "local_ja_mangaocr_mobile"
                }
                TranslationLanguage.EN_TO_ZH,
                TranslationLanguage.ZH_HANS_TO_TARGET,
                TranslationLanguage.ZH_HANT_TO_TARGET,
                TranslationLanguage.CHN_ENG_TO_ZH,
                TranslationLanguage.FR_TO_ZH,
                TranslationLanguage.ES_TO_ZH,
                TranslationLanguage.PT_TO_ZH,
                TranslationLanguage.DE_TO_ZH,
                TranslationLanguage.IT_TO_ZH -> "local_ppocrv6_small_rec"
                TranslationLanguage.KO_TO_ZH -> "local_ko"
                TranslationLanguage.RU_TO_ZH -> "api"
            }
        }
        val strategyTag = PipelineBitmapDecoder.readImageSize(imageFile)?.let { size ->
            buildDetectionStrategyTag(size.width, size.height)
        } ?: "det_full_balloon_conf_v3"
        return "$baseMode|$strategyTag"
    }

    private fun buildBaiduFullPageOcrCacheMode(imageFile: File): String {
        val strategyTag = PipelineBitmapDecoder.readImageSize(imageFile)?.let { size ->
            buildDetectionStrategyTag(size.width, size.height)
        } ?: "det_full_balloon_conf_v3"
        return "${BAIDU_FULL_PAGE_CACHE_MODE}|$strategyTag"
    }

    private fun buildBaiduFullPageOcrMetadata(
        imageFile: File,
        language: TranslationLanguage,
        cacheMode: String
    ): OcrMetadata {
        return OcrMetadata(
            sourceLastModified = imageFile.lastModified(),
            sourceFileSize = imageFile.length(),
            cacheMode = cacheMode,
            language = language.name,
            engineModel = BAIDU_FULL_PAGE_CACHE_MODE
        )
    }

    private suspend fun <T> executeWithModelResponseRetries(
        logTag: String,
        block: suspend () -> T?
    ): T? {
        var lastError: LlmResponseException? = null
        repeat(MODEL_RESPONSE_SILENT_RETRY_COUNT) { attempt ->
            try {
                return block()
            } catch (e: LlmResponseException) {
                lastError = e
                AppLogger.log(
                    logTag,
                    "Model response invalid, retry ${attempt + 1}/$MODEL_RESPONSE_SILENT_RETRY_COUNT",
                    e
                )
            }
        }
        throw requireNotNull(lastError)
    }

    private fun LlmResponseException.withPageName(pageName: String): LlmResponseException {
        val pagePrefix = appContext.getString(R.string.error_page_prefix)
        if (responseContent.startsWith(pagePrefix)) return this
        return LlmResponseException(
            errorCode = errorCode,
            responseContent = "$pagePrefix$pageName\n$responseContent",
            cause = this
        )
    }

}

internal fun buildDetectionStrategyTag(
    pageWidth: Int,
    pageHeight: Int
): String {
    return if (shouldUseHighResolutionTiling(pageWidth, pageHeight)) {
        "det_tiled_640_balloon_conf_v11"
    } else {
        "det_full_balloon_conf_v3"
    }
}

internal fun rectIoU(a: RectF, b: RectF): Float {
    val interLeft = maxOf(a.left, b.left)
    val interTop = maxOf(a.top, b.top)
    val interRight = minOf(a.right, b.right)
    val interBottom = minOf(a.bottom, b.bottom)
    val interWidth = maxOf(0f, interRight - interLeft)
    val interHeight = maxOf(0f, interBottom - interTop)
    val interArea = interWidth * interHeight
    val areaA = maxOf(0f, a.width()) * maxOf(0f, a.height())
    val areaB = maxOf(0f, b.width()) * maxOf(0f, b.height())
    val union = areaA + areaB - interArea
    return if (union <= 0f) 0f else interArea / union
}

data class OcrBubble(
    val id: Int,
    val rect: RectF,
    val text: String,
    val source: BubbleSource = BubbleSource.UNKNOWN,
    val maskContour: FloatArray? = null
)

data class PageOcrResult(
    val imageFile: File,
    val width: Int,
    val height: Int,
    val bubbles: List<OcrBubble>,
    val cacheMode: String = "",
    val metadata: OcrMetadata = OcrMetadata()
)

internal fun PageOcrResult.withRecognizedTextBubblesOnly(logTag: String? = null): PageOcrResult {
    val filtered = bubbles.filter { it.text.isNotBlank() }
    if (filtered.size == bubbles.size) return this
    logTag?.let {
        AppLogger.log(it, "Dropped OCR bubbles without text: ${bubbles.size} -> ${filtered.size}")
    }
    return copy(bubbles = filtered)
}


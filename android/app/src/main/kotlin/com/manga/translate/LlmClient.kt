package com.manga.translate
import com.github.kiastr.venera_ssr.R

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Call
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.net.SocketTimeoutException
import java.net.URLEncoder
import java.util.LinkedHashMap
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class LlmClient(
    context: Context,
    private val settingsStore: SettingsStore = SettingsStore(context.applicationContext),
    private val baiduTokenManager: BaiduAccessTokenManager = BaiduAccessTokenManager(context.applicationContext)
) : LlmGateway {
    private val appContext = context.applicationContext
    private val promptCache = ConcurrentHashMap<String, LlmPromptConfig>()
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()
    private val formUrlEncodedMediaType = "application/x-www-form-urlencoded".toMediaType()
    private val baseHttpClient = OkHttpClient()
    private val httpClientCache = object : LinkedHashMap<Int, OkHttpClient>(MAX_CACHED_HTTP_CLIENTS, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<Int, OkHttpClient>?): Boolean {
            return size > MAX_CACHED_HTTP_CLIENTS
        }
    }

    override fun isConfigured(apiSettings: ApiSettings?): Boolean {
        return (apiSettings ?: settingsStore.load()).isValid()
    }

    override fun isOcrConfigured(): Boolean {
        return settingsStore.loadOcrApiSettings().isValid()
    }

    suspend fun translate(
        text: String,
        glossary: Map<String, String>,
        promptAsset: String = PROMPT_CONFIG_ASSET,
        requestTimeoutMs: Int? = null,
        retryCount: Int = RETRY_COUNT,
        apiSettings: ApiSettings? = null
    ): LlmTranslationResult? =
        withContext(Dispatchers.IO) {
            val content = requestContent(
                text = text,
                glossary = glossary,
                promptAsset = promptAsset,
                useJsonPayload = true,
                requestTimeoutMs = requestTimeoutMs,
                retryCount = retryCount,
                apiSettings = apiSettings
            )
                ?: return@withContext null
            parseTranslationContent(content)
    }

    override suspend fun translateBubbleItems(
        items: List<LlmBubbleTranslationRequestItem>,
        glossary: Map<String, String>,
        promptAsset: String,
        requestTimeoutMs: Int?,
        retryCount: Int,
        apiSettings: ApiSettings?
    ): LlmBubbleTranslationResult? =
        withContext(Dispatchers.IO) {
            val content = requestContent(
                text = "",
                glossary = glossary,
                promptAsset = promptAsset,
                useJsonPayload = true,
                requestTimeoutMs = requestTimeoutMs,
                retryCount = retryCount,
                apiSettings = apiSettings,
                userPayloadOverride = buildBubbleItemsUserPayload(items, glossary)
            )
                ?: return@withContext null
            parseBubbleTranslationContent(content, items.map { it.id })
        }

    override suspend fun extractGlossary(
        text: String,
        glossary: Map<String, String>,
        promptAsset: String
    ): Map<String, String>? = withContext(Dispatchers.IO) {
        requestContent(text, glossary, promptAsset, useJsonPayload = true)
            ?.let { parseGlossaryContent(it) }
    }

    suspend fun fetchModelList(
        apiUrl: String,
        apiKey: String,
        apiFormat: ApiFormat
    ): List<String> = withContext(Dispatchers.IO) {
        requestModelList(apiUrl, apiKey, apiFormat)
    }

    override suspend fun recognizeImageText(image: Bitmap, language: TranslationLanguage): String? = withContext(Dispatchers.IO) {
        val ocrSettings = settingsStore.loadOcrApiSettings()
        if (!ocrSettings.isValid() || ocrSettings.useLocalOcr) {
            return@withContext null
        }
        return@withContext when (ocrSettings.ocrApiFormat) {
            OcrApiFormat.OPENAI_COMPATIBLE -> recognizeWithOpenAi(ocrSettings, image)
            OcrApiFormat.BAIDU_AI -> recognizeWithBaiduAi(ocrSettings, image, language)
        }
    }

    private suspend fun recognizeWithOpenAi(ocrSettings: OcrApiSettings, image: Bitmap): String? {
        val endpoint = buildOpenAiEndpoint(ocrSettings.apiUrl)
        val payload = buildImageOcrPayload(ocrSettings, image)
        val timeoutMs = ocrSettings.timeoutSeconds * 1000
        var lastErrorCode: String? = null
        var lastErrorBody: String? = null
        for (attempt in 1..RETRY_COUNT) {
            currentCoroutineContext().ensureActive()
            val result = try {
                executeRequest(
                    request = Request.Builder()
                        .url(endpoint)
                        .post(payload.toString().toRequestBody(jsonMediaType))
                        .header("Content-Type", "application/json")
                        .header("Authorization", "Bearer ${ocrSettings.apiKey}")
                        .build(),
                    timeoutMs = timeoutMs
                ).use { response ->
                    val code = response.code
                    val body = response.body?.string().orEmpty()
                    if (code !in 200..299) {
                        AppLogger.log("LlmClient", "OCR HTTP $code on ${redactEndpoint(endpoint)}: ${summarizeBody(body)}")
                        lastErrorCode = "HTTP $code"
                        lastErrorBody = body
                        null
                    } else {
                        parseResponseContent(body, ApiFormat.OPENAI_COMPATIBLE)?.trim()
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                AppLogger.log("LlmClient", "OCR request failed on ${redactEndpoint(endpoint)} (attempt $attempt)", e)
                lastErrorCode = "NETWORK_ERROR"
                null
            }
            if (result != null || attempt == RETRY_COUNT) {
                if (result != null) return result
                if (lastErrorCode != null) {
                    AppLogger.log(
                        "LlmClient",
                        "OCR request failed on ${redactEndpoint(endpoint)}: $lastErrorCode, body=${summarizeBody(lastErrorBody)}"
                    )
                }
                return null
            }
            maybeBackoffBeforeRetry(
                attempt,
                RetryPolicy(maxAttempts = RETRY_COUNT, mode = RetryMode.DEFAULT),
                lastErrorCode,
                lastErrorBody
            )
        }
        return null
    }

    private suspend fun recognizeWithBaiduAi(ocrSettings: OcrApiSettings, image: Bitmap, language: TranslationLanguage): String? {
        val accessToken = baiduTokenManager.getAccessToken(ocrSettings.apiKey, ocrSettings.secretKey)
            ?: run {
                AppLogger.log("LlmClient", "Baidu OCR: failed to obtain access token")
                return null
            }
        val endpoint = BAIDU_OCR_GENERAL_URL + "?access_token=" + accessToken
        val imageBase64 = ImageEncodingUtils.encodeBitmapToBase64(image)
            ?: run {
                AppLogger.log("LlmClient", "Baidu OCR: failed to encode image")
                return null
            }
        val body = "image=" + java.net.URLEncoder.encode(imageBase64, "UTF-8") +
                "&language_type=" + java.net.URLEncoder.encode(language.baiduLanguageType, "UTF-8")
        val timeoutMs = ocrSettings.timeoutSeconds * 1000
        var lastErrorCode: String? = null
        var lastErrorBody: String? = null
        for (attempt in 1..RETRY_COUNT) {
            currentCoroutineContext().ensureActive()
            val result = try {
                executeRequest(
                    request = Request.Builder()
                        .url(endpoint)
                        .post(body.toRequestBody(formUrlEncodedMediaType))
                        .build(),
                    timeoutMs = timeoutMs
                ).use { response ->
                    val code = response.code
                    val respBody = response.body?.string().orEmpty()
                    if (code !in 200..299) {
                        AppLogger.log("LlmClient", "Baidu OCR HTTP $code: ${summarizeBody(respBody)}")
                        lastErrorCode = "HTTP $code"
                        lastErrorBody = respBody
                        null
                    } else {
                        parseBaiduOcrResponse(respBody)?.trim()
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                AppLogger.log("LlmClient", "Baidu OCR request failed (attempt $attempt)", e)
                lastErrorCode = "NETWORK_ERROR"
                null
            }
            if (result != null || attempt == RETRY_COUNT) {
                if (result != null) return result
                if (lastErrorCode != null) {
                    AppLogger.log(
                        "LlmClient",
                        "Baidu OCR request failed: $lastErrorCode, body=${summarizeBody(lastErrorBody)}"
                    )
                }
                return null
            }
            maybeBackoffBeforeRetry(
                attempt,
                RetryPolicy(maxAttempts = RETRY_COUNT, mode = RetryMode.DEFAULT),
                lastErrorCode,
                lastErrorBody
            )
        }
        return null
    }

    override suspend fun recognizeFullPageWithBaidu(
        image: Bitmap,
        language: TranslationLanguage
    ): List<BaiduOcrWord>? {
        val ocrSettings = settingsStore.loadOcrApiSettings()
        val accessToken = baiduTokenManager.getAccessToken(ocrSettings.apiKey, ocrSettings.secretKey)
            ?: run {
                AppLogger.log("LlmClient", "Baidu full-page OCR: failed to obtain access token")
                return null
            }
        val endpoint = BAIDU_OCR_GENERAL_LOCATION_URL + "?access_token=" + accessToken
        val imageBase64 = ImageEncodingUtils.encodeBitmapToBase64(image)
            ?: run {
                AppLogger.log("LlmClient", "Baidu full-page OCR: failed to encode image")
                return null
            }
        val body = "image=" + java.net.URLEncoder.encode(imageBase64, "UTF-8") +
                "&language_type=" + java.net.URLEncoder.encode(language.baiduLanguageType, "UTF-8") +
                "&recognize_granularity=big" +
                "&detect_direction=false" +
                "&vertexes_location=false" +
                "&paragraph=false"
        val timeoutMs = ocrSettings.timeoutSeconds * 1000
        var lastErrorCode: String? = null
        var lastErrorBody: String? = null
        for (attempt in 1..RETRY_COUNT) {
            currentCoroutineContext().ensureActive()
            val result = try {
                executeRequest(
                    request = Request.Builder()
                        .url(endpoint)
                        .post(body.toRequestBody(formUrlEncodedMediaType))
                        .build(),
                    timeoutMs = timeoutMs
                ).use { response ->
                    val code = response.code
                    val respBody = response.body?.string().orEmpty()
                    if (code !in 200..299) {
                        AppLogger.log("LlmClient", "Baidu full-page OCR HTTP $code: ${summarizeBody(respBody)}")
                        lastErrorCode = "HTTP $code"
                        lastErrorBody = respBody
                        null
                    } else {
                        parseBaiduFullPageOcrResponse(respBody)
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                AppLogger.log("LlmClient", "Baidu full-page OCR request failed (attempt $attempt)", e)
                lastErrorCode = "NETWORK_ERROR"
                null
            }
            if (result != null || attempt == RETRY_COUNT) {
                if (result != null) return result
                if (lastErrorCode != null) {
                    AppLogger.log(
                        "LlmClient",
                        "Baidu full-page OCR request failed: $lastErrorCode, body=${summarizeBody(lastErrorBody)}"
                    )
                }
                return null
            }
            maybeBackoffBeforeRetry(
                attempt,
                RetryPolicy(maxAttempts = RETRY_COUNT, mode = RetryMode.DEFAULT),
                lastErrorCode,
                lastErrorBody
            )
        }
        return null
    }

    private fun parseBaiduFullPageOcrResponse(body: String): List<BaiduOcrWord>? {
        return try {
            val json = JSONObject(body)
            if (json.has("error_code")) {
                val errorCode = json.optInt("error_code")
                if (errorCode != 0) {
                    val errorMsg = json.optString("error_msg", "")
                    AppLogger.log("LlmClient", "Baidu full-page OCR error: code=$errorCode, msg=$errorMsg")
                    return null
                }
            }
            val wordsResult = json.optJSONArray("words_result")
            if (wordsResult != null && wordsResult.length() > 0) {
                val results = ArrayList<BaiduOcrWord>(wordsResult.length())
                for (i in 0 until wordsResult.length()) {
                    val item = wordsResult.optJSONObject(i) ?: continue
                    val words = item.optString("words").trim().orEmpty()
                    if (words.isBlank()) continue
                    val location = item.optJSONObject("location")?.let { loc ->
                        val top = loc.optInt("top", 0).toFloat()
                        val left = loc.optInt("left", 0).toFloat()
                        val width = loc.optInt("width", 0).toFloat()
                        val height = loc.optInt("height", 0).toFloat()
                        if (width > 0f && height > 0f) {
                            RectF(left, top, left + width, top + height)
                        } else {
                            null
                        }
                    }
                    results.add(BaiduOcrWord(words = words, location = location))
                }
                return results.ifEmpty { null }
            }
            null
        } catch (e: Exception) {
            AppLogger.log("LlmClient", "Baidu full-page OCR response parse failed", e)
            null
        }
    }

    private fun parseBaiduOcrResponse(body: String): String? {
        return try {
            val json = JSONObject(body)
            if (json.has("error_code")) {
                val errorCode = json.optInt("error_code")
                if (errorCode != 0) {
                    val errorMsg = json.optString("error_msg", "")
                    AppLogger.log("LlmClient", "Baidu OCR error: code=$errorCode, msg=$errorMsg")
                    return null
                }
            }
            // Try words_result (general_basic format)
            val wordsResult = json.optJSONArray("words_result")
            if (wordsResult != null && wordsResult.length() > 0) {
                val texts = ArrayList<String>(wordsResult.length())
                for (i in 0 until wordsResult.length()) {
                    val word = wordsResult.optJSONObject(i)?.optString("words")?.trim().orEmpty()
                    if (word.isNotBlank()) texts.add(word)
                }
                return texts.joinToString("\n").ifBlank { null }
            }
            // Try data.ret (iOCR format)
            val data = json.optJSONObject("data")
            val ret = data?.optJSONArray("ret")
            if (ret != null && ret.length() > 0) {
                val texts = ArrayList<String>(ret.length())
                for (i in 0 until ret.length()) {
                    val word = ret.optJSONObject(i)?.optString("word")?.trim().orEmpty()
                    if (word.isNotBlank()) texts.add(word)
                }
                return texts.joinToString("\n").ifBlank { null }
            }
            null
        } catch (e: Exception) {
            AppLogger.log("LlmClient", "Baidu OCR response parse failed", e)
            null
        }
    }

    override suspend fun translateImageBubble(
        imageBase64: String,
        promptAsset: String,
        requestTimeoutMs: Int?,
        retryCount: Int,
        apiSettings: ApiSettings?
    ): String? = withContext(Dispatchers.IO) {
        requestImageContent(
            imageBase64 = imageBase64,
            promptAsset = promptAsset,
            requestTimeoutMs = requestTimeoutMs,
            retryCount = retryCount,
            apiSettings = apiSettings
        )?.let { parseImageTranslationContent(it) }
    }

    private suspend fun requestContent(
        text: String,
        glossary: Map<String, String>,
        promptAsset: String,
        useJsonPayload: Boolean,
        requestTimeoutMs: Int? = null,
        retryCount: Int = RETRY_COUNT,
        apiSettings: ApiSettings? = null,
        userPayloadOverride: String? = null
    ): String? {
        val settings = apiSettings ?: settingsStore.load()
        if (!settings.isValid()) return null
        val selectedModel = settings.modelName.trim()
        val endpoint = buildEndpoint(settings, selectedModel)
        val userPayload = userPayloadOverride ?: if (useJsonPayload) {
            buildUserPayload(text, glossary)
        } else {
            text
        }
        val payload = buildPayload(
            settings = settings,
            modelName = selectedModel,
            promptAsset = promptAsset,
            apiFormat = settings.apiFormat,
            userPayload = userPayload
        )
        val logModelIo = settingsStore.loadModelIoLogging()
        if (logModelIo) {
            AppLogger.log("LlmClient", "Model input ($promptAsset): $payload")
            AppLogger.log("LlmClient", "Selected model: $selectedModel")
        }
        val timeoutMs = requestTimeoutMs?.coerceAtLeast(1_000) ?: settingsStore.loadApiTimeoutMs()
        val retryPolicy = buildRetryPolicy(retryCount)
        val retries = retryPolicy.maxAttempts
        var lastErrorCode: String? = null
        var lastErrorBody: String? = null
        var lastResponseException: LlmResponseException? = null
        for (attempt in 1..retries) {
            currentCoroutineContext().ensureActive()
            val result = try {
                executeRequest(
                    request = buildJsonPostRequest(endpoint, payload, settings),
                    timeoutMs = timeoutMs
                ).use { response ->
                    val code = response.code
                    val body = response.body?.string().orEmpty()
                    if (code !in 200..299) {
                        AppLogger.log(
                            "LlmClient",
                            "HTTP $code on ${redactEndpoint(endpoint)}: ${summarizeBody(body)}"
                        )
                        lastErrorCode = "HTTP $code"
                        lastErrorBody = body
                        null
                    } else {
                        val content = parseResponseContent(body, settings.apiFormat)
                        if (content == null) {
                            AppLogger.log(
                                "LlmClient",
                                "Empty or invalid response content from ${redactEndpoint(endpoint)}"
                            )
                            lastResponseException = LlmResponseException(
                                errorCode = LlmErrorCode.InvalidResponse,
                                responseContent = body.ifBlank {
                                    appContext.getString(R.string.model_response_empty_content)
                                }
                            )
                        } else if (logModelIo) {
                            AppLogger.log("LlmClient", "Model output: $content")
                        }
                        content
                    }
                }
            } catch (e: SocketTimeoutException) {
                AppLogger.log("LlmClient", "Request timeout on ${redactEndpoint(endpoint)} (attempt $attempt)", e)
                lastErrorCode = LlmErrorCode.Timeout.value
                null
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                AppLogger.log("LlmClient", "Request failed on ${redactEndpoint(endpoint)} (attempt $attempt)", e)
                lastErrorCode = LlmErrorCode.NetworkError.value
                null
            }
            if (result != null || attempt == retries) {
                if (result != null) {
                    return result
                }
                lastResponseException?.let {
                    AppLogger.log(
                        "LlmClient",
                        "Response invalid on ${redactEndpoint(endpoint)}: ${summarizeBody(it.responseContent)}"
                    )
                    throw it
                }
                if (lastErrorCode != null) {
                    AppLogger.log(
                        "LlmClient",
                        "Request failed on ${redactEndpoint(endpoint)}: $lastErrorCode, body=${summarizeBody(lastErrorBody)}"
                    )
                    throw LlmRequestException(LlmErrorCode.from(lastErrorCode), lastErrorBody)
                }
                return null
            }
            maybeBackoffBeforeRetry(attempt, retryPolicy, lastErrorCode, lastErrorBody)
        }
        return null
    }

    private suspend fun requestImageContent(
        imageBase64: String,
        promptAsset: String,
        requestTimeoutMs: Int? = null,
        retryCount: Int = RETRY_COUNT,
        apiSettings: ApiSettings? = null
    ): String? {
        val settings = apiSettings ?: settingsStore.load()
        if (!settings.isValid()) return null
        val selectedModel = settings.modelName.trim()
        val endpoint = buildEndpoint(settings, selectedModel)
        val payload = buildImageTranslationPayload(
            settings = settings,
            modelName = selectedModel,
            imageBase64 = imageBase64,
            promptAsset = promptAsset,
            apiFormat = settings.apiFormat
        )
        val logModelIo = settingsStore.loadModelIoLogging()
        if (logModelIo) {
            AppLogger.log("LlmClient", "Model input ($promptAsset): ${sanitizeModelIoForLog(payload.toString())}")
            AppLogger.log("LlmClient", "Selected model: $selectedModel")
        }
        val timeoutMs = requestTimeoutMs?.coerceAtLeast(1_000) ?: settingsStore.loadApiTimeoutMs()
        val retryPolicy = buildRetryPolicy(retryCount)
        val retries = retryPolicy.maxAttempts
        var lastErrorCode: String? = null
        var lastErrorBody: String? = null
        var lastResponseException: LlmResponseException? = null
        for (attempt in 1..retries) {
            currentCoroutineContext().ensureActive()
            val result = try {
                executeRequest(
                    request = buildJsonPostRequest(endpoint, payload, settings),
                    timeoutMs = timeoutMs
                ).use { response ->
                    val code = response.code
                    val body = response.body?.string().orEmpty()
                    if (code !in 200..299) {
                        AppLogger.log("LlmClient", "HTTP $code on ${redactEndpoint(endpoint)}: ${summarizeBody(body)}")
                        lastErrorCode = "HTTP $code"
                        lastErrorBody = body
                        null
                    } else {
                        val content = parseResponseContent(body, settings.apiFormat)
                        if (content == null) {
                            AppLogger.log(
                                "LlmClient",
                                "Empty or invalid image response content from ${redactEndpoint(endpoint)}"
                            )
                            lastResponseException = LlmResponseException(
                                errorCode = LlmErrorCode.InvalidResponse,
                                responseContent = body.ifBlank {
                                    appContext.getString(R.string.model_response_empty_content)
                                }
                            )
                        } else if (logModelIo) {
                            AppLogger.log("LlmClient", "Model output: ${sanitizeModelIoForLog(content)}")
                        }
                        content
                    }
                }
            } catch (e: SocketTimeoutException) {
                AppLogger.log("LlmClient", "Request timeout on ${redactEndpoint(endpoint)} (attempt $attempt)", e)
                lastErrorCode = LlmErrorCode.Timeout.value
                null
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                AppLogger.log("LlmClient", "Request failed on ${redactEndpoint(endpoint)} (attempt $attempt)", e)
                lastErrorCode = LlmErrorCode.NetworkError.value
                null
            }
            if (result != null || attempt == retries) {
                if (result != null) {
                    return result
                }
                lastResponseException?.let {
                    AppLogger.log(
                        "LlmClient",
                        "Image response invalid on ${redactEndpoint(endpoint)}: ${summarizeBody(it.responseContent)}"
                    )
                    throw it
                }
                if (lastErrorCode != null) {
                    AppLogger.log(
                        "LlmClient",
                        "Request failed on ${redactEndpoint(endpoint)}: $lastErrorCode, body=${summarizeBody(lastErrorBody)}"
                    )
                    throw LlmRequestException(LlmErrorCode.from(lastErrorCode), lastErrorBody)
                }
                return null
            }
            maybeBackoffBeforeRetry(attempt, retryPolicy, lastErrorCode, lastErrorBody)
        }
        return null
    }

    private fun buildOpenAiEndpoint(baseUrl: String): String {
        return buildOpenAiCompatibleChatEndpoint(baseUrl)
    }

    private fun buildOpenAiResponsesEndpoint(baseUrl: String): String {
        return buildOpenAiResponsesApiEndpoint(baseUrl)
    }

    private fun buildOpenAiModelsEndpoint(baseUrl: String): String {
        return buildOpenAiCompatibleModelsEndpoint(baseUrl)
    }

    private fun buildEndpoint(settings: ApiSettings, modelName: String): String {
        return when (settings.apiFormat) {
            ApiFormat.OPENAI_COMPATIBLE -> buildOpenAiEndpoint(settings.apiUrl)
            ApiFormat.OPENAI_RESPONSES -> buildOpenAiResponsesEndpoint(settings.apiUrl)
            ApiFormat.GEMINI -> buildGeminiGenerateEndpoint(settings.apiUrl, modelName, settings.apiKey)
        }
    }

    private fun buildGeminiGenerateEndpoint(baseUrl: String, modelName: String, apiKey: String): String {
        val trimmed = baseUrl.trimEnd('/')
        val normalizedModel = normalizeGeminiModelName(modelName)
        val baseEndpoint = when {
            trimmed.contains(":generateContent") -> trimmed
            trimmed.endsWith("/v1beta") || trimmed.endsWith("/v1") -> {
                "$trimmed/$normalizedModel:generateContent"
            }
            else -> "$trimmed/v1beta/$normalizedModel:generateContent"
        }
        return appendApiKeyQuery(baseEndpoint, apiKey)
    }

    private fun buildGeminiModelsEndpoint(baseUrl: String, apiKey: String): String {
        val trimmed = baseUrl.trimEnd('/')
        val baseEndpoint = when {
            trimmed.endsWith("/models") -> trimmed
            trimmed.endsWith("/v1beta") || trimmed.endsWith("/v1") -> "$trimmed/models"
            else -> "$trimmed/v1beta/models"
        }
        return appendApiKeyQuery(baseEndpoint, apiKey)
    }

    private fun normalizeGeminiModelName(modelName: String): String {
        val trimmed = modelName.trim().removePrefix("/")
        return if (trimmed.startsWith("models/")) trimmed else "models/$trimmed"
    }

    private fun appendApiKeyQuery(endpoint: String, apiKey: String): String {
        if (apiKey.isBlank()) return endpoint
        val separator = if (endpoint.contains("?")) "&" else "?"
        return endpoint + separator + "key=" + URLEncoder.encode(apiKey, Charsets.UTF_8.name())
    }

    private fun redactEndpoint(endpoint: String): String =
        endpoint.replace(Regex("(\\?|&)key=[^&]*"), "$1key=***")

    private fun buildPayload(
        settings: ApiSettings,
        modelName: String,
        promptAsset: String,
        apiFormat: ApiFormat,
        userPayload: String
    ): JSONObject {
        val config = getPromptConfig(promptAsset)
        return when (apiFormat) {
            ApiFormat.OPENAI_COMPATIBLE -> buildOpenAiPayload(
                settings = settings,
                modelName = modelName,
                config = config,
                userPayload = userPayload
            )
            ApiFormat.OPENAI_RESPONSES -> buildOpenAiResponsesPayload(
                settings = settings,
                modelName = modelName,
                config = config,
                userPayload = userPayload
            )
            ApiFormat.GEMINI -> buildGeminiTextPayload(
                config = config,
                userPayload = userPayload
            )
        }
    }

    private fun buildOpenAiPayload(
        settings: ApiSettings,
        modelName: String,
        config: LlmPromptConfig,
        userPayload: String
    ): JSONObject {
        val llmParams = settingsStore.loadLlmParameters()
        val messages = JSONArray()
        messages.put(
            JSONObject()
                .put("role", "system")
                .put("content", config.systemPrompt)
        )
        for (message in config.exampleMessages) {
            messages.put(
                JSONObject()
                    .put("role", message.role)
                    .put("content", message.content)
            )
        }
        messages.put(
            JSONObject()
                .put("role", "user")
                .put(
                    "content",
                    config.userPromptPrefix + userPayload
                )
        )
        val payload = JSONObject()
            .put("model", modelName)
            .put("messages", messages)
        applyOpenAiSamplingParams(payload, llmParams, settings)
        applyOpenAiThinkingParams(payload, llmParams)
        applyCustomRequestParameters(payload, settings)
        return payload
    }

    private fun buildOpenAiResponsesPayload(
        settings: ApiSettings,
        modelName: String,
        config: LlmPromptConfig,
        userPayload: String
    ): JSONObject {
        val llmParams = settingsStore.loadLlmParameters()
        val input = JSONArray()
        for (message in config.exampleMessages) {
            input.put(
                JSONObject()
                    .put("role", mapOpenAiResponsesRole(message.role))
                    .put("content", message.content)
            )
        }
        input.put(
            JSONObject()
                .put("role", "user")
                .put("content", config.userPromptPrefix + userPayload)
        )
        val payload = JSONObject()
            .put("model", modelName)
            .put("input", input)
        if (config.systemPrompt.isNotBlank()) {
            payload.put("instructions", config.systemPrompt)
        }
        applyOpenAiResponsesSamplingParams(payload, llmParams)
        applyOpenAiThinkingParams(payload, llmParams)
        applyCustomRequestParameters(payload, settings)
        return payload
    }

    private fun mapOpenAiResponsesRole(role: String): String {
        return when (role.lowercase()) {
            "assistant", "model" -> "assistant"
            "system", "developer" -> "developer"
            else -> "user"
        }
    }

    private fun buildGeminiTextPayload(
        config: LlmPromptConfig,
        userPayload: String
    ): JSONObject {
        val userText = config.userPromptPrefix + userPayload
        val payload = JSONObject()
            .put("contents", buildGeminiContents(config, buildGeminiUserParts(buildGeminiTextPart(userText))))
        if (config.systemPrompt.isNotBlank()) {
            payload.put("systemInstruction", buildGeminiSystemInstruction(config.systemPrompt))
        }
        buildGeminiGenerationConfig(useJsonPayload = true)?.let { payload.put("generationConfig", it) }
        applyCustomRequestParameters(
            payload,
            ApiSettings(apiUrl = "", apiKey = "", modelName = "", apiFormat = ApiFormat.GEMINI)
        )
        return payload
    }

    private fun parseResponseContent(body: String, apiFormat: ApiFormat): String? {
        return when (apiFormat) {
            ApiFormat.OPENAI_COMPATIBLE -> parseOpenAiResponseContent(body)
            ApiFormat.OPENAI_RESPONSES -> parseOpenAiResponsesContent(body)
            ApiFormat.GEMINI -> parseGeminiResponseContent(body)
        }
    }

    private fun parseOpenAiResponseContent(body: String): String? {
        return try {
            val json = JSONObject(body)
            val choices = json.optJSONArray("choices") ?: return null
            val first = choices.optJSONObject(0) ?: return null
            val message = first.optJSONObject("message") ?: return null
            val rawContent = message.opt("content")
            when (rawContent) {
                is String -> rawContent.trim().ifBlank { null }
                is JSONArray -> {
                    val parts = ArrayList<String>(rawContent.length())
                    for (i in 0 until rawContent.length()) {
                        val item = rawContent.opt(i)
                        when (item) {
                            is String -> if (item.isNotBlank()) parts.add(item.trim())
                            is JSONObject -> {
                                val text = item.optString("text").trim()
                                if (text.isNotBlank()) parts.add(text)
                            }
                        }
                    }
                    parts.joinToString("\n").trim().ifBlank { null }
                }
                else -> null
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun parseOpenAiResponsesContent(body: String): String? {
        return try {
            val json = JSONObject(body)
            val outputText = json.optString("output_text").trim()
            if (outputText.isNotBlank()) {
                return outputText
            }
            val output = json.optJSONArray("output") ?: return null
            val parts = ArrayList<String>()
            for (i in 0 until output.length()) {
                val item = output.optJSONObject(i) ?: continue
                val type = item.optString("type").trim().lowercase()
                if (type.isNotBlank() && type != "message") continue
                val content = item.opt("content")
                when (content) {
                    is String -> if (content.isNotBlank()) parts.add(content.trim())
                    is JSONArray -> {
                        for (j in 0 until content.length()) {
                            when (val part = content.opt(j)) {
                                is String -> if (part.isNotBlank()) parts.add(part.trim())
                                is JSONObject -> {
                                    val partType = part.optString("type").trim().lowercase()
                                    if (
                                        partType.isNotBlank() &&
                                        partType != "output_text" &&
                                        partType != "text"
                                    ) {
                                        continue
                                    }
                                    val text = part.optString("text").trim()
                                    if (text.isNotBlank()) parts.add(text)
                                }
                            }
                        }
                    }
                }
            }
            parts.joinToString("\n").trim().ifBlank { null }
        } catch (e: Exception) {
            null
        }
    }

    private fun parseGeminiResponseContent(body: String): String? {
        return try {
            val json = JSONObject(body)
            val candidates = json.optJSONArray("candidates") ?: return null
            val first = candidates.optJSONObject(0) ?: return null
            val content = first.optJSONObject("content") ?: return null
            val parts = content.optJSONArray("parts") ?: return null
            val texts = ArrayList<String>(parts.length())
            for (i in 0 until parts.length()) {
                val text = parts.optJSONObject(i)?.optString("text")?.trim().orEmpty()
                if (text.isNotBlank()) {
                    texts.add(text)
                }
            }
            texts.joinToString("\n").trim().ifBlank { null }
        } catch (e: Exception) {
            null
        }
    }

    private fun buildImageOcrPayload(ocrSettings: OcrApiSettings, image: Bitmap): JSONObject {
        val config = getPromptConfig(OCR_PROMPT_CONFIG_ASSET)
        val imageBase64 = ImageEncodingUtils.encodeBitmapToBase64(image)
            ?: throw LlmRequestException(
                LlmErrorCode.ImageEncodeFailed,
                "Failed to encode OCR image as JPEG"
            )
        val userInstruction = config.userPromptPrefix.ifBlank { DEFAULT_OCR_USER_PROMPT }
        val userContent = JSONArray()
            .put(
                JSONObject()
                    .put("type", "text")
                    .put("text", userInstruction)
            )
            .put(
                JSONObject()
                    .put("type", "image_url")
                    .put(
                        "image_url",
                        JSONObject().put("url", "data:image/jpeg;base64,$imageBase64")
                    )
            )
        val messages = JSONArray()
        if (config.systemPrompt.isNotBlank()) {
            messages.put(
                JSONObject()
                    .put("role", "system")
                    .put("content", config.systemPrompt)
            )
        }
        for (message in config.exampleMessages) {
            messages.put(
                JSONObject()
                    .put("role", message.role)
                    .put("content", message.content)
            )
        }
        messages.put(
            JSONObject()
                .put("role", "user")
                .put("content", userContent)
        )
        val payload = JSONObject()
            .put("model", ocrSettings.modelName)
            .put("messages", messages)
        applyCustomRequestParameters(
            payload,
            ApiSettings(
                apiUrl = ocrSettings.apiUrl,
                apiKey = ocrSettings.apiKey,
                modelName = ocrSettings.modelName,
                apiFormat = ApiFormat.OPENAI_COMPATIBLE,
                providerId = OCR_PROVIDER_ID
            )
        )
        return payload
    }

    private fun buildImageTranslationPayload(
        settings: ApiSettings,
        modelName: String,
        imageBase64: String,
        promptAsset: String,
        apiFormat: ApiFormat
    ): JSONObject {
        return when (apiFormat) {
            ApiFormat.OPENAI_COMPATIBLE -> buildOpenAiImageTranslationPayload(
                settings = settings,
                modelName = modelName,
                imageBase64 = imageBase64,
                promptAsset = promptAsset
            )
            ApiFormat.OPENAI_RESPONSES -> buildOpenAiResponsesImageTranslationPayload(
                settings = settings,
                modelName = modelName,
                imageBase64 = imageBase64,
                promptAsset = promptAsset
            )
            ApiFormat.GEMINI -> buildGeminiImageTranslationPayload(imageBase64, promptAsset)
        }
    }

    private fun buildOpenAiImageTranslationPayload(
        settings: ApiSettings,
        modelName: String,
        imageBase64: String,
        promptAsset: String
    ): JSONObject {
        val llmParams = settingsStore.loadLlmParameters()
        val config = getPromptConfig(promptAsset)
        val messages = JSONArray()
        if (config.systemPrompt.isNotBlank()) {
            messages.put(
                JSONObject()
                    .put("role", "system")
                    .put("content", config.systemPrompt)
            )
        }
        for (message in config.exampleMessages) {
            messages.put(
                JSONObject()
                    .put("role", message.role)
                    .put("content", message.content)
            )
        }
        messages.put(
            JSONObject()
                .put("role", "user")
                .put(
                    "content",
                    JSONArray()
                        .put(
                            JSONObject()
                                .put("type", "text")
                                .put("text", config.userPromptPrefix.ifBlank {
                                    DEFAULT_IMAGE_TRANSLATION_USER_PROMPT
                                })
                        )
                        .put(
                            JSONObject()
                                .put("type", "image_url")
                                .put(
                                    "image_url",
                                    JSONObject().put("url", "data:image/jpeg;base64,$imageBase64")
                                )
                        )
                )
        )
        val payload = JSONObject()
            .put("model", modelName)
            .put("messages", messages)
        applyOpenAiSamplingParams(payload, llmParams, settings)
        applyOpenAiThinkingParams(payload, llmParams)
        applyCustomRequestParameters(payload, settings)
        return payload
    }

    private fun buildOpenAiResponsesImageTranslationPayload(
        settings: ApiSettings,
        modelName: String,
        imageBase64: String,
        promptAsset: String
    ): JSONObject {
        val llmParams = settingsStore.loadLlmParameters()
        val config = getPromptConfig(promptAsset)
        val input = JSONArray()
        for (message in config.exampleMessages) {
            input.put(
                JSONObject()
                    .put("role", mapOpenAiResponsesRole(message.role))
                    .put("content", message.content)
            )
        }
        input.put(
            JSONObject()
                .put("role", "user")
                .put(
                    "content",
                    JSONArray()
                        .put(
                            JSONObject()
                                .put("type", "input_text")
                                .put(
                                    "text",
                                    config.userPromptPrefix.ifBlank {
                                        DEFAULT_IMAGE_TRANSLATION_USER_PROMPT
                                    }
                                )
                        )
                        .put(
                            JSONObject()
                                .put("type", "input_image")
                                .put("image_url", "data:image/jpeg;base64,$imageBase64")
                        )
                )
        )
        val payload = JSONObject()
            .put("model", modelName)
            .put("input", input)
        if (config.systemPrompt.isNotBlank()) {
            payload.put("instructions", config.systemPrompt)
        }
        applyOpenAiResponsesSamplingParams(payload, llmParams)
        applyOpenAiThinkingParams(payload, llmParams)
        applyCustomRequestParameters(payload, settings)
        return payload
    }

    private fun applyOpenAiSamplingParams(
        payload: JSONObject,
        llmParams: LlmParameterSettings,
        settings: ApiSettings
    ) {
        llmParams.temperature?.let { payload.put("temperature", it) }
        llmParams.topP?.let { payload.put("top_p", it) }
        llmParams.topK?.let { payload.put("top_k", it) }
        llmParams.maxOutputTokens?.let { payload.put("max_output_tokens", it) }
        llmParams.frequencyPenalty?.let { payload.put("frequency_penalty", it) }
        llmParams.presencePenalty?.let { payload.put("presence_penalty", it) }
    }

    private fun applyOpenAiResponsesSamplingParams(
        payload: JSONObject,
        llmParams: LlmParameterSettings
    ) {
        llmParams.temperature?.let { payload.put("temperature", it) }
        llmParams.topP?.let { payload.put("top_p", it) }
        llmParams.maxOutputTokens?.let { payload.put("max_output_tokens", it) }
    }

    private fun buildGeminiImageTranslationPayload(
        imageBase64: String,
        promptAsset: String
    ): JSONObject {
        val config = getPromptConfig(promptAsset)
        val userText = config.userPromptPrefix.ifBlank {
            DEFAULT_IMAGE_TRANSLATION_USER_PROMPT
        }
        val payload = JSONObject().put(
            "contents",
            buildGeminiContents(
                config,
                buildGeminiUserParts(
                    buildGeminiTextPart(userText),
                    buildGeminiInlineImagePart(imageBase64)
                )
            )
        )
        if (config.systemPrompt.isNotBlank()) {
            payload.put("systemInstruction", buildGeminiSystemInstruction(config.systemPrompt))
        }
        buildGeminiGenerationConfig(useJsonPayload = false)?.let {
            payload.put("generationConfig", it)
        }
        applyCustomRequestParameters(
            payload,
            ApiSettings(apiUrl = "", apiKey = "", modelName = "", apiFormat = ApiFormat.GEMINI)
        )
        return payload
    }

    private fun applyCustomRequestParameters(payload: JSONObject, settings: ApiSettings) {
        val parameters = settingsStore.loadCustomRequestParameters()
        if (parameters.isEmpty()) return
        val providerId = settings.providerId.ifBlank { PRIMARY_PROVIDER_ID }
        val reservedKeys = reservedRequestKeys(settings.apiFormat)
        val seenKeys = LinkedHashSet<String>()
        parameters.forEach { parameter ->
            if (!parameter.enabled) return@forEach
            if (parameter.targetProviderId != providerId) return@forEach
            val key = parameter.key.trim()
            val value = parameter.value.trim()
            if (key.isBlank() && value.isBlank()) return@forEach
            if (key.isBlank()) {
                throw LlmRequestException(LlmErrorCode.CustomParamConflict, "blank key")
            }
            if (!seenKeys.add(key)) {
                throw LlmRequestException(LlmErrorCode.CustomParamConflict, key)
            }
            if (key in reservedKeys || payload.has(key)) {
                throw LlmRequestException(LlmErrorCode.CustomParamConflict, key)
            }
            payload.put(key, parseCustomRequestParameterValue(key, parameter.value))
        }
    }

    private fun parseCustomRequestParameterValue(key: String, rawValue: String): Any {
        val trimmed = rawValue.trim()
        if (trimmed.equals("true", ignoreCase = true)) return true
        if (trimmed.equals("false", ignoreCase = true)) return false
        if (trimmed.equals("null", ignoreCase = true)) return JSONObject.NULL
        trimmed.toLongOrNull()?.let { return it }
        trimmed.toDoubleOrNull()?.let { return it }
        if (trimmed.startsWith("{")) {
            return runCatching { JSONObject(trimmed) }
                .getOrElse { throw LlmRequestException(LlmErrorCode.CustomParamInvalidValue, key) }
        }
        if (trimmed.startsWith("[")) {
            return runCatching { JSONArray(trimmed) }
                .getOrElse { throw LlmRequestException(LlmErrorCode.CustomParamInvalidValue, key) }
        }
        return rawValue
    }

    private fun applyOpenAiThinkingParams(
        payload: JSONObject,
        llmParams: LlmParameterSettings
    ) {
        payload.put("enable_thinking", llmParams.enableThinking)
        if (llmParams.enableThinking) {
            payload.put("thinking_budget", llmParams.thinkingLength.openAiBudgetTokens())
        }
    }

    private fun resolveGeminiThinkingBudget(llmParams: LlmParameterSettings): Int {
        if (!llmParams.enableThinking) return 0
        return llmParams.thinkingLength.geminiThinkingBudget()
    }

    private fun sanitizeModelIoForLog(content: String): String {
        val sanitizedJson = runCatching {
            when {
                content.trimStart().startsWith("{") -> {
                    when (val sanitized = sanitizeJsonValue(JSONObject(content))) {
                        is JSONObject -> sanitized.toString()
                        is JSONArray -> sanitized.toString()
                        else -> null
                    }
                }
                content.trimStart().startsWith("[") -> {
                    when (val sanitized = sanitizeJsonValue(JSONArray(content))) {
                        is JSONObject -> sanitized.toString()
                        is JSONArray -> sanitized.toString()
                        else -> null
                    }
                }
                else -> null
            }
        }.getOrNull()
        return sanitizedJson ?: content
            .replace(Regex("""data:image/[^;]+;base64,[A-Za-z0-9+/=]+"""), "data:image/<base64 omitted>")
    }

    private fun sanitizeJsonValue(value: Any?): Any? {
        return when (value) {
            is JSONObject -> {
                val sanitized = JSONObject()
                val keys = value.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    val child = value.opt(key)
                    sanitized.put(key, sanitizeJsonField(key, child))
                }
                sanitized
            }
            is JSONArray -> {
                JSONArray().also { array ->
                    for (i in 0 until value.length()) {
                        array.put(sanitizeJsonValue(value.opt(i)))
                    }
                }
            }
            else -> value
        }
    }

    private fun sanitizeJsonField(key: String, value: Any?): Any? {
        val normalizedKey = key.lowercase()
        return when {
            normalizedKey == "url" && value is String && value.startsWith("data:image/", ignoreCase = true) -> {
                "data:image/<base64 omitted>"
            }
            normalizedKey == "data" && value is String -> {
                "<base64 omitted>"
            }
            normalizedKey == "image_url" || normalizedKey == "inline_data" || normalizedKey == "inlinedata" -> {
                sanitizeJsonValue(value)
            }
            else -> sanitizeJsonValue(value)
        }
    }

    private fun parseTranslationContent(content: String): LlmTranslationResult {
        val cleaned = stripCodeFence(content)
        val directFallback = parseTranslationFallback(cleaned)
        if (directFallback != null) {
            return directFallback
        }
        return try {
            val json = JSONObject(cleaned)
            val translation = extractTranslationText(json)
            if (translation.isBlank()) {
                AppLogger.log("LlmClient", "Missing translation field in response")
                throw LlmResponseException(LlmErrorCode.MissingTranslation, content)
            }
            LlmTranslationResult(translation, parseGlossaryUsed(json))
        } catch (e: LlmResponseException) {
            throw e
        } catch (e: Exception) {
            AppLogger.log(
                "LlmClient",
                "Invalid translation response format: ${summarizeBody(content)}",
                e
            )
            throw LlmResponseException(LlmErrorCode.InvalidFormat, content, e)
        }
    }

    private fun parseTranslationFallback(content: String): LlmTranslationResult? {
        val trimmed = content.trim()
        if (trimmed.isBlank()) return null
        if (trimmed.startsWith("{") || trimmed.startsWith("[")) return null
        // Some OpenAI-compatible providers still return the translation as plain text.
        return LlmTranslationResult(trimmed, emptyMap())
    }

    private fun parseBubbleTranslationContent(
        content: String,
        requestedIds: List<Int>
    ): LlmBubbleTranslationResult {
        val cleaned = stripCodeFence(content)
        val directFallback = parseBubbleTranslationFallback(cleaned, requestedIds)
        if (directFallback != null) {
            return directFallback
        }
        return try {
            if (cleaned.trim().startsWith("[")) {
                val items = parseBubbleTranslationItems(JSONArray(cleaned))
                if (items.isEmpty()) {
                    throw LlmResponseException(LlmErrorCode.MissingTranslationItems, content)
                }
                return LlmBubbleTranslationResult(items = items, glossaryUsed = emptyMap())
            }
            val json = JSONObject(cleaned)
            val items = extractBubbleTranslationItems(json, requestedIds)
            if (items.isEmpty()) {
                AppLogger.log("LlmClient", "Missing items field in structured translation response")
                throw LlmResponseException(LlmErrorCode.MissingTranslationItems, content)
            }
            LlmBubbleTranslationResult(items = items, glossaryUsed = parseGlossaryUsed(json))
        } catch (e: LlmResponseException) {
            throw e
        } catch (e: Exception) {
            AppLogger.log(
                "LlmClient",
                "Invalid structured translation response format: ${summarizeBody(content)}",
                e
            )
            throw LlmResponseException(LlmErrorCode.InvalidFormat, content, e)
        }
    }

    private fun parseBubbleTranslationFallback(
        content: String,
        requestedIds: List<Int>
    ): LlmBubbleTranslationResult? {
        val trimmed = content.trim()
        if (trimmed.isBlank()) return null
        if (trimmed.startsWith("{") || trimmed.startsWith("[")) return null
        val singleId = requestedIds.singleOrNull() ?: return null
        return LlmBubbleTranslationResult(
            items = listOf(LlmBubbleTranslationItem(id = singleId, translation = trimmed)),
            glossaryUsed = emptyMap()
        )
    }

    private fun extractBubbleTranslationItems(
        json: JSONObject,
        requestedIds: List<Int>
    ): List<LlmBubbleTranslationItem> {
        findBubbleTranslationItemsArray(json)?.let { array ->
            return parseBubbleTranslationItems(array)
        }
        val singleId = requestedIds.singleOrNull()
        val translation = extractStructuredTranslationText(json)
        if (singleId != null && translation.isNotBlank()) {
            return listOf(LlmBubbleTranslationItem(id = singleId, translation = translation))
        }
        return emptyList()
    }

    private fun findBubbleTranslationItemsArray(json: JSONObject): JSONArray? {
        val directKeys = listOf("items", "translations", "translation_items", "translationItems")
        for (key in directKeys) {
            json.optJSONArray(key)?.let { return it }
        }
        val nestedKeys = listOf("data", "result", "output", "response", "message")
        for (key in nestedKeys) {
            val nested = json.optJSONObject(key) ?: continue
            findBubbleTranslationItemsArray(nested)?.let { return it }
        }
        return null
    }

    private fun parseBubbleTranslationItems(array: JSONArray): List<LlmBubbleTranslationItem> {
        val items = ArrayList<LlmBubbleTranslationItem>(array.length())
        for (i in 0 until array.length()) {
            when (val item = array.opt(i)) {
                is JSONObject -> {
                    val id = parseBubbleTranslationItemId(item.opt("id"))
                    val translation = extractStructuredTranslationText(item).trim()
                    if (id != null) {
                        items.add(LlmBubbleTranslationItem(id = id, translation = translation))
                    }
                }
            }
        }
        return items
    }

    private fun parseBubbleTranslationItemId(value: Any?): Int? {
        return when (value) {
            is Number -> value.toInt()
            is String -> value.trim().toIntOrNull()
            else -> null
        }
    }

    private fun extractTranslationText(json: JSONObject): String {
        val directKeys = listOf("translation", "translated_text", "translatedText", "text", "content")
        for (key in directKeys) {
            val value = json.opt(key)
            when (value) {
                is String -> value.trim().takeIf { it.isNotBlank() }?.let { return it }
                is JSONObject -> extractTranslationText(value).takeIf { it.isNotBlank() }?.let { return it }
                is JSONArray -> joinJsonText(value).takeIf { it.isNotBlank() }?.let { return it }
            }
        }
        val nestedKeys = listOf("data", "result", "output", "response", "message")
        for (key in nestedKeys) {
            val nested = json.optJSONObject(key) ?: continue
            extractTranslationText(nested).takeIf { it.isNotBlank() }?.let { return it }
        }
        return ""
    }

    private fun extractStructuredTranslationText(json: JSONObject): String {
        val translationKeys = listOf("translation", "translated_text", "translatedText")
        for (key in translationKeys) {
            val value = json.optString(key, "").trim()
            if (value.isNotBlank()) return value
        }
        return ""
    }

    private fun joinJsonText(array: JSONArray): String {
        val parts = ArrayList<String>(array.length())
        for (i in 0 until array.length()) {
            when (val item = array.opt(i)) {
                is String -> item.trim().takeIf { it.isNotBlank() }?.let(parts::add)
                is JSONObject -> extractTranslationText(item).takeIf { it.isNotBlank() }?.let(parts::add)
            }
        }
        return parts.joinToString("\n").trim()
    }

    private fun parseImageTranslationContent(content: String): String? {
        val cleaned = stripCodeFence(content).trim()
        if (cleaned.isBlank()) return null
        return try {
            parseTranslationContent(cleaned).translation.trim().ifBlank { null }
        } catch (_: Exception) {
            // Some compatible providers may still return plain text for image translation.
            cleaned.ifBlank { null }
        }
    }

    private fun parseGlossaryContent(content: String): Map<String, String> {
        return try {
            val cleaned = stripCodeFence(content)
            val json = JSONObject(cleaned)
            parseGlossaryUsed(json)
        } catch (e: Exception) {
            AppLogger.log("LlmClient", "Glossary parse failed", e)
            emptyMap()
        }
    }

    private suspend fun requestModelList(
        apiUrl: String,
        apiKey: String,
        apiFormat: ApiFormat
    ): List<String> {
        if (apiUrl.isBlank()) {
            throw LlmRequestException("MISSING_URL")
        }
        val endpoint = when (apiFormat) {
            ApiFormat.OPENAI_COMPATIBLE,
            ApiFormat.OPENAI_RESPONSES -> buildOpenAiModelsEndpoint(apiUrl)
            ApiFormat.GEMINI -> buildGeminiModelsEndpoint(apiUrl, apiKey)
        }
        val timeoutMs = settingsStore.loadApiTimeoutMs()
        var lastErrorCode: String? = null
        var lastErrorBody: String? = null
        for (attempt in 1..RETRY_COUNT) {
            currentCoroutineContext().ensureActive()
            val result = try {
                val requestBuilder = Request.Builder()
                    .url(endpoint)
                    .get()
                    .header("Content-Type", "application/json")
                if (apiFormat.usesOpenAiAuth && apiKey.isNotBlank()) {
                    requestBuilder.header("Authorization", "Bearer $apiKey")
                }
                executeRequest(requestBuilder.build(), timeoutMs).use { response ->
                    val code = response.code
                    val body = response.body?.string().orEmpty()
                    if (code !in 200..299) {
                        AppLogger.log(
                            "LlmClient",
                            "Model list HTTP $code on ${redactEndpoint(endpoint)}: ${summarizeBody(body)}"
                        )
                        lastErrorCode = "HTTP $code"
                        lastErrorBody = body
                        null
                    } else {
                        val models = parseModelList(body, apiFormat)
                        if (models.isEmpty()) {
                            lastErrorCode = "EMPTY_RESPONSE"
                            lastErrorBody = body
                        }
                        models
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                AppLogger.log(
                    "LlmClient",
                    "Model list request failed on ${redactEndpoint(endpoint)} (attempt $attempt)",
                    e
                )
                lastErrorCode = "NETWORK_ERROR"
                null
            }
            if (!result.isNullOrEmpty() || attempt == RETRY_COUNT) {
                if (!result.isNullOrEmpty()) {
                    return result
                }
                if (lastErrorCode != null) {
                    AppLogger.log(
                        "LlmClient",
                        "Model list failed on ${redactEndpoint(endpoint)}: $lastErrorCode, body=${summarizeBody(lastErrorBody)}"
                    )
                    throw LlmRequestException(lastErrorCode, lastErrorBody)
                }
                return emptyList()
            }
            maybeBackoffBeforeRetry(
                attempt,
                RetryPolicy(maxAttempts = RETRY_COUNT, mode = RetryMode.DEFAULT),
                lastErrorCode,
                lastErrorBody
            )
        }
        return emptyList()
    }

    private suspend fun maybeBackoffBeforeRetry(
        attempt: Int,
        retryPolicy: RetryPolicy,
        errorCode: String?,
        errorBody: String?
    ) {
        if (attempt >= retryPolicy.maxAttempts || !shouldRetry(errorCode, errorBody, retryPolicy.mode)) {
            return
        }
        val delayMs = when (retryPolicy.mode) {
            RetryMode.DEFAULT -> (RETRY_BASE_DELAY_MS shl (attempt - 1)).coerceAtMost(RETRY_MAX_DELAY_MS)
            RetryMode.CONFIGURABLE -> CONFIGURED_RETRY_DELAY_MS
        }
        AppLogger.log(
            "LlmClient",
            "Retrying request after ${delayMs}ms delay (attempt ${attempt + 1}/${retryPolicy.maxAttempts}, error=$errorCode)"
        )
        delay(delayMs.toLong())
    }

    private fun buildRetryPolicy(retryCount: Int): RetryPolicy {
        val configuredRetryCount = settingsStore.loadApiRetryCount()
        return RetryPolicy(
            maxAttempts = if (retryCount == RETRY_COUNT) configuredRetryCount else retryCount.coerceAtLeast(1),
            mode = RetryMode.CONFIGURABLE
        )
    }

    private fun shouldRetry(
        errorCode: String?,
        errorBody: String?,
        mode: RetryMode
    ): Boolean {
        return when (mode) {
            RetryMode.DEFAULT -> shouldRetryWithBackoff(errorCode)
            RetryMode.CONFIGURABLE -> shouldRetryWithConfiguredMode(errorCode, errorBody)
        }
    }

    private fun shouldRetryWithBackoff(errorCode: String?): Boolean {
        if (errorCode == null) return false
        if (errorCode == "TIMEOUT" || errorCode == "NETWORK_ERROR" || errorCode == "HTTP 408" || errorCode == "HTTP 429") {
            return true
        }
        if (!errorCode.startsWith("HTTP ")) {
            return false
        }
        val status = errorCode.removePrefix("HTTP ").toIntOrNull() ?: return false
        return status >= 500
    }

    private fun shouldRetryWithConfiguredMode(errorCode: String?, errorBody: String?): Boolean {
        if (errorCode == null) return false
        if (errorCode == "TIMEOUT" || errorCode == "NETWORK_ERROR" || errorCode == "HTTP 408" || errorCode == "HTTP 429") {
            return true
        }
        val status = errorCode.removePrefix("HTTP ").toIntOrNull()
        if (status != null && status >= 500) {
            return true
        }
        if (errorBody != null) {
            val normalizedBody = errorBody.lowercase()
            if (
                normalizedBody.contains("temporarily unavailable") ||
                normalizedBody.contains("temporary unavailable") ||
                normalizedBody.contains("service unavailable") ||
                normalizedBody.contains("try again later") ||
                normalizedBody.contains("server busy") ||
                normalizedBody.contains("overloaded")
            ) {
                return true
            }
        }
        return false
    }

    private fun parseModelList(body: String, apiFormat: ApiFormat): List<String> {
        return when (apiFormat) {
            ApiFormat.OPENAI_COMPATIBLE,
            ApiFormat.OPENAI_RESPONSES -> parseOpenAiModelList(body)
            ApiFormat.GEMINI -> parseGeminiModelList(body)
        }
    }

    private fun parseOpenAiModelList(body: String): List<String> {
        return try {
            val json = JSONObject(body)
            val data = json.optJSONArray("data") ?: return emptyList()
            val models = ArrayList<String>(data.length())
            for (i in 0 until data.length()) {
                val id = data.optJSONObject(i)?.optString("id")?.trim().orEmpty()
                if (id.isNotBlank()) {
                    models.add(id)
                }
            }
            models
        } catch (e: Exception) {
            AppLogger.log("LlmClient", "Model list parse failed", e)
            emptyList()
        }
    }

    private fun parseGeminiModelList(body: String): List<String> {
        return try {
            val json = JSONObject(body)
            val modelsJson = json.optJSONArray("models") ?: return emptyList()
            val models = ArrayList<String>(modelsJson.length())
            for (i in 0 until modelsJson.length()) {
                val item = modelsJson.optJSONObject(i) ?: continue
                val id = item.optString("baseModelId").trim().ifBlank {
                    item.optString("name").trim().removePrefix("models/")
                }
                if (id.isNotBlank()) {
                    models.add(id)
                }
            }
            models
        } catch (e: Exception) {
            AppLogger.log("LlmClient", "Gemini model list parse failed", e)
            emptyList()
        }
    }

    private fun buildGeminiContents(config: LlmPromptConfig, userParts: JSONArray): JSONArray {
        val contents = JSONArray()
        for (message in config.exampleMessages) {
            val role = when (message.role.lowercase()) {
                "assistant", "model" -> "model"
                else -> "user"
            }
            contents.put(
                JSONObject()
                    .put("role", role)
                    .put("parts", buildGeminiUserParts(buildGeminiTextPart(message.content)))
            )
        }
        contents.put(
            JSONObject()
                .put("role", "user")
                .put("parts", userParts)
        )
        return contents
    }

    private fun buildGeminiSystemInstruction(systemPrompt: String): JSONObject {
        return JSONObject().put("parts", buildGeminiUserParts(buildGeminiTextPart(systemPrompt)))
    }

    private fun buildGeminiUserParts(vararg parts: JSONObject): JSONArray {
        val array = JSONArray()
        parts.forEach { array.put(it) }
        return array
    }

    private fun buildGeminiTextPart(text: String): JSONObject {
        return JSONObject().put("text", text)
    }

    private fun buildGeminiInlineImagePart(imageBase64: String): JSONObject {
        return JSONObject().put(
            "inline_data",
            JSONObject()
                .put("mime_type", "image/jpeg")
                .put("data", imageBase64)
        )
    }

    private fun buildGeminiGenerationConfig(useJsonPayload: Boolean): JSONObject? {
        val llmParams = settingsStore.loadLlmParameters()
        val config = JSONObject()
        if (useJsonPayload) {
            config.put("responseMimeType", "application/json")
        }
        llmParams.temperature?.let { config.put("temperature", it) }
        llmParams.topP?.let { config.put("topP", it) }
        llmParams.topK?.let { config.put("topK", it) }
        llmParams.maxOutputTokens?.let { config.put("maxOutputTokens", it) }
        llmParams.frequencyPenalty?.let { config.put("frequencyPenalty", it) }
        llmParams.presencePenalty?.let { config.put("presencePenalty", it) }
        config.put("thinkingConfig", buildGeminiThinkingConfig(llmParams))
        return config.takeIf { it.length() > 0 }
    }

    private fun buildGeminiThinkingConfig(llmParams: LlmParameterSettings): JSONObject {
        return JSONObject().put("thinkingBudget", resolveGeminiThinkingBudget(llmParams))
    }

    private fun buildJsonPostRequest(
        endpoint: String,
        payload: JSONObject,
        settings: ApiSettings
    ): Request {
        val requestBuilder = Request.Builder()
            .url(endpoint)
            .post(payload.toString().toRequestBody(jsonMediaType))
            .header("Content-Type", "application/json")
        if (settings.apiFormat.usesOpenAiAuth) {
            requestBuilder.header("Authorization", "Bearer ${settings.apiKey}")
        }
        return requestBuilder.build()
    }

    private fun getOrBuildClient(timeoutMs: Int): OkHttpClient {
        return synchronized(httpClientCache) {
            httpClientCache.getOrPut(timeoutMs) {
                baseHttpClient.newBuilder()
                    .connectTimeout(timeoutMs.toLong(), TimeUnit.MILLISECONDS)
                    .readTimeout(timeoutMs.toLong(), TimeUnit.MILLISECONDS)
                    .writeTimeout(timeoutMs.toLong(), TimeUnit.MILLISECONDS)
                    .callTimeout(timeoutMs.toLong(), TimeUnit.MILLISECONDS)
                    .build()
            }
        }
    }

    private suspend fun executeRequest(request: Request, timeoutMs: Int): Response {
        val client = getOrBuildClient(timeoutMs)
        return executeCallCancellable(client.newCall(request))
    }

    private suspend fun executeCallCancellable(call: Call): Response =
        suspendCancellableCoroutine { continuation ->
            continuation.invokeOnCancellation {
                call.cancel()
            }
            try {
                val response = call.execute()
                if (continuation.isActive) {
                    continuation.resume(response)
                } else {
                    response.close()
                }
            } catch (t: Throwable) {
                if (continuation.isActive) {
                    continuation.resumeWithException(t)
                }
            }
        }

    private fun getPromptConfig(name: String): LlmPromptConfig {
        val resolvedName = PromptAssetResolver.resolve(appContext, name)
        val style = settingsStore.loadTranslationStyle()
        val cacheKey = "$resolvedName $style"
        return promptCache.getOrPut(cacheKey) { loadPromptConfig(resolvedName, style) }
    }

    private fun loadPromptConfig(name: String, styleHint: String): LlmPromptConfig {
        val json = JSONObject(readAsset(name))
        val systemPrompt = json.optString("system_prompt")
            .replace("{{STYLE_HINT}}", styleHint)
        val userPromptPrefix = json.optString("user_prompt_prefix")
        val examplesJson = json.optJSONArray("example_messages") ?: JSONArray()
        val examples = ArrayList<PromptMessage>(examplesJson.length())
        for (i in 0 until examplesJson.length()) {
            val messageObj = examplesJson.optJSONObject(i) ?: continue
            val role = messageObj.optString("role")
            val content = messageObj.optString("content")
            if (role.isNotBlank() && content.isNotBlank()) {
                examples.add(PromptMessage(role, content))
            }
        }
        return LlmPromptConfig(systemPrompt, userPromptPrefix, examples)
    }

    private fun readAsset(name: String): String {
        return appContext.assets.open(name).bufferedReader().use { it.readText() }
    }

    private fun buildUserPayload(text: String, glossary: Map<String, String>): String {
        return JSONObject()
            .put("text", text)
            .put("glossary", buildGlossaryJson(glossary))
            .toString()
    }

    private fun buildBubbleItemsUserPayload(
        items: List<LlmBubbleTranslationRequestItem>,
        glossary: Map<String, String>
    ): String {
        val itemsJson = JSONArray()
        items.forEach { item ->
            itemsJson.put(
                JSONObject()
                    .put("id", item.id)
                    .put("text", item.text)
            )
        }
        return JSONObject()
            .put("items", itemsJson)
            .put("glossary", buildGlossaryJson(glossary))
            .toString()
    }

    private fun buildGlossaryJson(glossary: Map<String, String>): JSONObject {
        val glossaryJson = JSONObject()
        for ((key, value) in glossary) {
            glossaryJson.put(key, value)
        }
        return glossaryJson
    }

    private fun parseGlossaryUsed(json: JSONObject): Map<String, String> {
        json.optJSONObject("glossary_used")?.let { return parseGlossaryJson(it) }
        val nestedKeys = listOf("data", "result", "output", "response", "message")
        for (key in nestedKeys) {
            val nested = json.optJSONObject(key) ?: continue
            parseGlossaryUsed(nested).takeIf { it.isNotEmpty() }?.let { return it }
        }
        return emptyMap()
    }

    private fun parseGlossaryJson(glossaryJson: JSONObject): Map<String, String> {
        val glossary = mutableMapOf<String, String>()
        for (key in glossaryJson.keys()) {
            val value = glossaryJson.optString(key).trim()
            if (key.isNotBlank() && value.isNotBlank()) {
                glossary[key] = value
            }
        }
        return glossary
    }

    private fun stripCodeFence(content: String): String {
        val trimmed = content.trim()
        if (!trimmed.startsWith("```") || !trimmed.endsWith("```")) {
            return trimmed
        }
        var inner = trimmed.removePrefix("```").removeSuffix("```").trim()
        if (inner.startsWith("json", ignoreCase = true)) {
            inner = inner.removePrefix("json").trim()
        }
        return inner
    }

    private fun summarizeBody(body: String?, limit: Int = 600): String {
        if (body.isNullOrBlank()) return "(empty)"
        val normalized = body.replace("\n", " ").replace("\r", " ").trim()
        return if (normalized.length <= limit) normalized else normalized.take(limit) + "...(truncated)"
    }

    companion object {
        private const val PROMPT_CONFIG_ASSET = "prompts/llm_prompts.json"
        private const val OCR_PROMPT_CONFIG_ASSET = "prompts/ocr_prompts.json"
        private const val DEFAULT_OCR_USER_PROMPT =
            "<image>\nExtract only visible text from this image. Do not describe objects, people, or scene. If no text is visible, return None."
        private const val DEFAULT_IMAGE_TRANSLATION_USER_PROMPT =
            "Translate only the text visible in this manga bubble into Simplified Chinese. Output only the translated text."
        private const val RETRY_COUNT = 3
        private const val BAIDU_OCR_GENERAL_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
        private const val BAIDU_OCR_GENERAL_LOCATION_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
        private const val MAX_CACHED_HTTP_CLIENTS = 4
        private const val RETRY_BASE_DELAY_MS = 750
        private const val RETRY_MAX_DELAY_MS = 4_000
        private const val CONFIGURED_RETRY_DELAY_MS = 3_000
        internal fun buildOpenAiCompatibleChatEndpoint(baseUrl: String): String {
            val trimmed = normalizeOpenAiCompatibleBaseUrl(baseUrl)
            return if (trimmed.endsWith("/chat/completions", ignoreCase = true)) {
                trimmed
            } else {
                "$trimmed/chat/completions"
            }
        }

        internal fun buildOpenAiResponsesApiEndpoint(baseUrl: String): String {
            val trimmed = normalizeOpenAiCompatibleBaseUrl(baseUrl)
            return if (trimmed.endsWith("/responses", ignoreCase = true)) {
                trimmed
            } else {
                "$trimmed/responses"
            }
        }

        internal fun buildOpenAiCompatibleModelsEndpoint(baseUrl: String): String {
            val trimmed = normalizeOpenAiCompatibleBaseUrl(baseUrl)
            return if (trimmed.endsWith("/models", ignoreCase = true)) {
                trimmed
            } else {
                "$trimmed/models"
            }
        }

        private fun normalizeOpenAiCompatibleBaseUrl(baseUrl: String): String {
            return baseUrl.trim().trimEnd('/')
        }

        fun reservedRequestKeys(apiFormat: ApiFormat): Set<String> {
            return when (apiFormat) {
                ApiFormat.OPENAI_COMPATIBLE -> setOf(
                    "model",
                    "messages",
                    "temperature",
                    "top_p",
                    "top_k",
                    "max_output_tokens",
                    "frequency_penalty",
                    "presence_penalty",
                    "enable_thinking",
                    "thinking_budget"
                )
                ApiFormat.OPENAI_RESPONSES -> setOf(
                    "model",
                    "input",
                    "instructions",
                    "temperature",
                    "top_p",
                    "max_output_tokens",
                    "enable_thinking",
                    "thinking_budget"
                )
                ApiFormat.GEMINI -> setOf(
                    "contents",
                    "systemInstruction",
                    "generationConfig"
                )
            }
        }
    }

    override fun resourceContext(): Context = appContext
}

private data class RetryPolicy(
    val maxAttempts: Int,
    val mode: RetryMode
)

private enum class RetryMode {
    DEFAULT,
    CONFIGURABLE
}

class LlmRequestException(
    val errorCode: LlmErrorCode,
    val responseBody: String? = null
) : Exception("LLM request failed: ${errorCode.value}") {
    constructor(errorCode: String, responseBody: String? = null) : this(
        LlmErrorCode.from(errorCode),
        responseBody
    )
}

class LlmResponseException(
    val errorCode: LlmErrorCode,
    val responseContent: String,
    cause: Throwable? = null
) : Exception("LLM response invalid: ${errorCode.value}", cause) {
    constructor(errorCode: String, responseContent: String, cause: Throwable? = null) : this(
        LlmErrorCode.from(errorCode),
        responseContent,
        cause
    )
}

data class LlmTranslationResult(
    val translation: String,
    val glossaryUsed: Map<String, String>
)

data class LlmBubbleTranslationRequestItem(
    val id: Int,
    val text: String
)

data class LlmBubbleTranslationResult(
    val items: List<LlmBubbleTranslationItem>,
    val glossaryUsed: Map<String, String>
)

data class LlmBubbleTranslationItem(
    val id: Int,
    val translation: String
)

data class BaiduOcrWord(
    val words: String,
    val location: RectF?
)

private data class LlmPromptConfig(
    val systemPrompt: String,
    val userPromptPrefix: String,
    val exampleMessages: List<PromptMessage>
)

private data class PromptMessage(
    val role: String,
    val content: String
)

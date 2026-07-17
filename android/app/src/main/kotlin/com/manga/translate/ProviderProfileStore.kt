package com.manga.translate
import com.github.kiastr.venera_ssr.R

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

internal class ProviderProfileStore(
    private val storage: SettingsStoreStorage,
    private val apiSettingsStore: ApiSettingsStore,
    private val ocrSettingsStore: OcrSettingsStore,
    private val llmParameterStore: LlmParameterStore
) {
    fun loadCustomRequestParameters(): List<CustomRequestParameter> {
        val raw = storage.prefs.getString(SettingsStore.KEY_CUSTOM_REQUEST_PARAMETERS, null).orEmpty()
        if (raw.isBlank()) return emptyList()
        return runCatching {
            val array = storage.parseVersionedArrayPayload(
                raw = raw,
                arrayKey = "items",
                label = SettingsStore.KEY_CUSTOM_REQUEST_PARAMETERS
            )
            buildList {
                for (index in 0 until array.length()) {
                    val item = array.optJSONObject(index) ?: continue
                    val key = item.optString("key").trim()
                    val value = item.optString("value")
                    val enabled = item.optBoolean("enabled", true)
                    val targetProviderId = item.optString("targetProviderId")
                        .trim()
                        .ifBlank { PRIMARY_PROVIDER_ID }
                    if (key.isBlank() && value.isBlank()) continue
                    add(
                        CustomRequestParameter(
                            key = key,
                            value = value,
                            enabled = enabled,
                            targetProviderId = targetProviderId
                        )
                    )
                }
            }
        }.getOrDefault(emptyList())
    }

    fun saveCustomRequestParameters(parameters: List<CustomRequestParameter>) {
        val array = JSONArray()
        parameters.forEach { parameter ->
            val key = parameter.key.trim()
            val value = parameter.value
            if (key.isBlank() && value.isBlank()) return@forEach
            array.put(
                JSONObject()
                    .put("key", key)
                    .put("value", value)
                    .put("enabled", parameter.enabled)
                    .put(
                        "targetProviderId",
                        parameter.targetProviderId.trim().ifBlank { PRIMARY_PROVIDER_ID }
                    )
            )
        }
        storage.editSettings(setOf(SettingsStore.KEY_CUSTOM_REQUEST_PARAMETERS)) {
            putString(
                SettingsStore.KEY_CUSTOM_REQUEST_PARAMETERS,
                JSONObject()
                    .put("version", SettingsStore.SETTINGS_JSON_SCHEMA_VERSION)
                    .put("items", array)
                    .toString()
            )
        }
    }

    fun loadAdditionalTranslationProviders(): List<AdditionalTranslationProvider> {
        val raw = storage.prefs.getString(SettingsStore.KEY_ADDITIONAL_TRANSLATION_PROVIDERS, null)
            .orEmpty()
        if (raw.isBlank()) return emptyList()
        return runCatching {
            val array = storage.parseVersionedArrayPayload(
                raw = raw,
                arrayKey = "providers",
                label = SettingsStore.KEY_ADDITIONAL_TRANSLATION_PROVIDERS
            )
            buildList {
                for (index in 0 until array.length()) {
                    val item = array.optJSONObject(index) ?: continue
                    val provider = parseAdditionalTranslationProvider(item, index)
                    if (
                        provider.apiUrl.isBlank() &&
                        provider.apiKey.isBlank() &&
                        provider.modelName.isBlank()
                    ) {
                        continue
                    }
                    add(provider)
                }
            }
        }.getOrDefault(emptyList())
    }

    fun saveAdditionalTranslationProviders(providers: List<AdditionalTranslationProvider>) {
        val array = JSONArray()
        providers.forEachIndexed { index, provider ->
            val apiUrl = provider.apiUrl.trim()
            val apiKey = provider.apiKey.trim()
            val modelName = provider.modelName.trim()
            if (apiUrl.isBlank() && apiKey.isBlank() && modelName.isBlank()) return@forEachIndexed
            array.put(
                JSONObject()
                    .put("name", provider.name.ifBlank { defaultAdditionalProviderName(index) })
                    .put("apiUrl", apiUrl)
                    .put("apiKey", apiKey)
                    .put("modelName", modelName)
                    .put("weight", provider.weight.coerceAtLeast(1))
                    .put("enabled", provider.enabled)
            )
        }
        storage.editSettings(setOf(SettingsStore.KEY_ADDITIONAL_TRANSLATION_PROVIDERS)) {
            putString(
                SettingsStore.KEY_ADDITIONAL_TRANSLATION_PROVIDERS,
                JSONObject()
                    .put("version", SettingsStore.SETTINGS_JSON_SCHEMA_VERSION)
                    .put("providers", array)
                    .toString()
            )
        }
    }

    fun countEnabledConfiguredAdditionalProviders(): Int {
        return loadAdditionalTranslationProviders().count { it.enabled && it.isConfigured() }
    }

    fun loadMainTranslationProviderPool(): List<WeightedProviderCandidate> {
        val main = apiSettingsStore.load()
        val candidates = ArrayList<WeightedProviderCandidate>()
        if (main.isValid()) {
            candidates += WeightedProviderCandidate(
                providerId = PRIMARY_PROVIDER_ID,
                displayName = storage.appContext.getString(R.string.provider_name_primary),
                settings = main,
                weight = SettingsStore.PRIMARY_PROVIDER_WEIGHT,
                isPrimary = true
            )
        }
        loadAdditionalTranslationProviders().forEachIndexed { index, provider ->
            if (!provider.enabled || !provider.isConfigured()) return@forEachIndexed
            candidates += WeightedProviderCandidate(
                providerId = "additional_${index + 1}",
                displayName = provider.name.ifBlank { defaultAdditionalProviderName(index) },
                settings = ApiSettings(
                    apiUrl = provider.apiUrl.trim(),
                    apiKey = provider.apiKey.trim(),
                    modelName = provider.modelName.trim(),
                    apiFormat = main.apiFormat,
                    providerId = "additional_${index + 1}"
                ),
                weight = provider.weight.coerceAtLeast(1),
                isPrimary = false
            )
        }
        return candidates
    }

    fun loadAiProviderProfilesState(): AiProviderProfilesState {
        val raw = runCatching {
            if (storage.aiProviderProfilesFile.exists()) {
                storage.aiProviderProfilesFile.readText()
            } else {
                ""
            }
        }.getOrDefault("")
        if (raw.isBlank()) {
            return AiProviderProfilesState(activeProfileName = null, profiles = emptyList())
        }
        return runCatching {
            val root = JSONObject(raw)
            val version = when {
                !root.has("version") -> SettingsStore.LEGACY_SETTINGS_JSON_VERSION
                else -> root.optInt("version", SettingsStore.LEGACY_SETTINGS_JSON_VERSION)
            }
            if (version !in SettingsStore.LEGACY_SETTINGS_JSON_VERSION..SettingsStore.SETTINGS_JSON_SCHEMA_VERSION) {
                AppLogger.log(
                    "SettingsStore",
                    "Skip ai provider profiles for unsupported version=$version"
                )
                return AiProviderProfilesState(activeProfileName = null, profiles = emptyList())
            }
            val profilesJson = root.optJSONArray("profiles") ?: JSONArray()
            val profiles = buildList {
                for (index in 0 until profilesJson.length()) {
                    val item = profilesJson.optJSONObject(index) ?: continue
                    parseAiProviderProfile(item)?.let(::add)
                }
            }
            val activeProfileName = root.optString("activeProfileName").trim().ifBlank { null }
            val normalizedActive = activeProfileName?.takeIf { active ->
                profiles.any { it.name == active }
            }
            AiProviderProfilesState(
                activeProfileName = normalizedActive,
                profiles = profiles.sortedBy { it.name.lowercase() }
            )
        }.getOrDefault(AiProviderProfilesState(activeProfileName = null, profiles = emptyList()))
    }

    fun saveCurrentAsAiProviderProfile(name: String): Boolean {
        val normalizedName = name.trim()
        if (normalizedName.isBlank()) return false
        val currentState = loadAiProviderProfilesState()
        if (currentState.profiles.any { it.name == normalizedName }) return false
        val updatedProfiles = currentState.profiles + captureCurrentAiProviderProfile(normalizedName)
        writeAiProviderProfilesState(
            AiProviderProfilesState(
                activeProfileName = normalizedName,
                profiles = updatedProfiles
            )
        )
        return true
    }

    fun overwriteActiveAiProviderProfile(): Boolean {
        val currentState = loadAiProviderProfilesState()
        val activeProfileName = currentState.activeProfileName ?: return false
        val updatedProfiles = currentState.profiles.map { profile ->
            if (profile.name == activeProfileName) {
                captureCurrentAiProviderProfile(activeProfileName)
            } else {
                profile
            }
        }
        writeAiProviderProfilesState(currentState.copy(profiles = updatedProfiles))
        return true
    }

    fun applyAiProviderProfile(name: String): Boolean {
        val currentState = loadAiProviderProfilesState()
        val profile = currentState.profiles.firstOrNull { it.name == name } ?: return false
        apiSettingsStore.save(profile.mainSettings)
        apiSettingsStore.saveApiTimeoutSeconds(profile.apiTimeoutSeconds)
        ocrSettingsStore.saveOcrApiSettings(profile.ocrSettings)
        apiSettingsStore.saveFloatingTranslateApiSettings(profile.floatingTranslateSettings)
        llmParameterStore.saveLlmParameters(profile.llmParameters)
        saveCustomRequestParameters(profile.customRequestParameters)
        saveAdditionalTranslationProviders(profile.additionalTranslationProviders)
        writeAiProviderProfilesState(currentState.copy(activeProfileName = profile.name))
        return true
    }

    fun deleteAiProviderProfile(name: String): Boolean {
        val currentState = loadAiProviderProfilesState()
        val updatedProfiles = currentState.profiles.filterNot { it.name == name }
        if (updatedProfiles.size == currentState.profiles.size) return false
        val updatedActive = currentState.activeProfileName?.takeIf { it != name }
        writeAiProviderProfilesState(
            AiProviderProfilesState(
                activeProfileName = updatedActive,
                profiles = updatedProfiles
            )
        )
        return true
    }

    fun defaultAdditionalProviderName(index: Int): String {
        return defaultAdditionalProviderName(storage.appContext, index)
    }

    private fun captureCurrentAiProviderProfile(name: String): AiProviderProfile {
        return AiProviderProfile(
            name = name,
            mainSettings = apiSettingsStore.load(),
            apiTimeoutSeconds = apiSettingsStore.loadApiTimeoutSeconds(),
            ocrSettings = ocrSettingsStore.loadOcrApiSettings(),
            floatingTranslateSettings = apiSettingsStore.loadFloatingTranslateApiSettings(),
            llmParameters = llmParameterStore.loadLlmParameters(),
            customRequestParameters = loadCustomRequestParameters(),
            additionalTranslationProviders = loadAdditionalTranslationProviders()
        )
    }

    private fun writeAiProviderProfilesState(state: AiProviderProfilesState) {
        val root = JSONObject()
        root.put("version", SettingsStore.SETTINGS_JSON_SCHEMA_VERSION)
        root.put("activeProfileName", state.activeProfileName.orEmpty())
        val profilesArray = JSONArray()
        state.profiles
            .sortedBy { it.name.lowercase() }
            .forEach { profile ->
                profilesArray.put(serializeAiProviderProfile(profile))
            }
        root.put("profiles", profilesArray)
        storage.aiProviderProfilesFile.parentFile?.mkdirs()
        val tmp = File(
            storage.aiProviderProfilesFile.parentFile,
            "${storage.aiProviderProfilesFile.name}.tmp"
        )
        try {
            tmp.writeText(root.toString())
            if (!tmp.renameTo(storage.aiProviderProfilesFile)) {
                if (storage.aiProviderProfilesFile.exists()) {
                    storage.aiProviderProfilesFile.delete()
                }
                if (!tmp.renameTo(storage.aiProviderProfilesFile)) {
                    AppLogger.log("Settings", "Atomic rename failed for AI provider profiles")
                }
            }
        } catch (e: Exception) {
            AppLogger.log("Settings", "Failed to write AI provider profiles", e)
            tmp.delete()
            return
        }
        storage.settingsObserver.publish(setOf(SettingsStore.KEY_AI_PROVIDER_PROFILES_STATE))
    }

    private fun serializeAiProviderProfile(profile: AiProviderProfile): JSONObject {
        return JSONObject()
            .put("name", profile.name)
            .put(
                "mainSettings",
                JSONObject()
                    .put("apiUrl", profile.mainSettings.apiUrl)
                    .put("apiKey", profile.mainSettings.apiKey)
                    .put("modelName", profile.mainSettings.modelName)
                    .put("apiFormat", profile.mainSettings.apiFormat.prefValue)
                    .put("apiTimeoutSeconds", profile.apiTimeoutSeconds)
            )
            .put(
                "ocrSettings",
                JSONObject()
                    .put("useLocalOcr", profile.ocrSettings.useLocalOcr)
                    .put(
                        "japaneseLocalOcrEngine",
                        profile.ocrSettings.japaneseLocalOcrEngine.prefValue
                    )
                    .put("apiUrl", profile.ocrSettings.apiUrl)
                    .put("apiKey", profile.ocrSettings.apiKey)
                    .put("modelName", profile.ocrSettings.modelName)
                    .put("timeoutSeconds", profile.ocrSettings.timeoutSeconds)
                    .put("apiOcrConcurrencyLimit", profile.ocrSettings.apiOcrConcurrencyLimit)
                    .put("localOcrConcurrencyLimit", profile.ocrSettings.localOcrConcurrencyLimit)
                    .put("ocrApiFormat", profile.ocrSettings.ocrApiFormat.prefValue)
                    .put("secretKey", profile.ocrSettings.secretKey)
            )
            .put(
                "floatingTranslateSettings",
                JSONObject()
                    .put("apiUrl", profile.floatingTranslateSettings.apiUrl)
                    .put("apiKey", profile.floatingTranslateSettings.apiKey)
                    .put("modelName", profile.floatingTranslateSettings.modelName)
                    .put("timeoutSeconds", profile.floatingTranslateSettings.timeoutSeconds)
                    .put(
                        "useVlDirectTranslate",
                        profile.floatingTranslateSettings.useVlDirectTranslate
                    )
                    .put(
                        "ocrConcurrencyLimit",
                        profile.floatingTranslateSettings.ocrConcurrencyLimit
                    )
                    .put(
                        "aiApiConcurrencyLimit",
                        profile.floatingTranslateSettings.aiApiConcurrencyLimit
                    )
                    .put(
                        "proofreadingModeEnabled",
                        profile.floatingTranslateSettings.proofreadingModeEnabled
                    )
                    .put(
                        "autoCloseOnScreenChangeEnabled",
                        profile.floatingTranslateSettings.autoCloseOnScreenChangeEnabled
                    )
                    .put(
                        "singleTapAction",
                        profile.floatingTranslateSettings.singleTapAction.prefValue
                    )
                    .put(
                        "doubleTapAction",
                        profile.floatingTranslateSettings.doubleTapAction.prefValue
                    )
                    .put(
                        "longPressAction",
                        profile.floatingTranslateSettings.longPressAction.prefValue
                    )
                    .put(
                        "tripleTapAction",
                        profile.floatingTranslateSettings.tripleTapAction.prefValue
                    )
            )
            .put(
                "llmParameters",
                JSONObject()
                    .put("temperature", profile.llmParameters.temperature)
                    .put("topP", profile.llmParameters.topP)
                    .put("topK", profile.llmParameters.topK)
                    .put("maxOutputTokens", profile.llmParameters.maxOutputTokens)
                    .put("enableThinking", profile.llmParameters.enableThinking)
                    .put("thinkingLength", profile.llmParameters.thinkingLength.prefValue)
                    .put("frequencyPenalty", profile.llmParameters.frequencyPenalty)
                    .put("presencePenalty", profile.llmParameters.presencePenalty)
            )
            .put(
                "customRequestParameters",
                JSONArray().apply {
                    profile.customRequestParameters.forEach { parameter ->
                        put(
                            JSONObject()
                                .put("key", parameter.key)
                                .put("value", parameter.value)
                                .put("enabled", parameter.enabled)
                                .put(
                                    "targetProviderId",
                                    parameter.targetProviderId.ifBlank { PRIMARY_PROVIDER_ID }
                                )
                        )
                    }
                }
            )
            .put(
                "additionalTranslationProviders",
                JSONArray().apply {
                    profile.additionalTranslationProviders.forEachIndexed { index, provider ->
                        put(
                            JSONObject()
                                .put(
                                    "name",
                                    provider.name.ifBlank { defaultAdditionalProviderName(index) }
                                )
                                .put("apiUrl", provider.apiUrl)
                                .put("apiKey", provider.apiKey)
                                .put("modelName", provider.modelName)
                                .put("weight", provider.weight)
                                .put("enabled", provider.enabled)
                        )
                    }
                }
            )
    }

    private fun parseAiProviderProfile(item: JSONObject): AiProviderProfile? {
        val name = item.optString("name").trim()
        if (name.isBlank()) return null
        val mainJson = item.optJSONObject("mainSettings") ?: JSONObject()
        val ocrJson = item.optJSONObject("ocrSettings") ?: JSONObject()
        val floatingJson = item.optJSONObject("floatingTranslateSettings") ?: JSONObject()
        val llmJson = item.optJSONObject("llmParameters") ?: JSONObject()
        val customParams = item.optJSONArray("customRequestParameters") ?: JSONArray()
        val additionalProviders = item.optJSONArray("additionalTranslationProviders") ?: JSONArray()
        return AiProviderProfile(
            name = name,
            mainSettings = ApiSettings(
                apiUrl = mainJson.optString("apiUrl", SettingsStore.DEFAULT_API_URL),
                apiKey = mainJson.optString("apiKey"),
                modelName = mainJson.optString("modelName", SettingsStore.DEFAULT_MODEL),
                apiFormat = ApiFormat.fromPref(mainJson.optStringOrNull("apiFormat")),
                providerId = PRIMARY_PROVIDER_ID
            ),
            apiTimeoutSeconds = mainJson.optInt(
                "apiTimeoutSeconds",
                SettingsStore.DEFAULT_API_TIMEOUT_SECONDS
            ).coerceIn(
                SettingsStore.MIN_API_TIMEOUT_SECONDS,
                SettingsStore.MAX_API_TIMEOUT_SECONDS
            ),
            ocrSettings = OcrApiSettings(
                useLocalOcr = ocrJson.optBoolean("useLocalOcr", true),
                japaneseLocalOcrEngine = JapaneseLocalOcrEngine.fromPref(
                    ocrJson.optStringOrNull("japaneseLocalOcrEngine")
                ),
                apiUrl = ocrJson.optString("apiUrl", SettingsStore.DEFAULT_OCR_API_URL),
                apiKey = ocrJson.optString("apiKey"),
                modelName = ocrJson.optString("modelName", SettingsStore.DEFAULT_OCR_MODEL_NAME),
                timeoutSeconds = ocrJson.optInt(
                    "timeoutSeconds",
                    SettingsStore.DEFAULT_OCR_API_TIMEOUT_SECONDS
                ).coerceIn(
                    SettingsStore.MIN_OCR_API_TIMEOUT_SECONDS,
                    SettingsStore.MAX_OCR_API_TIMEOUT_SECONDS
                ),
                apiOcrConcurrencyLimit = ocrJson.optInt(
                    "apiOcrConcurrencyLimit",
                    SettingsStore.DEFAULT_OCR_API_CONCURRENCY
                ).coerceIn(
                    SettingsStore.MIN_OCR_API_CONCURRENCY,
                    SettingsStore.MAX_OCR_API_CONCURRENCY
                ),
                localOcrConcurrencyLimit = ocrJson.optInt(
                    "localOcrConcurrencyLimit",
                    SettingsStore.DEFAULT_LOCAL_OCR_CONCURRENCY
                ).coerceIn(
                    SettingsStore.MIN_LOCAL_OCR_CONCURRENCY,
                    SettingsStore.MAX_LOCAL_OCR_CONCURRENCY
                ),
                ocrApiFormat = OcrApiFormat.fromPref(
                    ocrJson.optStringOrNull("ocrApiFormat")
                ),
                secretKey = ocrJson.optString("secretKey")
            ),
            floatingTranslateSettings = FloatingTranslateApiSettings(
                apiUrl = floatingJson.optString("apiUrl"),
                apiKey = floatingJson.optString("apiKey"),
                modelName = floatingJson.optString("modelName"),
                timeoutSeconds = floatingJson.optInt(
                    "timeoutSeconds",
                    SettingsStore.DEFAULT_FLOATING_API_TIMEOUT_SECONDS
                ).coerceIn(
                    SettingsStore.MIN_FLOATING_API_TIMEOUT_SECONDS,
                    SettingsStore.MAX_FLOATING_API_TIMEOUT_SECONDS
                ),
                useVlDirectTranslate = floatingJson.optBoolean("useVlDirectTranslate", false),
                ocrConcurrencyLimit = floatingJson.optInt(
                    "ocrConcurrencyLimit",
                    if (floatingJson.has("vlTranslateConcurrency")) {
                        floatingJson.optInt(
                            "vlTranslateConcurrency",
                            SettingsStore.DEFAULT_FLOATING_OCR_CONCURRENCY
                        )
                    } else {
                        SettingsStore.DEFAULT_FLOATING_OCR_CONCURRENCY
                    }
                ).coerceIn(
                    SettingsStore.MIN_FLOATING_OCR_CONCURRENCY,
                    SettingsStore.MAX_FLOATING_OCR_CONCURRENCY
                ),
                aiApiConcurrencyLimit = floatingJson.optInt(
                    "aiApiConcurrencyLimit",
                    floatingJson.optInt(
                        "vlTranslateConcurrency",
                        SettingsStore.DEFAULT_FLOATING_AI_API_CONCURRENCY
                    )
                ).coerceIn(
                    SettingsStore.MIN_FLOATING_AI_API_CONCURRENCY,
                    SettingsStore.MAX_FLOATING_AI_API_CONCURRENCY
                ),
                proofreadingModeEnabled = floatingJson.optBoolean(
                    "proofreadingModeEnabled",
                    false
                ),
                autoCloseOnScreenChangeEnabled = floatingJson.optBoolean(
                    "autoCloseOnScreenChangeEnabled",
                    false
                ),
                singleTapAction = FloatingBallGestureAction.fromPref(
                    floatingJson.optStringOrNull("singleTapAction"),
                    SettingsStore.DEFAULT_FLOATING_SINGLE_TAP_ACTION
                ),
                doubleTapAction = FloatingBallGestureAction.fromPref(
                    floatingJson.optStringOrNull("doubleTapAction"),
                    SettingsStore.DEFAULT_FLOATING_DOUBLE_TAP_ACTION
                ),
                longPressAction = FloatingBallGestureAction.fromPref(
                    floatingJson.optStringOrNull("longPressAction"),
                    SettingsStore.DEFAULT_FLOATING_LONG_PRESS_ACTION
                ),
                tripleTapAction = FloatingBallGestureAction.fromPref(
                    floatingJson.optStringOrNull("tripleTapAction"),
                    SettingsStore.DEFAULT_FLOATING_TRIPLE_TAP_ACTION
                )
            ),
            llmParameters = LlmParameterSettings(
                temperature = llmJson.optOptionalDouble("temperature"),
                topP = llmJson.optOptionalDouble("topP"),
                topK = llmJson.optOptionalInt("topK"),
                maxOutputTokens = llmJson.optOptionalInt("maxOutputTokens"),
                enableThinking = llmJson.optBoolean(
                    "enableThinking",
                    SettingsStore.DEFAULT_LLM_ENABLE_THINKING
                ),
                thinkingLength = parseThinkingLength(llmJson),
                frequencyPenalty = llmJson.optOptionalDouble("frequencyPenalty"),
                presencePenalty = llmJson.optOptionalDouble("presencePenalty")
            ),
            customRequestParameters = buildList {
                for (index in 0 until customParams.length()) {
                    val param = customParams.optJSONObject(index) ?: continue
                    val key = param.optString("key").trim()
                    val value = param.optString("value")
                    if (key.isBlank() && value.isBlank()) continue
                    add(
                        CustomRequestParameter(
                            key = key,
                            value = value,
                            enabled = param.optBoolean("enabled", true),
                            targetProviderId = param.optString("targetProviderId")
                                .trim()
                                .ifBlank { PRIMARY_PROVIDER_ID }
                        )
                    )
                }
            },
            additionalTranslationProviders = buildList {
                for (index in 0 until additionalProviders.length()) {
                    val providerJson = additionalProviders.optJSONObject(index) ?: continue
                    add(parseAdditionalTranslationProvider(providerJson, index))
                }
            }
        )
    }

    private fun parseThinkingLength(llmJson: JSONObject): ThinkingLength {
        val stored = llmJson.optStringOrNull("thinkingLength")
        if (!stored.isNullOrBlank()) {
            return ThinkingLength.fromPref(stored)
        }
        return ThinkingLength.fromLegacyBudget(llmJson.optOptionalInt("thinkingBudget"))
    }

    private fun parseAdditionalTranslationProvider(
        item: JSONObject,
        index: Int
    ): AdditionalTranslationProvider {
        return AdditionalTranslationProvider(
            name = item.optString("name").trim().ifBlank { defaultAdditionalProviderName(index) },
            apiUrl = item.optString("apiUrl"),
            apiKey = item.optString("apiKey"),
            modelName = item.optString("modelName"),
            weight = item.optInt("weight", 1).coerceAtLeast(1),
            enabled = item.optBoolean("enabled", true)
        )
    }
}

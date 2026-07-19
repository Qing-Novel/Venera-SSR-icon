package com.manga.translate
import com.github.kiastr.venera_ssr.R

import android.content.Context
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow

const val PRIMARY_PROVIDER_ID = "primary"
const val OCR_PROVIDER_ID = "ocr"

data class ApiSettings(
    val apiUrl: String,
    val apiKey: String,
    val modelName: String,
    val apiFormat: ApiFormat = ApiFormat.OPENAI_COMPATIBLE,
    val providerId: String = PRIMARY_PROVIDER_ID
) {
    fun isValid(): Boolean {
        return when (apiFormat) {
            // 谷歌公共翻译走免 Key 公开端点，apiUrl 可空（留空用默认端点）
            ApiFormat.GOOGLE_PUBLIC -> true
            else -> apiUrl.isNotBlank() && apiKey.isNotBlank() && modelName.isNotBlank()
        }
    }
}

data class OcrApiSettings(
    val useLocalOcr: Boolean,
    val japaneseLocalOcrEngine: JapaneseLocalOcrEngine,
    val apiUrl: String,
    val apiKey: String,
    val modelName: String,
    val timeoutSeconds: Int,
    val apiOcrConcurrencyLimit: Int = 1,
    // 0 = auto (determined by device performance); positive = manual override
    val localOcrConcurrencyLimit: Int = 0,
    val ocrApiFormat: OcrApiFormat = OcrApiFormat.OPENAI_COMPATIBLE,
    val secretKey: String = ""
) {
    fun isValid(): Boolean {
        if (!useLocalOcr) {
            when (ocrApiFormat) {
                OcrApiFormat.OPENAI_COMPATIBLE -> {
                    if (apiUrl.isBlank() || apiKey.isBlank() || modelName.isBlank()) return false
                }
                OcrApiFormat.BAIDU_AI -> {
                    if (apiKey.isBlank() || secretKey.isBlank()) return false
                }
            }
        }
        return true
    }
}

enum class JapaneseLocalOcrEngine(
    val prefValue: String
) {
    MANGA_OCR_MOBILE("manga_ocr_mobile");

    companion object {
        fun fromPref(value: String?): JapaneseLocalOcrEngine {
            return entries.firstOrNull { it.prefValue == value } ?: MANGA_OCR_MOBILE
        }
    }
}

data class FloatingTranslateApiSettings(
    val apiUrl: String,
    val apiKey: String,
    val modelName: String,
    val timeoutSeconds: Int,
    val useVlDirectTranslate: Boolean,
    val ocrConcurrencyLimit: Int,
    val aiApiConcurrencyLimit: Int,
    val proofreadingModeEnabled: Boolean,
    val autoCloseOnScreenChangeEnabled: Boolean,
    val singleTapAction: FloatingBallGestureAction,
    val doubleTapAction: FloatingBallGestureAction,
    val longPressAction: FloatingBallGestureAction,
    val tripleTapAction: FloatingBallGestureAction
)

data class NormalBubbleRenderSettings(
    val shrinkPercent: Int,
    val opacityPercent: Int,
    val freeBubbleShrinkPercent: Int,
    val freeBubbleOpacityPercent: Int,
    val minAreaPerCharSp: Float,
    val useHorizontalText: Boolean,
    val autoAdaptFreeBubbleColor: Boolean = true,
    val font: BubbleFont = BubbleFont.SYSTEM_DEFAULT,
    val customFontUrl: String = "",
    val customFontFileName: String = "",
    val isBold: Boolean = false
)

data class BubbleFontSettings(
    val font: BubbleFont = BubbleFont.SYSTEM_DEFAULT,
    val customFontFileName: String = "",
    val isBold: Boolean = false
)

enum class FloatingBubbleShape(val prefValue: String, val labelRes: Int) {
    RECTANGLE("rectangle", R.string.floating_bubble_shape_rectangle),
    INSCRIBED_ELLIPSE("inscribed_ellipse", R.string.floating_bubble_shape_inscribed_ellipse);

    companion object {
        fun fromPref(value: String?): FloatingBubbleShape {
            return entries.firstOrNull { it.prefValue == value } ?: RECTANGLE
        }
    }
}

data class FloatingBubbleRenderSettings(
    val sizeAdjustPercent: Int,
    val opacityPercent: Int,
    val shape: FloatingBubbleShape,
    val useHorizontalText: Boolean,
    val minAreaPerCharSp: Float,
    val autoAdaptBubbleColor: Boolean = true,
    val font: BubbleFont = BubbleFont.SYSTEM_DEFAULT,
    val customFontUrl: String = "",
    val customFontFileName: String = "",
    val isBold: Boolean = false
)

data class CustomRequestParameter(
    val key: String,
    val value: String,
    val enabled: Boolean = true,
    val targetProviderId: String = PRIMARY_PROVIDER_ID
)

data class AiProviderProfile(
    val name: String,
    val mainSettings: ApiSettings,
    val apiTimeoutSeconds: Int,
    val ocrSettings: OcrApiSettings,
    val floatingTranslateSettings: FloatingTranslateApiSettings,
    val llmParameters: LlmParameterSettings,
    val customRequestParameters: List<CustomRequestParameter>,
    val additionalTranslationProviders: List<AdditionalTranslationProvider>
)

data class AiProviderProfilesState(
    val activeProfileName: String?,
    val profiles: List<AiProviderProfile>
)

internal data class SettingsMainForm(
    val apiUrl: String,
    val apiKey: String,
    val modelName: String,
    val apiFormat: ApiFormat,
    val apiTimeoutSeconds: Int,
    val apiRetryCount: Int,
    val maxConcurrency: Int
)

internal data class SettingsPersistenceResult(
    val apiTimeoutSeconds: Int,
    val apiRetryCount: Int,
    val maxConcurrency: Int,
    val concurrencySaved: Boolean
)

class SettingsStore(context: Context) {
    private val storage = SettingsStoreStorage(context)
    private val apiSettingsStore = ApiSettingsStore(storage)
    private val ocrSettingsStore = OcrSettingsStore(storage)
    private val renderSettingsStore = RenderSettingsStore(storage)
    private val appSettingsStore = AppSettingsStore(storage)
    private val llmParameterStore = LlmParameterStore(storage)
    private val providerProfileStore = ProviderProfileStore(
        storage = storage,
        apiSettingsStore = apiSettingsStore,
        ocrSettingsStore = ocrSettingsStore,
        llmParameterStore = llmParameterStore
    )

    val settingsVersion: StateFlow<Long>
        get() = storage.settingsObserver.version

    val settingChanges: SharedFlow<Set<String>>
        get() = storage.settingsObserver.changes

    fun load(): ApiSettings = apiSettingsStore.load()

    fun save(settings: ApiSettings) {
        apiSettingsStore.save(settings)
    }

    fun loadFloatingTranslateApiSettings(): FloatingTranslateApiSettings {
        return apiSettingsStore.loadFloatingTranslateApiSettings()
    }

    fun loadResolvedFloatingTranslateApiSettings(): ApiSettings {
        return apiSettingsStore.loadResolvedFloatingTranslateApiSettings()
    }

    fun saveFloatingTranslateApiSettings(settings: FloatingTranslateApiSettings) {
        apiSettingsStore.saveFloatingTranslateApiSettings(settings)
    }

    fun loadOcrApiSettings(): OcrApiSettings = ocrSettingsStore.loadOcrApiSettings()

    fun saveOcrApiSettings(settings: OcrApiSettings) {
        ocrSettingsStore.saveOcrApiSettings(settings)
    }

    fun loadUseHorizontalText(): Boolean = renderSettingsStore.loadUseHorizontalText()

    fun saveUseHorizontalText(enabled: Boolean) {
        renderSettingsStore.saveUseHorizontalText(enabled)
    }

    fun loadNormalBubbleRenderSettings(): NormalBubbleRenderSettings {
        return renderSettingsStore.loadNormalBubbleRenderSettings()
    }

    fun saveNormalBubbleRenderSettings(settings: NormalBubbleRenderSettings) {
        renderSettingsStore.saveNormalBubbleRenderSettings(settings)
    }

    fun loadFloatingBubbleRenderSettings(): FloatingBubbleRenderSettings {
        return renderSettingsStore.loadFloatingBubbleRenderSettings()
    }

    fun saveFloatingBubbleRenderSettings(settings: FloatingBubbleRenderSettings) {
        renderSettingsStore.saveFloatingBubbleRenderSettings(settings)
    }

    fun loadBubbleFontSettings(): BubbleFontSettings {
        return renderSettingsStore.loadBubbleFontSettings()
    }

    fun saveBubbleFontSettings(settings: BubbleFontSettings) {
        renderSettingsStore.saveBubbleFontSettings(settings)
    }

    fun loadModelIoLogging(): Boolean = apiSettingsStore.loadModelIoLogging()

    fun saveModelIoLogging(enabled: Boolean) {
        apiSettingsStore.saveModelIoLogging(enabled)
    }

    internal fun persistMainSettings(form: SettingsMainForm): SettingsPersistenceResult {
        return apiSettingsStore.persistMainSettings(
            form = form,
            additionalProviderCount = providerProfileStore.countEnabledConfiguredAdditionalProviders()
        )
    }

    fun loadApiRetryCount(): Int = apiSettingsStore.loadApiRetryCount()

    fun saveApiRetryCount(value: Int) {
        apiSettingsStore.saveApiRetryCount(value)
    }

    fun loadMaxConcurrency(): Int = apiSettingsStore.loadMaxConcurrency()

    fun saveMaxConcurrency(value: Int) {
        apiSettingsStore.saveMaxConcurrency(value)
    }

    fun loadApiTimeoutSeconds(): Int = apiSettingsStore.loadApiTimeoutSeconds()

    fun loadApiTimeoutMs(): Int = apiSettingsStore.loadApiTimeoutMs()

    fun saveApiTimeoutSeconds(value: Int) {
        apiSettingsStore.saveApiTimeoutSeconds(value)
    }

    fun loadThemeMode(): ThemeMode = appSettingsStore.loadThemeMode()

    fun loadAppLanguage(): AppLanguage = appSettingsStore.loadAppLanguage()

    fun saveAppLanguage(language: AppLanguage) {
        appSettingsStore.saveAppLanguage(language)
    }

    fun saveThemeMode(mode: ThemeMode) {
        appSettingsStore.saveThemeMode(mode)
    }

    fun loadReadingDisplayMode(): ReadingDisplayMode = appSettingsStore.loadReadingDisplayMode()

    fun saveReadingDisplayMode(mode: ReadingDisplayMode) {
        appSettingsStore.saveReadingDisplayMode(mode)
    }

    fun loadReadingPageAnimationMode(): ReadingPageAnimationMode {
        return appSettingsStore.loadReadingPageAnimationMode()
    }

    fun saveReadingPageAnimationMode(mode: ReadingPageAnimationMode) {
        appSettingsStore.saveReadingPageAnimationMode(mode)
    }

    fun loadBubbleConfThresholdPercent(): Int = renderSettingsStore.loadBubbleConfThresholdPercent()

    fun saveBubbleConfThresholdPercent(value: Int) {
        renderSettingsStore.saveBubbleConfThresholdPercent(value)
    }

    fun loadTranslationBubbleOpacityPercent(): Int {
        return renderSettingsStore.loadTranslationBubbleOpacityPercent()
    }

    fun loadTranslationBubbleOpacity(): Float = renderSettingsStore.loadTranslationBubbleOpacity()

    fun saveTranslationBubbleOpacityPercent(value: Int) {
        renderSettingsStore.saveTranslationBubbleOpacityPercent(value)
    }

    fun loadTranslationStyle(): String = appSettingsStore.loadTranslationStyle()

    fun saveTranslationStyle(style: String) {
        appSettingsStore.saveTranslationStyle(style)
    }

    fun loadLinkSource(): LinkSource = appSettingsStore.loadLinkSource()

    fun saveLinkSource(source: LinkSource) {
        appSettingsStore.saveLinkSource(source)
    }

    fun loadLlmParameters(): LlmParameterSettings = llmParameterStore.loadLlmParameters()

    fun saveLlmParameters(settings: LlmParameterSettings) {
        llmParameterStore.saveLlmParameters(settings)
    }

    fun loadCustomRequestParameters(): List<CustomRequestParameter> {
        return providerProfileStore.loadCustomRequestParameters()
    }

    fun saveCustomRequestParameters(parameters: List<CustomRequestParameter>) {
        providerProfileStore.saveCustomRequestParameters(parameters)
    }

    fun loadAdditionalTranslationProviders(): List<AdditionalTranslationProvider> {
        return providerProfileStore.loadAdditionalTranslationProviders()
    }

    fun saveAdditionalTranslationProviders(providers: List<AdditionalTranslationProvider>) {
        providerProfileStore.saveAdditionalTranslationProviders(providers)
    }

    fun loadMainTranslationProviderPool(): List<WeightedProviderCandidate> {
        return providerProfileStore.loadMainTranslationProviderPool()
    }

    fun loadAiProviderProfilesState(): AiProviderProfilesState {
        return providerProfileStore.loadAiProviderProfilesState()
    }

    fun saveCurrentAsAiProviderProfile(name: String): Boolean {
        return providerProfileStore.saveCurrentAsAiProviderProfile(name)
    }

    fun overwriteActiveAiProviderProfile(): Boolean {
        return providerProfileStore.overwriteActiveAiProviderProfile()
    }

    fun applyAiProviderProfile(name: String): Boolean {
        return providerProfileStore.applyAiProviderProfile(name)
    }

    fun deleteAiProviderProfile(name: String): Boolean {
        return providerProfileStore.deleteAiProviderProfile(name)
    }

    fun defaultAdditionalProviderName(index: Int): String {
        return providerProfileStore.defaultAdditionalProviderName(index)
    }


    fun loadUseLocalTranslation(): Boolean {
        return storage.prefs.getBoolean(SettingsStore.KEY_USE_LOCAL_TRANSLATION, false)
    }

    fun saveUseLocalTranslation(enabled: Boolean) {
        storage.prefs.edit().putBoolean(SettingsStore.KEY_USE_LOCAL_TRANSLATION, enabled).apply()
    }

    fun loadLocalTranslationModelDir(): String {
        return storage.prefs.getString(SettingsStore.KEY_LOCAL_TRANSLATION_MODEL_DIR, SettingsStore.DEFAULT_LOCAL_TRANSLATION_MODEL_DIR) ?: SettingsStore.DEFAULT_LOCAL_TRANSLATION_MODEL_DIR
    }

    fun saveLocalTranslationModelDir(dir: String) {
        storage.prefs.edit().putString(SettingsStore.KEY_LOCAL_TRANSLATION_MODEL_DIR, dir).apply()
    }
    companion object {
        internal const val PREFS_NAME = "manga_translate_settings"
        internal const val AI_PROVIDER_PROFILES_FILE_NAME = "ai_provider_profiles.json"
        internal const val KEY_API_URL = "api_url"
        internal const val KEY_API_KEY = "api_key"
        internal const val KEY_MODEL_NAME = "model_name"
        internal const val KEY_API_FORMAT = "api_format"
        internal const val KEY_OCR_USE_LOCAL = "ocr_use_local"
        internal const val KEY_USE_LOCAL_TRANSLATION = "use_local_translation"
        internal const val KEY_LOCAL_TRANSLATION_MODEL_DIR = "local_translation_model_dir"
        internal const val DEFAULT_LOCAL_TRANSLATION_MODEL_DIR = ""
        internal const val KEY_JAPANESE_LOCAL_OCR_ENGINE = "japanese_local_ocr_engine"
        internal const val KEY_OCR_API_URL = "ocr_api_url"
        internal const val KEY_OCR_API_KEY = "ocr_api_key"
        internal const val KEY_OCR_MODEL_NAME = "ocr_model_name"
        internal const val KEY_FLOATING_API_URL = "floating_api_url"
        internal const val KEY_FLOATING_API_KEY = "floating_api_key"
        internal const val KEY_FLOATING_MODEL_NAME = "floating_model_name"
        internal const val KEY_FLOATING_TIMEOUT_SECONDS = "floating_timeout_seconds"
        internal const val KEY_FLOATING_USE_VL_DIRECT_TRANSLATE = "floating_use_vl_direct_translate"
        internal const val KEY_FLOATING_VL_TRANSLATE_CONCURRENCY = "floating_vl_translate_concurrency"
        internal const val KEY_FLOATING_OCR_CONCURRENCY = "floating_ocr_concurrency"
        internal const val KEY_FLOATING_PROOFREADING_MODE_ENABLED =
            "floating_proofreading_mode_enabled"
        internal const val KEY_FLOATING_AUTO_CLOSE_ON_SCREEN_CHANGE_ENABLED =
            "floating_auto_close_on_screen_change_enabled"
        internal const val KEY_FLOATING_SINGLE_TAP_ACTION = "floating_single_tap_action"
        internal const val KEY_FLOATING_DOUBLE_TAP_ACTION = "floating_double_tap_action"
        internal const val KEY_FLOATING_LONG_PRESS_ACTION = "floating_long_press_action"
        internal const val KEY_FLOATING_TRIPLE_TAP_ACTION = "floating_triple_tap_action"
        internal const val KEY_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT =
            "floating_bubble_size_adjust_percent"
        internal const val KEY_FLOATING_BUBBLE_OPACITY_PERCENT = "floating_bubble_opacity_percent"
        internal const val KEY_FLOATING_BUBBLE_SHAPE = "floating_bubble_shape"
        internal const val KEY_FLOATING_BUBBLE_HORIZONTAL_TEXT = "floating_bubble_horizontal_text"
        internal const val KEY_FLOATING_BUBBLE_MIN_AREA_PER_CHAR_SP =
            "floating_bubble_min_area_per_char_sp"
        internal const val KEY_FLOATING_BUBBLE_AUTO_ADAPT_COLOR =
            "floating_bubble_auto_adapt_color"
        internal const val KEY_OCR_API_TIMEOUT_SECONDS = "ocr_api_timeout_seconds"
        internal const val KEY_OCR_API_CONCURRENCY = "ocr_api_concurrency"
        internal const val KEY_LOCAL_OCR_CONCURRENCY = "local_ocr_concurrency"
        internal const val KEY_OCR_API_FORMAT = "ocr_api_format"
        internal const val KEY_OCR_SECRET_KEY = "ocr_secret_key"
        internal const val KEY_HORIZONTAL_TEXT = "horizontal_text_layout"
        internal const val KEY_NORMAL_BUBBLE_SHRINK_PERCENT = "normal_bubble_shrink_percent"
        internal const val KEY_NORMAL_BUBBLE_MIN_AREA_PER_CHAR_SP =
            "normal_bubble_min_area_per_char_sp"
        internal const val KEY_NORMAL_FREE_BUBBLE_SHRINK_PERCENT =
            "normal_free_bubble_shrink_percent"
        internal const val KEY_NORMAL_FREE_BUBBLE_OPACITY_PERCENT =
            "normal_free_bubble_opacity_percent"
        internal const val KEY_NORMAL_FREE_BUBBLE_AUTO_ADAPT_COLOR =
            "normal_free_bubble_auto_adapt_color"
        internal const val KEY_NORMAL_BUBBLE_FONT = "normal_bubble_font"
        internal const val KEY_NORMAL_BUBBLE_CUSTOM_FONT_URL = "normal_bubble_custom_font_url"
        internal const val KEY_NORMAL_BUBBLE_CUSTOM_FONT_FILE = "normal_bubble_custom_font_file"
        internal const val KEY_NORMAL_BUBBLE_FONT_BOLD = "normal_bubble_font_bold"
        internal const val KEY_FLOATING_BUBBLE_FONT = "floating_bubble_font"
        internal const val KEY_FLOATING_BUBBLE_CUSTOM_FONT_URL = "floating_bubble_custom_font_url"
        internal const val KEY_FLOATING_BUBBLE_CUSTOM_FONT_FILE = "floating_bubble_custom_font_file"
        internal const val KEY_FLOATING_BUBBLE_FONT_BOLD = "floating_bubble_font_bold"
        internal const val KEY_BUBBLE_FONT = "bubble_font"
        internal const val KEY_BUBBLE_CUSTOM_FONT_FILE = "bubble_custom_font_file"
        internal const val KEY_BUBBLE_FONT_BOLD = "bubble_font_bold"
        internal const val KEY_MODEL_IO_LOGGING = "model_io_logging"
        internal const val KEY_API_RETRY_COUNT = "api_retry_count"
        internal const val KEY_MAX_CONCURRENCY = "max_concurrency"
        internal const val KEY_API_TIMEOUT_SECONDS = "api_timeout_seconds"
        internal const val KEY_APP_LANGUAGE = "app_language"
        internal const val KEY_THEME_MODE = "theme_mode"
        internal const val KEY_READING_DISPLAY_MODE = "reading_display_mode"
        internal const val KEY_READING_PAGE_ANIMATION_MODE = "reading_page_animation_mode"
        internal const val KEY_TRANSLATION_BUBBLE_OPACITY_PERCENT =
            "translation_bubble_opacity_percent"
        internal const val KEY_BUBBLE_CONF_THRESHOLD_PERCENT =
            "manga109_bubble_conf_threshold_percent_v2"
        internal const val KEY_LINK_SOURCE = "link_source"
        internal const val KEY_LLM_TEMPERATURE = "llm_temperature"
        internal const val KEY_LLM_TOP_P = "llm_top_p"
        internal const val KEY_LLM_TOP_K = "llm_top_k"
        internal const val KEY_LLM_MAX_OUTPUT_TOKENS = "llm_max_output_tokens"
        internal const val KEY_ADDITIONAL_TRANSLATION_PROVIDERS =
            "additional_translation_providers"
        internal const val KEY_LLM_ENABLE_THINKING = "llm_enable_thinking"
        internal const val KEY_LLM_THINKING_BUDGET = "llm_thinking_budget"
        internal const val KEY_LLM_THINKING_LENGTH = "llm_thinking_length"
        internal const val KEY_LLM_FREQUENCY_PENALTY = "llm_frequency_penalty"
        internal const val KEY_LLM_PRESENCE_PENALTY = "llm_presence_penalty"
        internal const val KEY_CUSTOM_REQUEST_PARAMETERS = "custom_request_parameters"
        internal const val KEY_TRANSLATION_STYLE = "translation_style"
        internal const val KEY_AI_PROVIDER_PROFILES_STATE = "ai_provider_profiles_state"
        internal const val LEGACY_SETTINGS_JSON_VERSION = 1
        internal const val SETTINGS_JSON_SCHEMA_VERSION = 2
        internal const val PRIMARY_PROVIDER_WEIGHT = 10
        internal const val DEFAULT_LLM_TEMPERATURE = 0.8
        internal const val DEFAULT_LLM_TOP_P = 1.0
        internal const val DEFAULT_LLM_ENABLE_THINKING = false
        internal const val DEFAULT_TRANSLATION_STYLE =
            "请以普通日漫翻译风格翻译，语言自然流畅，符合中文漫画阅读习惯。"
        internal const val DEFAULT_TRANSLATION_STYLE_HANT =
            "請以普通日漫翻譯風格翻譯，語言自然流暢，符合中文漫畫閱讀習慣。"
        internal const val DEFAULT_API_URL = "https://api.siliconflow.cn/v1"
        internal const val DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
        internal const val DEFAULT_OCR_API_URL = "https://api.siliconflow.cn/v1"
        internal const val DEFAULT_OCR_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
        internal const val DEFAULT_OCR_API_TIMEOUT_SECONDS = 300
        const val MIN_OCR_API_TIMEOUT_SECONDS = 30
        const val MAX_OCR_API_TIMEOUT_SECONDS = 1200
        internal const val DEFAULT_OCR_API_CONCURRENCY = 1
        const val MIN_OCR_API_CONCURRENCY = 1
        const val MAX_OCR_API_CONCURRENCY = 50
        internal const val DEFAULT_LOCAL_OCR_CONCURRENCY = 0
        internal const val MIN_LOCAL_OCR_CONCURRENCY = 0
        internal const val MAX_LOCAL_OCR_CONCURRENCY = 8
        internal const val DEFAULT_FLOATING_OCR_CONCURRENCY = 1
        internal const val MIN_FLOATING_OCR_CONCURRENCY = 1
        internal const val MAX_FLOATING_OCR_CONCURRENCY = 50
        internal const val DEFAULT_FLOATING_AI_API_CONCURRENCY = 15
        internal const val MIN_FLOATING_AI_API_CONCURRENCY = 1
        internal const val MAX_FLOATING_AI_API_CONCURRENCY = 50
        internal val DEFAULT_FLOATING_SINGLE_TAP_ACTION = FloatingBallGestureAction.START_TRANSLATE
        internal val DEFAULT_FLOATING_DOUBLE_TAP_ACTION = FloatingBallGestureAction.CLEAR_SCREEN
        internal val DEFAULT_FLOATING_LONG_PRESS_ACTION = FloatingBallGestureAction.OPEN_MENU
        internal val DEFAULT_FLOATING_TRIPLE_TAP_ACTION = FloatingBallGestureAction.NONE
        internal const val DEFAULT_FLOATING_API_TIMEOUT_SECONDS = 300
        const val MIN_FLOATING_API_TIMEOUT_SECONDS = 30
        const val MAX_FLOATING_API_TIMEOUT_SECONDS = 1200
        internal const val DEFAULT_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT = 0
        internal const val MIN_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT = -30
        internal const val MAX_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT = 30
        internal const val DEFAULT_FLOATING_MIN_AREA_PER_CHAR_SP = 48f
        internal const val DEFAULT_FLOATING_BUBBLE_AUTO_ADAPT_COLOR = true
        internal const val MIN_FLOATING_MIN_AREA_PER_CHAR_SP = 16f
        internal const val MAX_FLOATING_MIN_AREA_PER_CHAR_SP = 256f
        internal const val DEFAULT_NORMAL_BUBBLE_SHRINK_PERCENT = 10
        internal const val MIN_NORMAL_BUBBLE_SHRINK_PERCENT = 0
        internal const val MAX_NORMAL_BUBBLE_SHRINK_PERCENT = 30
        internal const val DEFAULT_NORMAL_FREE_BUBBLE_SHRINK_PERCENT = 10
        internal const val DEFAULT_NORMAL_FREE_BUBBLE_OPACITY_PERCENT = 90
        internal const val DEFAULT_NORMAL_FREE_BUBBLE_AUTO_ADAPT_COLOR = true
        internal const val DEFAULT_NORMAL_MIN_AREA_PER_CHAR_SP = 48f
        internal const val MIN_NORMAL_MIN_AREA_PER_CHAR_SP = 16f
        internal const val MAX_NORMAL_MIN_AREA_PER_CHAR_SP = 256f
        internal const val DEFAULT_MAX_CONCURRENCY = 3
        internal const val MIN_MAX_CONCURRENCY = 1
        internal const val MAX_MAX_CONCURRENCY = 200
        internal const val DEFAULT_API_RETRY_COUNT = 3
        internal const val MIN_API_RETRY_COUNT = 1
        internal const val MAX_API_RETRY_COUNT = 50
        internal const val DEFAULT_API_TIMEOUT_SECONDS = 300
        internal const val MIN_API_TIMEOUT_SECONDS = 30
        internal const val MAX_API_TIMEOUT_SECONDS = 1200
        internal const val DEFAULT_TRANSLATION_BUBBLE_OPACITY_PERCENT = 100
        internal const val MIN_TRANSLATION_BUBBLE_OPACITY_PERCENT = 0
        internal const val MAX_TRANSLATION_BUBBLE_OPACITY_PERCENT = 100
        internal const val DEFAULT_BUBBLE_CONF_THRESHOLD_PERCENT = 10
        internal const val MIN_BUBBLE_CONF_THRESHOLD_PERCENT = 1
        internal const val MAX_BUBBLE_CONF_THRESHOLD_PERCENT = 95
    }
}

data class LlmParameterSettings(
    val temperature: Double?,
    val topP: Double?,
    val topK: Int?,
    val maxOutputTokens: Int?,
    val enableThinking: Boolean,
    val thinkingLength: ThinkingLength = ThinkingLength.DEFAULT,
    val frequencyPenalty: Double?,
    val presencePenalty: Double?
)

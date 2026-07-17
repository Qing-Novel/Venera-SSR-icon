package com.manga.translate

internal class AppSettingsStore(
    private val storage: SettingsStoreStorage
) {
    fun loadThemeMode(): ThemeMode {
        val saved = storage.prefs.getString(
            SettingsStore.KEY_THEME_MODE,
            ThemeMode.FOLLOW_SYSTEM.prefValue
        )
        return ThemeMode.fromPref(saved)
    }

    fun loadAppLanguage(): AppLanguage {
        val saved = storage.prefs.getString(
            SettingsStore.KEY_APP_LANGUAGE,
            AppLanguage.FOLLOW_SYSTEM.prefValue
        )
        return AppLanguage.fromPref(saved)
    }

    fun saveAppLanguage(language: AppLanguage) {
        storage.editSettings(setOf(SettingsStore.KEY_APP_LANGUAGE)) {
            putString(SettingsStore.KEY_APP_LANGUAGE, language.prefValue)
        }
    }

    fun saveThemeMode(mode: ThemeMode) {
        storage.editSettings(setOf(SettingsStore.KEY_THEME_MODE)) {
            putString(SettingsStore.KEY_THEME_MODE, mode.prefValue)
        }
    }

    fun loadReadingDisplayMode(): ReadingDisplayMode {
        val saved = storage.prefs.getString(
            SettingsStore.KEY_READING_DISPLAY_MODE,
            ReadingDisplayMode.FIT_WIDTH.prefValue
        )
        return ReadingDisplayMode.fromPref(saved)
    }

    fun saveReadingDisplayMode(mode: ReadingDisplayMode) {
        storage.editSettings(setOf(SettingsStore.KEY_READING_DISPLAY_MODE)) {
            putString(SettingsStore.KEY_READING_DISPLAY_MODE, mode.prefValue)
        }
    }

    fun loadReadingPageAnimationMode(): ReadingPageAnimationMode {
        val saved = storage.prefs.getString(
            SettingsStore.KEY_READING_PAGE_ANIMATION_MODE,
            ReadingPageAnimationMode.HORIZONTAL_SLIDE.prefValue
        )
        return ReadingPageAnimationMode.fromPref(saved)
    }

    fun saveReadingPageAnimationMode(mode: ReadingPageAnimationMode) {
        storage.editSettings(setOf(SettingsStore.KEY_READING_PAGE_ANIMATION_MODE)) {
            putString(SettingsStore.KEY_READING_PAGE_ANIMATION_MODE, mode.prefValue)
        }
    }

    fun loadTranslationStyle(): String {
        val saved = storage.prefs.getString(SettingsStore.KEY_TRANSLATION_STYLE, null)
        if (!saved.isNullOrBlank()) return saved
        return if (PromptAssetResolver.isTraditionalChinese(storage.appContext)) {
            SettingsStore.DEFAULT_TRANSLATION_STYLE_HANT
        } else {
            SettingsStore.DEFAULT_TRANSLATION_STYLE
        }
    }

    fun saveTranslationStyle(style: String) {
        storage.editSettings(setOf(SettingsStore.KEY_TRANSLATION_STYLE)) {
            putString(SettingsStore.KEY_TRANSLATION_STYLE, style.trim())
        }
    }

    fun loadLinkSource(): LinkSource {
        val saved = storage.prefs.getString(
            SettingsStore.KEY_LINK_SOURCE,
            LinkSource.GITHUB.prefValue
        )
        return LinkSource.fromPref(saved)
    }

    fun saveLinkSource(source: LinkSource) {
        storage.editSettings(setOf(SettingsStore.KEY_LINK_SOURCE)) {
            putString(SettingsStore.KEY_LINK_SOURCE, source.prefValue)
        }
    }
}

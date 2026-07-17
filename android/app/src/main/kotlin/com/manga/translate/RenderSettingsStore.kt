package com.manga.translate

internal class RenderSettingsStore(
    private val storage: SettingsStoreStorage
) {
    fun loadBubbleFontSettings(): BubbleFontSettings {
        if (
            storage.prefs.contains(SettingsStore.KEY_BUBBLE_FONT) ||
            storage.prefs.contains(SettingsStore.KEY_BUBBLE_CUSTOM_FONT_FILE) ||
            storage.prefs.contains(SettingsStore.KEY_BUBBLE_FONT_BOLD)
        ) {
            return BubbleFontSettings(
                font = BubbleFont.fromPref(
                    storage.prefs.getString(SettingsStore.KEY_BUBBLE_FONT, null)
                ),
                customFontFileName = storage.prefs.getString(
                    SettingsStore.KEY_BUBBLE_CUSTOM_FONT_FILE,
                    ""
                ) ?: "",
                isBold = storage.prefs.getBoolean(SettingsStore.KEY_BUBBLE_FONT_BOLD, false)
            )
        }

        val normalFont = BubbleFont.fromPref(
            storage.prefs.getString(SettingsStore.KEY_NORMAL_BUBBLE_FONT, null)
        )
        val floatingFont = BubbleFont.fromPref(
            storage.prefs.getString(SettingsStore.KEY_FLOATING_BUBBLE_FONT, null)
        )
        val customFileName = storage.prefs.getString(
            SettingsStore.KEY_NORMAL_BUBBLE_CUSTOM_FONT_FILE,
            ""
        )?.takeIf { it.isNotBlank() }
            ?: storage.prefs.getString(SettingsStore.KEY_FLOATING_BUBBLE_CUSTOM_FONT_FILE, "")
            ?: ""
        val isBold = storage.prefs.getBoolean(SettingsStore.KEY_NORMAL_BUBBLE_FONT_BOLD, false) ||
            storage.prefs.getBoolean(SettingsStore.KEY_FLOATING_BUBBLE_FONT_BOLD, false)
        val selectedFont = when {
            normalFont == BubbleFont.CUSTOM_FILE || floatingFont == BubbleFont.CUSTOM_FILE ->
                BubbleFont.CUSTOM_FILE
            else -> BubbleFont.SYSTEM_DEFAULT
        }
        return BubbleFontSettings(
            font = selectedFont,
            customFontFileName = customFileName,
            isBold = isBold
        )
    }

    fun saveBubbleFontSettings(settings: BubbleFontSettings) {
        storage.editSettings(
            setOf(
                SettingsStore.KEY_BUBBLE_FONT,
                SettingsStore.KEY_BUBBLE_CUSTOM_FONT_FILE,
                SettingsStore.KEY_BUBBLE_FONT_BOLD
            )
        ) {
            putString(SettingsStore.KEY_BUBBLE_FONT, settings.font.prefValue)
                .putString(
                    SettingsStore.KEY_BUBBLE_CUSTOM_FONT_FILE,
                    settings.customFontFileName
                )
                .putBoolean(SettingsStore.KEY_BUBBLE_FONT_BOLD, settings.isBold)
        }
    }

    fun loadUseHorizontalText(): Boolean {
        return storage.prefs.getBoolean(SettingsStore.KEY_HORIZONTAL_TEXT, true)
    }

    fun saveUseHorizontalText(enabled: Boolean) {
        storage.editSettings(setOf(SettingsStore.KEY_HORIZONTAL_TEXT)) {
            putBoolean(SettingsStore.KEY_HORIZONTAL_TEXT, enabled)
        }
    }

    fun loadNormalBubbleRenderSettings(): NormalBubbleRenderSettings {
        val fontSettings = loadBubbleFontSettings()
        return NormalBubbleRenderSettings(
            shrinkPercent = storage.prefs.getInt(
                SettingsStore.KEY_NORMAL_BUBBLE_SHRINK_PERCENT,
                SettingsStore.DEFAULT_NORMAL_BUBBLE_SHRINK_PERCENT
            ).coerceIn(
                SettingsStore.MIN_NORMAL_BUBBLE_SHRINK_PERCENT,
                SettingsStore.MAX_NORMAL_BUBBLE_SHRINK_PERCENT
            ),
            opacityPercent = loadTranslationBubbleOpacityPercent(),
            freeBubbleShrinkPercent = storage.prefs.getInt(
                SettingsStore.KEY_NORMAL_FREE_BUBBLE_SHRINK_PERCENT,
                SettingsStore.DEFAULT_NORMAL_FREE_BUBBLE_SHRINK_PERCENT
            ).coerceIn(
                SettingsStore.MIN_NORMAL_BUBBLE_SHRINK_PERCENT,
                SettingsStore.MAX_NORMAL_BUBBLE_SHRINK_PERCENT
            ),
            freeBubbleOpacityPercent = storage.prefs.getInt(
                SettingsStore.KEY_NORMAL_FREE_BUBBLE_OPACITY_PERCENT,
                SettingsStore.DEFAULT_NORMAL_FREE_BUBBLE_OPACITY_PERCENT
            ).coerceIn(
                SettingsStore.MIN_TRANSLATION_BUBBLE_OPACITY_PERCENT,
                SettingsStore.MAX_TRANSLATION_BUBBLE_OPACITY_PERCENT
            ),
            minAreaPerCharSp = storage.prefs.getFloat(
                SettingsStore.KEY_NORMAL_BUBBLE_MIN_AREA_PER_CHAR_SP,
                SettingsStore.DEFAULT_NORMAL_MIN_AREA_PER_CHAR_SP
            ).coerceIn(
                SettingsStore.MIN_NORMAL_MIN_AREA_PER_CHAR_SP,
                SettingsStore.MAX_NORMAL_MIN_AREA_PER_CHAR_SP
            ),
            useHorizontalText = loadUseHorizontalText(),
            autoAdaptFreeBubbleColor = storage.prefs.getBoolean(
                SettingsStore.KEY_NORMAL_FREE_BUBBLE_AUTO_ADAPT_COLOR,
                SettingsStore.DEFAULT_NORMAL_FREE_BUBBLE_AUTO_ADAPT_COLOR
            ),
            font = fontSettings.font,
            customFontUrl = "",
            customFontFileName = fontSettings.customFontFileName,
            isBold = fontSettings.isBold
        )
    }

    fun saveNormalBubbleRenderSettings(settings: NormalBubbleRenderSettings) {
        storage.editSettings(
            setOf(
                SettingsStore.KEY_NORMAL_BUBBLE_SHRINK_PERCENT,
                SettingsStore.KEY_TRANSLATION_BUBBLE_OPACITY_PERCENT,
                SettingsStore.KEY_NORMAL_BUBBLE_MIN_AREA_PER_CHAR_SP,
                SettingsStore.KEY_NORMAL_FREE_BUBBLE_SHRINK_PERCENT,
                SettingsStore.KEY_NORMAL_FREE_BUBBLE_OPACITY_PERCENT,
                SettingsStore.KEY_NORMAL_FREE_BUBBLE_AUTO_ADAPT_COLOR,
                SettingsStore.KEY_HORIZONTAL_TEXT
            )
        ) {
            putInt(
                SettingsStore.KEY_NORMAL_BUBBLE_SHRINK_PERCENT,
                settings.shrinkPercent.coerceIn(
                    SettingsStore.MIN_NORMAL_BUBBLE_SHRINK_PERCENT,
                    SettingsStore.MAX_NORMAL_BUBBLE_SHRINK_PERCENT
                )
            )
                .putInt(
                    SettingsStore.KEY_TRANSLATION_BUBBLE_OPACITY_PERCENT,
                    settings.opacityPercent.coerceIn(
                        SettingsStore.MIN_TRANSLATION_BUBBLE_OPACITY_PERCENT,
                        SettingsStore.MAX_TRANSLATION_BUBBLE_OPACITY_PERCENT
                    )
                )
                .putFloat(
                    SettingsStore.KEY_NORMAL_BUBBLE_MIN_AREA_PER_CHAR_SP,
                    settings.minAreaPerCharSp.coerceIn(
                        SettingsStore.MIN_NORMAL_MIN_AREA_PER_CHAR_SP,
                        SettingsStore.MAX_NORMAL_MIN_AREA_PER_CHAR_SP
                    )
                )
                .putInt(
                    SettingsStore.KEY_NORMAL_FREE_BUBBLE_SHRINK_PERCENT,
                    settings.freeBubbleShrinkPercent.coerceIn(
                        SettingsStore.MIN_NORMAL_BUBBLE_SHRINK_PERCENT,
                        SettingsStore.MAX_NORMAL_BUBBLE_SHRINK_PERCENT
                    )
                )
                .putInt(
                    SettingsStore.KEY_NORMAL_FREE_BUBBLE_OPACITY_PERCENT,
                    settings.freeBubbleOpacityPercent.coerceIn(
                        SettingsStore.MIN_TRANSLATION_BUBBLE_OPACITY_PERCENT,
                        SettingsStore.MAX_TRANSLATION_BUBBLE_OPACITY_PERCENT
                    )
                )
                .putBoolean(SettingsStore.KEY_HORIZONTAL_TEXT, settings.useHorizontalText)
                .putBoolean(
                    SettingsStore.KEY_NORMAL_FREE_BUBBLE_AUTO_ADAPT_COLOR,
                    settings.autoAdaptFreeBubbleColor
                )
        }
    }

    fun loadFloatingBubbleRenderSettings(): FloatingBubbleRenderSettings {
        val fontSettings = loadBubbleFontSettings()
        return FloatingBubbleRenderSettings(
            sizeAdjustPercent = storage.prefs.getInt(
                SettingsStore.KEY_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT,
                SettingsStore.DEFAULT_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT
            ).coerceIn(
                SettingsStore.MIN_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT,
                SettingsStore.MAX_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT
            ),
            opacityPercent = storage.prefs.getInt(
                SettingsStore.KEY_FLOATING_BUBBLE_OPACITY_PERCENT,
                loadTranslationBubbleOpacityPercent()
            ).coerceIn(
                SettingsStore.MIN_TRANSLATION_BUBBLE_OPACITY_PERCENT,
                SettingsStore.MAX_TRANSLATION_BUBBLE_OPACITY_PERCENT
            ),
            shape = FloatingBubbleShape.fromPref(
                storage.prefs.getString(
                    SettingsStore.KEY_FLOATING_BUBBLE_SHAPE,
                    FloatingBubbleShape.RECTANGLE.prefValue
                )
            ),
            useHorizontalText = storage.prefs.getBoolean(
                SettingsStore.KEY_FLOATING_BUBBLE_HORIZONTAL_TEXT,
                true
            ),
            minAreaPerCharSp = storage.prefs.getFloat(
                SettingsStore.KEY_FLOATING_BUBBLE_MIN_AREA_PER_CHAR_SP,
                SettingsStore.DEFAULT_FLOATING_MIN_AREA_PER_CHAR_SP
            ).coerceIn(
                SettingsStore.MIN_FLOATING_MIN_AREA_PER_CHAR_SP,
                SettingsStore.MAX_FLOATING_MIN_AREA_PER_CHAR_SP
            ),
            autoAdaptBubbleColor = storage.prefs.getBoolean(
                SettingsStore.KEY_FLOATING_BUBBLE_AUTO_ADAPT_COLOR,
                SettingsStore.DEFAULT_FLOATING_BUBBLE_AUTO_ADAPT_COLOR
            ),
            font = fontSettings.font,
            customFontUrl = "",
            customFontFileName = fontSettings.customFontFileName,
            isBold = fontSettings.isBold
        )
    }

    fun saveFloatingBubbleRenderSettings(settings: FloatingBubbleRenderSettings) {
        storage.editSettings(
            setOf(
                SettingsStore.KEY_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT,
                SettingsStore.KEY_FLOATING_BUBBLE_OPACITY_PERCENT,
                SettingsStore.KEY_FLOATING_BUBBLE_SHAPE,
                SettingsStore.KEY_FLOATING_BUBBLE_HORIZONTAL_TEXT,
                SettingsStore.KEY_FLOATING_BUBBLE_MIN_AREA_PER_CHAR_SP,
                SettingsStore.KEY_FLOATING_BUBBLE_AUTO_ADAPT_COLOR
            )
        ) {
            putInt(
                SettingsStore.KEY_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT,
                settings.sizeAdjustPercent.coerceIn(
                    SettingsStore.MIN_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT,
                    SettingsStore.MAX_FLOATING_BUBBLE_SIZE_ADJUST_PERCENT
                )
            )
                .putInt(
                    SettingsStore.KEY_FLOATING_BUBBLE_OPACITY_PERCENT,
                    settings.opacityPercent.coerceIn(
                        SettingsStore.MIN_TRANSLATION_BUBBLE_OPACITY_PERCENT,
                        SettingsStore.MAX_TRANSLATION_BUBBLE_OPACITY_PERCENT
                    )
                )
                .putString(SettingsStore.KEY_FLOATING_BUBBLE_SHAPE, settings.shape.prefValue)
                .putBoolean(
                    SettingsStore.KEY_FLOATING_BUBBLE_HORIZONTAL_TEXT,
                    settings.useHorizontalText
                )
                .putFloat(
                    SettingsStore.KEY_FLOATING_BUBBLE_MIN_AREA_PER_CHAR_SP,
                    settings.minAreaPerCharSp.coerceIn(
                        SettingsStore.MIN_FLOATING_MIN_AREA_PER_CHAR_SP,
                        SettingsStore.MAX_FLOATING_MIN_AREA_PER_CHAR_SP
                    )
                )
                .putBoolean(
                    SettingsStore.KEY_FLOATING_BUBBLE_AUTO_ADAPT_COLOR,
                    settings.autoAdaptBubbleColor
                )
        }
    }

    fun loadBubbleConfThresholdPercent(): Int {
        val saved = storage.prefs.getInt(
            SettingsStore.KEY_BUBBLE_CONF_THRESHOLD_PERCENT,
            SettingsStore.DEFAULT_BUBBLE_CONF_THRESHOLD_PERCENT
        )
        return saved.coerceIn(
            SettingsStore.MIN_BUBBLE_CONF_THRESHOLD_PERCENT,
            SettingsStore.MAX_BUBBLE_CONF_THRESHOLD_PERCENT
        )
    }

    fun saveBubbleConfThresholdPercent(value: Int) {
        val normalized = value.coerceIn(
            SettingsStore.MIN_BUBBLE_CONF_THRESHOLD_PERCENT,
            SettingsStore.MAX_BUBBLE_CONF_THRESHOLD_PERCENT
        )
        storage.editSettings(setOf(SettingsStore.KEY_BUBBLE_CONF_THRESHOLD_PERCENT)) {
            putInt(SettingsStore.KEY_BUBBLE_CONF_THRESHOLD_PERCENT, normalized)
        }
    }

    fun loadTranslationBubbleOpacityPercent(): Int {
        val saved = storage.prefs.getInt(
            SettingsStore.KEY_TRANSLATION_BUBBLE_OPACITY_PERCENT,
            SettingsStore.DEFAULT_TRANSLATION_BUBBLE_OPACITY_PERCENT
        )
        return saved.coerceIn(
            SettingsStore.MIN_TRANSLATION_BUBBLE_OPACITY_PERCENT,
            SettingsStore.MAX_TRANSLATION_BUBBLE_OPACITY_PERCENT
        )
    }

    fun loadTranslationBubbleOpacity(): Float {
        return loadTranslationBubbleOpacityPercent() / 100f
    }

    fun saveTranslationBubbleOpacityPercent(value: Int) {
        val normalized = value.coerceIn(
            SettingsStore.MIN_TRANSLATION_BUBBLE_OPACITY_PERCENT,
            SettingsStore.MAX_TRANSLATION_BUBBLE_OPACITY_PERCENT
        )
        storage.editSettings(setOf(SettingsStore.KEY_TRANSLATION_BUBBLE_OPACITY_PERCENT)) {
            putInt(SettingsStore.KEY_TRANSLATION_BUBBLE_OPACITY_PERCENT, normalized)
        }
    }
}

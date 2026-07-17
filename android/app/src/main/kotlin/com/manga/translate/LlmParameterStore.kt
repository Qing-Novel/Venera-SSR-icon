package com.manga.translate

internal class LlmParameterStore(
    private val storage: SettingsStoreStorage
) {
    fun loadLlmParameters(): LlmParameterSettings {
        return LlmParameterSettings(
            temperature = storage.readDoubleWithDefault(
                SettingsStore.KEY_LLM_TEMPERATURE,
                SettingsStore.DEFAULT_LLM_TEMPERATURE
            ),
            topP = storage.readDoubleWithDefault(
                SettingsStore.KEY_LLM_TOP_P,
                SettingsStore.DEFAULT_LLM_TOP_P
            ),
            topK = storage.readIntOptional(SettingsStore.KEY_LLM_TOP_K),
            maxOutputTokens = storage.readIntOptional(SettingsStore.KEY_LLM_MAX_OUTPUT_TOKENS),
            enableThinking = storage.prefs.getBoolean(
                SettingsStore.KEY_LLM_ENABLE_THINKING,
                SettingsStore.DEFAULT_LLM_ENABLE_THINKING
            ),
            thinkingLength = loadThinkingLength(),
            frequencyPenalty = storage.readDoubleOptional(SettingsStore.KEY_LLM_FREQUENCY_PENALTY),
            presencePenalty = storage.readDoubleOptional(SettingsStore.KEY_LLM_PRESENCE_PENALTY)
        )
    }

    fun saveLlmParameters(settings: LlmParameterSettings) {
        storage.editSettings(
            setOf(
                SettingsStore.KEY_LLM_TEMPERATURE,
                SettingsStore.KEY_LLM_TOP_P,
                SettingsStore.KEY_LLM_TOP_K,
                SettingsStore.KEY_LLM_MAX_OUTPUT_TOKENS,
                SettingsStore.KEY_LLM_ENABLE_THINKING,
                SettingsStore.KEY_LLM_THINKING_LENGTH,
                SettingsStore.KEY_LLM_THINKING_BUDGET,
                SettingsStore.KEY_LLM_FREQUENCY_PENALTY,
                SettingsStore.KEY_LLM_PRESENCE_PENALTY
            )
        ) {
            putOptionalNumber(SettingsStore.KEY_LLM_TEMPERATURE, settings.temperature)
                .putOptionalNumber(SettingsStore.KEY_LLM_TOP_P, settings.topP)
                .putOptionalNumber(SettingsStore.KEY_LLM_TOP_K, settings.topK)
                .putOptionalNumber(SettingsStore.KEY_LLM_MAX_OUTPUT_TOKENS, settings.maxOutputTokens)
                .putBoolean(SettingsStore.KEY_LLM_ENABLE_THINKING, settings.enableThinking)
                .putString(SettingsStore.KEY_LLM_THINKING_LENGTH, settings.thinkingLength.prefValue)
                .remove(SettingsStore.KEY_LLM_THINKING_BUDGET)
                .putOptionalNumber(
                    SettingsStore.KEY_LLM_FREQUENCY_PENALTY,
                    settings.frequencyPenalty
                )
                .putOptionalNumber(
                    SettingsStore.KEY_LLM_PRESENCE_PENALTY,
                    settings.presencePenalty
                )
        }
    }

    private fun loadThinkingLength(): ThinkingLength {
        val stored = storage.prefs.getString(SettingsStore.KEY_LLM_THINKING_LENGTH, null)
        if (!stored.isNullOrBlank()) {
            return ThinkingLength.fromPref(stored)
        }
        return ThinkingLength.fromLegacyBudget(
            storage.readIntOptional(SettingsStore.KEY_LLM_THINKING_BUDGET)
        )
    }
}

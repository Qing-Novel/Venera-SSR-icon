package com.manga.translate

import androidx.annotation.StringRes

enum class ThinkingLength(
    val prefValue: String,
    @param:StringRes val labelRes: Int
) {
    AUTO("auto", R.string.thinking_length_auto),
    LOW("low", R.string.thinking_length_low),
    MEDIUM("medium", R.string.thinking_length_medium),
    HIGH("high", R.string.thinking_length_high),
    XHIGH("xhigh", R.string.thinking_length_xhigh);

    fun openAiBudgetTokens(): Int {
        return when (this) {
            AUTO -> DEFAULT_OPENAI_MEDIUM
            LOW -> 1024
            MEDIUM -> DEFAULT_OPENAI_MEDIUM
            HIGH -> 8192
            XHIGH -> 16384
        }
    }

    fun geminiThinkingBudget(): Int {
        return when (this) {
            AUTO -> -1
            LOW -> 1024
            MEDIUM -> 8192
            HIGH -> 24576
            XHIGH -> 32768
        }
    }

    companion object {
        val DEFAULT = MEDIUM
        private const val DEFAULT_OPENAI_MEDIUM = 4096

        fun fromPref(value: String?): ThinkingLength {
            return entries.firstOrNull { it.prefValue == value } ?: DEFAULT
        }

        fun optionsFor(apiFormat: ApiFormat): List<ThinkingLength> {
            return when (apiFormat) {
                ApiFormat.GEMINI -> entries.toList()
                ApiFormat.OPENAI_COMPATIBLE,
                ApiFormat.OPENAI_RESPONSES -> entries.filter { it != AUTO }
            }
        }

        fun fromLegacyBudget(budget: Int?): ThinkingLength {
            if (budget == null) return DEFAULT
            if (budget < 0) return AUTO
            return entries
                .filter { it != AUTO }
                .minByOrNull { kotlin.math.abs(it.openAiBudgetTokens() - budget) }
                ?: DEFAULT
        }
    }
}

package com.manga.translate
import com.github.kiastr.venera_ssr.R

import androidx.annotation.StringRes

enum class ApiFormat(
    val prefValue: String,
    @param:StringRes val labelRes: Int
) {
    OPENAI_COMPATIBLE("openai_compatible", R.string.api_format_openai_compatible),
    OPENAI_RESPONSES("openai_responses", R.string.api_format_openai_responses),
    GEMINI("gemini", R.string.api_format_gemini);

    companion object {
        fun fromPref(value: String?): ApiFormat {
            return entries.firstOrNull { it.prefValue == value } ?: OPENAI_COMPATIBLE
        }
    }

    val usesOpenAiAuth: Boolean
        get() = this == OPENAI_COMPATIBLE || this == OPENAI_RESPONSES
}

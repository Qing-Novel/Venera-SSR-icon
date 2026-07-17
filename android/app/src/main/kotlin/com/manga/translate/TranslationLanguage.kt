package com.manga.translate
import com.github.kiastr.venera_ssr.R

import android.content.Context
import androidx.annotation.StringRes

enum class TranslationLanguage(
    val prefValue: String,
    @param:StringRes val sourceNameResId: Int,
    val baiduLanguageType: String
) {
    JA_TO_ZH("ja_to_zh", R.string.translation_source_japanese, "JAP"),
    EN_TO_ZH("en_to_zh", R.string.translation_source_english, "ENG"),
    KO_TO_ZH("ko_to_zh", R.string.translation_source_korean, "KOR"),
    ZH_HANS_TO_TARGET("zh_hans_to_target", R.string.translation_source_simplified_chinese, "CHN_ENG"),
    ZH_HANT_TO_TARGET("zh_hant_to_target", R.string.translation_source_traditional_chinese, "CHN_ENG"),
    CHN_ENG_TO_ZH("chn_eng_to_zh", R.string.translation_source_mixed_chinese_english, "CHN_ENG"),
    FR_TO_ZH("fr_to_zh", R.string.translation_source_french, "FRE"),
    ES_TO_ZH("es_to_zh", R.string.translation_source_spanish, "SPA"),
    PT_TO_ZH("pt_to_zh", R.string.translation_source_portuguese, "POR"),
    DE_TO_ZH("de_to_zh", R.string.translation_source_german, "GER"),
    IT_TO_ZH("it_to_zh", R.string.translation_source_italian, "ITA"),
    RU_TO_ZH("ru_to_zh", R.string.translation_source_russian, "RUS");

    fun displayName(context: Context): String {
        return context.getString(
            R.string.translation_language_pair_format,
            context.getString(sourceNameResId),
            PromptAssetResolver.translationTargetDisplayName(context)
        )
    }

    fun supportsLocalOcr(): Boolean {
        return when (this) {
            JA_TO_ZH, EN_TO_ZH, KO_TO_ZH, FR_TO_ZH, ES_TO_ZH, PT_TO_ZH, DE_TO_ZH, IT_TO_ZH -> true
            ZH_HANS_TO_TARGET, ZH_HANT_TO_TARGET, CHN_ENG_TO_ZH -> true
            RU_TO_ZH -> false
        }
    }

    fun usesLatinOcr(): Boolean {
        return this in setOf(EN_TO_ZH, FR_TO_ZH, ES_TO_ZH, PT_TO_ZH, DE_TO_ZH, IT_TO_ZH)
    }

    companion object {
        fun fromPref(value: String?): TranslationLanguage {
            return entries.firstOrNull { it.prefValue == value || it.name == value } ?: JA_TO_ZH
        }

        fun fromString(value: String?): TranslationLanguage = fromPref(value)

        fun supportedForOcr(useLocalOcr: Boolean): List<TranslationLanguage> {
            return if (useLocalOcr) {
                entries.filter { it.supportsLocalOcr() }
            } else {
                entries
            }
        }

        fun resolveForOcr(language: TranslationLanguage, useLocalOcr: Boolean): TranslationLanguage {
            return if (!useLocalOcr || language.supportsLocalOcr()) {
                language
            } else {
                JA_TO_ZH
            }
        }
    }
}

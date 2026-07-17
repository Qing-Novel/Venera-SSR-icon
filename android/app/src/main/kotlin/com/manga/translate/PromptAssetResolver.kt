package com.manga.translate
import com.github.kiastr.venera_ssr.R

import android.content.Context
import androidx.annotation.StringRes
import java.util.Locale

object PromptAssetResolver {
    private data class PromptLocaleVariant(
        val key: String,
        val suffix: String,
        @param:StringRes val targetLabelRes: Int
    )

    fun resolve(context: Context, assetName: String): String {
        val variant = promptLocaleVariant(context)
        if (variant.suffix.isBlank()) return assetName
        val candidate = assetName.toVariantName(variant.suffix)
        if (candidate == assetName) return assetName
        return if (assetExists(context, candidate)) candidate else assetName
    }

    fun translationTargetKey(context: Context): String = promptLocaleVariant(context).key

    @StringRes
    fun translationTargetLabelRes(context: Context): Int = promptLocaleVariant(context).targetLabelRes

    fun translationTargetDisplayName(context: Context): String {
        return context.getString(translationTargetLabelRes(context))
    }

    private fun assetExists(context: Context, assetName: String): Boolean {
        return try {
            context.assets.open(assetName).close()
            true
        } catch (_: Exception) {
            false
        }
    }

    fun isTraditionalChinese(context: Context): Boolean {
        return promptLocaleVariant(context) == HANT_VARIANT
    }

    private fun promptLocaleVariant(context: Context): PromptLocaleVariant {
        val locale = runCatching {
            context.resources.configuration.locales[0]
        }.getOrNull() ?: Locale.getDefault()
        val normalizedLanguage = locale.language.lowercase(Locale.ROOT)
        return when {
            normalizedLanguage == "en" -> EN_VARIANT
            normalizedLanguage == "ru" -> RU_VARIANT
            !normalizedLanguage.equals(Locale.CHINESE.language, ignoreCase = true) -> DEFAULT_VARIANT
            isTraditionalChineseLocale(locale) -> HANT_VARIANT
            else -> DEFAULT_VARIANT
        }
    }

    private fun isTraditionalChineseLocale(locale: Locale): Boolean {
        val script = locale.script.orEmpty()
        return script.equals("Hant", ignoreCase = true) ||
            locale.country.uppercase(Locale.US) in TRADITIONAL_CHINESE_REGIONS
    }

    private fun String.toVariantName(suffix: String): String {
        val dotIndex = lastIndexOf('.')
        if (dotIndex <= 0) return this
        return substring(0, dotIndex) + suffix + substring(dotIndex)
    }

    private val DEFAULT_VARIANT = PromptLocaleVariant(
        key = "zh_hans",
        suffix = "",
        targetLabelRes = R.string.language_simplified_chinese
    )
    private val HANT_VARIANT = PromptLocaleVariant(
        key = "zh_hant",
        suffix = "_hant",
        targetLabelRes = R.string.language_traditional_chinese
    )
    private val EN_VARIANT = PromptLocaleVariant(
        key = "en",
        suffix = "_en",
        targetLabelRes = R.string.language_english
    )
    private val RU_VARIANT = PromptLocaleVariant(
        key = "ru",
        suffix = "_ru",
        targetLabelRes = R.string.language_russian
    )
    private val TRADITIONAL_CHINESE_REGIONS = setOf("TW", "HK", "MO")
}

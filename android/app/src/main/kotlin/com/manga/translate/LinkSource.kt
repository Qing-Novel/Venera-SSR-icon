package com.manga.translate
import com.github.kiastr.venera_ssr.R

enum class LinkSource(
    val prefValue: String,
    val labelRes: Int
) {
    GITEE("gitee", R.string.tutorial_option_gitee_recommended),
    GITHUB("github", R.string.tutorial_option_github);

    companion object {
        fun fromPref(value: String?): LinkSource {
            return entries.firstOrNull { it.prefValue == value } ?: GITHUB
        }
    }
}

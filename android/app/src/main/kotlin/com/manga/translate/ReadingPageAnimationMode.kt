package com.manga.translate
import com.github.kiastr.venera_ssr.R

import androidx.annotation.StringRes

enum class ReadingPageAnimationMode(
    val prefValue: String,
    @param:StringRes val labelRes: Int
) {
    NONE("none", R.string.reading_page_animation_none),
    HORIZONTAL_SLIDE("horizontal_slide", R.string.reading_page_animation_horizontal_slide);

    companion object {
        fun fromPref(value: String?): ReadingPageAnimationMode {
            return entries.firstOrNull { it.prefValue == value } ?: HORIZONTAL_SLIDE
        }
    }
}

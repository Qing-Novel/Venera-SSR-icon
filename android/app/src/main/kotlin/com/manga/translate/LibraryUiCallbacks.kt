package com.manga.translate

import java.io.File

internal interface LibraryUiCallbacks {
    fun setFolderStatus(left: String, right: String = "")
    fun clearFolderStatus()
    fun setTranslationActionsEnabled(enabled: Boolean)
    fun showToast(resId: Int)
    fun showToastMessage(message: String)
    fun showApiError(code: String, detail: String? = null)
    fun showModelError(
        content: String,
        useSystemOverlay: Boolean,
        onRetry: (() -> Unit)?,
        onSkip: (() -> Unit)? = null
    )
    fun refreshFolders()
    fun refreshImages(folder: File)
    fun isUiAttached(): Boolean
    fun isFragmentActive(): Boolean
    fun isAppInForeground(): Boolean
    fun isLibraryInForeground(): Boolean
    fun canShowSystemOverlay(): Boolean
}

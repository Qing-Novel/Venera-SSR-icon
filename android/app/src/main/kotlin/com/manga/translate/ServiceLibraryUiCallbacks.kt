package com.manga.translate

import java.io.File

internal object ServiceLibraryUiCallbacks : LibraryUiCallbacks {
    override fun setFolderStatus(left: String, right: String) {
        LibraryUiBridge.setFolderStatus(left, right)
    }

    override fun clearFolderStatus() {
        LibraryUiBridge.clearFolderStatus()
    }

    override fun setTranslationActionsEnabled(enabled: Boolean) {
        LibraryUiBridge.setTranslationActionsEnabled(enabled)
    }

    override fun showToast(resId: Int) {
        LibraryUiBridge.showToast(resId)
    }

    override fun showToastMessage(message: String) {
        LibraryUiBridge.showToastMessage(message)
    }

    override fun showApiError(code: String, detail: String?) {
        LibraryUiBridge.showApiError(code, detail)
    }

    override fun showModelError(
        content: String,
        useSystemOverlay: Boolean,
        onRetry: (() -> Unit)?,
        onSkip: (() -> Unit)?
    ) {
        LibraryUiBridge.showModelError(content, useSystemOverlay, onRetry, onSkip)
    }

    override fun refreshFolders() {
        LibraryUiBridge.refreshFolders()
    }

    override fun refreshImages(folder: File) {
        LibraryUiBridge.refreshImages(folder)
    }

    override fun isUiAttached(): Boolean = LibraryUiBridge.hasAttachedUi()

    override fun isFragmentActive(): Boolean = LibraryUiBridge.hasAttachedUi()

    override fun isAppInForeground(): Boolean = LibraryUiBridge.isAppInForeground()

    override fun isLibraryInForeground(): Boolean = LibraryUiBridge.hasAttachedUi()

    override fun canShowSystemOverlay(): Boolean = LibraryUiBridge.canShowSystemOverlay()
}

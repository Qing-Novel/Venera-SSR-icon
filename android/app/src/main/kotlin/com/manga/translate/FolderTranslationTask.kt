package com.manga.translate

import java.io.File

/**
 * 从 jedzqer/manga-translator-android 的 FolderTranslationCoordinator.kt 抽出，
 * 供 TranslationTaskPersistence 等核心文件引用（原文件 UI 耦合重，不整体 vendor）。
 */
internal data class FolderTranslationTask(
    val folder: File,
    val images: List<File>,
    val force: Boolean,
    val fullTranslate: Boolean,
    val glossaryProcessingEnabled: Boolean,
    val useVlDirectTranslate: Boolean,
    val language: TranslationLanguage
)

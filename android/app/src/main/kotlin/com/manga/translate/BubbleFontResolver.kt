package com.manga.translate

import android.content.Context
import android.graphics.Typeface
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException

enum class BubbleFont(
    val prefValue: String,
    val labelRes: Int
) {
    SYSTEM_DEFAULT("system_default", R.string.bubble_font_system_default),
    CUSTOM_FILE("custom_file", R.string.bubble_font_custom_file);

    companion object {
        fun fromPref(value: String?): BubbleFont {
            return entries.firstOrNull { it.prefValue == value } ?: SYSTEM_DEFAULT
        }
    }
}

object BubbleFontResolver {

    private const val CUSTOM_FONT_DIR = "custom_fonts"
    private const val FILE_FONT_KIND = "file"

    private data class CustomFontTarget(
        val file: File? = null,
        val assetPath: String? = null
    )

    private fun managedFontFile(context: Context, tag: String, kind: String): File {
        return File(
            context.getDir(CUSTOM_FONT_DIR, Context.MODE_PRIVATE),
            "bubble_custom_font_${tag}_${kind}.ttf"
        )
    }

    private fun stagedCustomFontFile(context: Context, tag: String): File {
        return managedFontFile(context, tag, FILE_FONT_KIND)
    }

    private fun markerFile(fontFile: File, suffix: String): File {
        return File(fontFile.parentFile, fontFile.name + suffix)
    }

    private fun readMarker(file: File): String? {
        return runCatching { file.readText().trim() }
            .getOrNull()
            ?.takeIf { it.isNotEmpty() }
    }

    fun resolveTypeface(
        context: Context,
        font: BubbleFont,
        customUrl: String? = null,
        customFileName: String? = null,
        tag: String = "normal"
    ): Typeface {
        return when (font) {
            BubbleFont.CUSTOM_FILE -> {
                resolveCustomFileTarget(context, customFileName)?.let { target ->
                    when {
                        target.file != null -> loadTypefaceFromFile(target.file)
                        target.assetPath != null -> {
                            val stagedFile = stagedCustomFontFile(context, tag)
                            val storedAssetPath = readMarker(markerFile(stagedFile, ".asset"))
                            if (
                                storedAssetPath == target.assetPath &&
                                stagedFile.exists() &&
                                stagedFile.length() > 0L
                            ) {
                                loadTypefaceFromFile(stagedFile)
                            } else {
                                Typeface.DEFAULT
                            }
                        }
                        else -> Typeface.DEFAULT
                    }
                } ?: Typeface.DEFAULT
            }
            BubbleFont.SYSTEM_DEFAULT -> Typeface.DEFAULT
        }
    }

    fun resolveTypefaceSignature(
        context: Context,
        font: BubbleFont,
        customFileName: String? = null
    ): String {
        return when (font) {
            BubbleFont.SYSTEM_DEFAULT -> "system_default"
            BubbleFont.CUSTOM_FILE -> {
                val target = resolveCustomFileTarget(context, customFileName)
                when {
                    target?.file != null -> {
                        val file = target.file
                        "file:${file.absolutePath}:${file.lastModified()}:${file.length()}"
                    }
                    target?.assetPath != null -> "asset:${target.assetPath}"
                    else -> "custom_file:missing"
                }
            }
        }
    }

    suspend fun ensureTypeface(
        context: Context,
        font: BubbleFont,
        customUrl: String?,
        customFileName: String? = null,
        tag: String = "normal"
    ): Typeface {
        return when (font) {
            BubbleFont.CUSTOM_FILE -> {
                when (val target = resolveCustomFileTarget(context, customFileName)) {
                    null -> Typeface.DEFAULT
                    else -> when {
                        target.file != null -> loadTypefaceFromFile(target.file)
                        target.assetPath != null -> stageAssetFontAndLoad(context, target.assetPath, tag)
                        else -> Typeface.DEFAULT
                    }
                }
            }
            BubbleFont.SYSTEM_DEFAULT -> Typeface.DEFAULT
        }
    }

    fun listUploadedFonts(context: Context): List<String> {
        val fontDir = context.getDir(CUSTOM_FONT_DIR, Context.MODE_PRIVATE)
        return fontDir.listFiles()
            ?.asSequence()
            ?.filter { isUserUploadedFontFile(it) }
            ?.map { it.name }
            ?.sortedBy { it.lowercase() }
            ?.toList()
            .orEmpty()
    }

    fun deleteUploadedFont(context: Context, fileName: String): Boolean {
        val trimmed = fileName.trim()
        if (trimmed.isBlank()) return false
        val fontFile = resolvePrivateFontFile(context, trimmed) ?: return false
        if (!isUserUploadedFontFile(fontFile)) return false
        return fontFile.delete()
    }

    suspend fun importUploadedFont(
        context: Context,
        uri: Uri
    ): String = withContext(Dispatchers.IO) {
        val displayName = DocumentFile.fromSingleUri(context, uri)?.name.orEmpty()
        val fontDir = context.getDir(CUSTOM_FONT_DIR, Context.MODE_PRIVATE)
        val destFile = allocateUploadedFontFile(fontDir, displayName)
        val tempFile = File(destFile.parentFile, "${destFile.name}.tmp")
        try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                tempFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            } ?: throw IOException("Unable to open selected font file")

            validateFontFile(tempFile)
            if (destFile.exists()) {
                destFile.delete()
            }
            if (!tempFile.renameTo(destFile)) {
                tempFile.copyTo(destFile, overwrite = true)
                tempFile.delete()
            }
            destFile.name
        } catch (e: Exception) {
            tempFile.delete()
            throw e
        }
    }

    private fun isUserUploadedFontFile(file: File): Boolean {
        if (!file.isFile || file.length() <= 0L) return false
        val name = file.name
        if (name.endsWith(".tmp", ignoreCase = true)) return false
        if (name.endsWith(".asset", ignoreCase = true)) return false
        if (name.startsWith("bubble_custom_font_")) return false
        val ext = name.substringAfterLast('.', "").lowercase()
        return ext in setOf("ttf", "otf", "ttc", "otc")
    }

    private fun allocateUploadedFontFile(fontDir: File, displayName: String): File {
        val preferredName = sanitizeUploadedFontFileName(displayName)
        val preferredFile = File(fontDir, preferredName)
        if (!preferredFile.exists()) {
            return preferredFile
        }
        val ext = preferredName.substringAfterLast('.', "ttf")
        val stem = preferredName.substringBeforeLast('.', preferredName)
        var index = 2
        while (true) {
            val candidate = File(fontDir, "${stem}_$index.$ext")
            if (!candidate.exists()) {
                return candidate
            }
            index += 1
        }
    }

    private fun sanitizeUploadedFontFileName(displayName: String): String {
        val rawName = displayName.trim().ifBlank { "uploaded_font.ttf" }
        val baseName = File(rawName).name
        val ext = baseName.substringAfterLast('.', "")
            .lowercase()
            .takeIf { it.isNotBlank() }
            ?: "ttf"
        val stem = baseName.substringBeforeLast('.', baseName)
            .replace(Regex("[^A-Za-z0-9._-]"), "_")
            .trim('_', '.')
            .ifBlank { "uploaded_font" }
        return "$stem.$ext"
    }

    private fun validateFontFile(file: File) {
        try {
            Typeface.createFromFile(file)
        } catch (e: Exception) {
            throw IOException("Invalid font file", e)
        }
    }

    private fun loadTypefaceFromFile(file: File): Typeface {
        return try {
            Typeface.createFromFile(file)
        } catch (e: Exception) {
            AppLogger.log("BubbleFontResolver", "Failed to load typeface from file ${file.path}", e)
            Typeface.DEFAULT
        }
    }

    private fun resolveCustomFileTarget(context: Context, customFileName: String?): CustomFontTarget? {
        val trimmed = customFileName?.trim().orEmpty()
        if (trimmed.isBlank()) return null

        resolvePrivateFontFile(context, trimmed)?.let { file ->
            return CustomFontTarget(file = file)
        }

        resolveAssetFontPath(context, trimmed)?.let { assetPath ->
            return CustomFontTarget(assetPath = assetPath)
        }

        return null
    }

    private fun resolvePrivateFontFile(context: Context, customFileName: String): File? {
        val directFile = File(customFileName)
        if (directFile.isAbsolute && directFile.isFile && directFile.length() > 0L) {
            return directFile
        }

        val fontDir = context.getDir(CUSTOM_FONT_DIR, Context.MODE_PRIVATE)
        val baseName = File(customFileName).name
        val candidates = linkedSetOf(
            File(fontDir, customFileName),
            File(fontDir, baseName),
            File(context.filesDir, customFileName),
            File(context.filesDir, baseName),
            File(context.cacheDir, customFileName),
            File(context.cacheDir, baseName)
        )
        if (customFileName.startsWith("files/")) {
            candidates += File(context.filesDir, customFileName.removePrefix("files/"))
        }
        if (customFileName.startsWith("cache/")) {
            candidates += File(context.cacheDir, customFileName.removePrefix("cache/"))
        }
        return candidates.firstOrNull { it.isFile && it.length() > 0L }
    }

    private fun resolveAssetFontPath(context: Context, customFileName: String): String? {
        val normalized = customFileName.removePrefix("assets/").trimStart('/')
        val candidates = linkedSetOf(normalized, customFileName)
        return candidates.firstOrNull { path ->
            path.isNotBlank() && runCatching {
                context.assets.open(path).close()
                true
            }.getOrDefault(false)
        }
    }

    private suspend fun stageAssetFontAndLoad(
        context: Context,
        assetPath: String,
        tag: String
    ): Typeface = withContext(Dispatchers.IO) {
        val destFile = stagedCustomFontFile(context, tag)
        try {
            destFile.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                destFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            markerFile(destFile, ".asset").writeText(assetPath)
            loadTypefaceFromFile(destFile)
        } catch (e: Exception) {
            AppLogger.log("BubbleFontResolver", "Failed to stage font asset: $assetPath", e)
            val storedAssetPath = readMarker(markerFile(destFile, ".asset"))
            if (storedAssetPath == assetPath && destFile.exists() && destFile.length() > 0L) {
                loadTypefaceFromFile(destFile)
            } else {
                Typeface.DEFAULT
            }
        }
    }
}

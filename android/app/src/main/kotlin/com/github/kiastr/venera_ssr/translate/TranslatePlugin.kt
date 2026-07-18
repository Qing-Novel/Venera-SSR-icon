package com.github.kiastr.venera_ssr.translate

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import com.manga.translate.ApiFormat
import com.manga.translate.ApiSettings
import com.manga.translate.BubbleRenderer
import com.manga.translate.SettingsStore
import com.manga.translate.TranslationLanguage
import com.manga.translate.TranslationPipeline
import kotlinx.coroutines.runBlocking
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

/**
 * MethodChannel 桥接：Dart 侧通过 'translate' 调用端上漫画翻译核心。
 *
 * 移植自 jedzqer/manga-translator-android（MIT），仅搬运检测 + OCR 的端上能力；
 * 文本翻译本身仍由远程 LLM 完成（LlmClient 需要用户在设置中配置 API 地址与 Key）。
 *
 * 调用约定（与 colorize 通道对齐，传内存字节避免落盘）：
 *   method: "translateImage"
 *   args:   imageBytes: ByteArray, language: String? ("ja_to_zh" 等),
 *          forceOcr: Boolean?, renderImage: Boolean? (默认 true)
 *   result:
 *     - renderImage = true（默认）：合成后的翻译图 PNG 字节 (ByteArray)，
 *       直接可被 Flutter 当作图片显示（与 colorize 返回 PNG 的契约一致）。
 *     - renderImage = false：返回气泡结构化数据
 *       Map { name, width, height, bubbles: [ { id, text, rect, source } ] }
 *     未配置 LLM / OCR 或渲染失败时返回错误：
 *       TRANSLATE_FAILED（pipeline 返回 null）/ TRANSLATE_RENDER_FAILED（重绘失败）。
 */
class TranslatePlugin(private val context: Context) : MethodChannel.MethodCallHandler {

    companion object {
        const val CHANNEL = "com.github.kiastr.venera_ssr/translate"

        fun registerWith(context: Context, messenger: BinaryMessenger) {
            MethodChannel(messenger, CHANNEL).setMethodCallHandler(TranslatePlugin(context))
        }
    }

    private val pipeline = TranslationPipeline(context)
    private val executor = Executors.newSingleThreadExecutor()

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "translateImage" -> handleTranslate(call, result)
            // 读取已保存的远程 LLM 配置（apiUrl / apiKey / modelName / apiFormat）
            "getLlmConfig" -> {
                val s = SettingsStore(context).load()
                result.success(
                    mapOf(
                        "apiUrl" to s.apiUrl,
                        "apiKey" to s.apiKey,
                        "modelName" to s.modelName,
                        "apiFormat" to s.apiFormat.prefValue,
                        "configured" to s.isValid()
                    )
                )
            }
            // 保存远程 LLM 配置
            "setLlmConfig" -> {
                val apiUrl = call.argument<String>("apiUrl") ?: ""
                val apiKey = call.argument<String>("apiKey") ?: ""
                val modelName = call.argument<String>("modelName") ?: ""
                val apiFormat = ApiFormat.fromPref(call.argument<String>("apiFormat"))
                try {
                    SettingsStore(context).save(
                        ApiSettings(
                            apiUrl = apiUrl,
                            apiKey = apiKey,
                            modelName = modelName,
                            apiFormat = apiFormat
                        )
                    )
                    result.success(true)
                } catch (e: Throwable) {
                    result.error("SAVE_FAILED", e.message ?: "save failed", null)
                }
            }
            "getLocalTranslationConfig" -> {
                val s = SettingsStore(context)
                val modelDirPath = s.loadLocalTranslationModelDir()
                val modelDir = if (modelDirPath.isNotBlank()) File(modelDirPath) else null
                val modelAvailable = modelDir != null && (
                    File(modelDir, "model.onnx").exists() || File(modelDir, "encoder_model.onnx").exists()
                ) && File(modelDir, "vocab.json").exists()
                result.success(mapOf(
                    "useLocalTranslation" to s.loadUseLocalTranslation(),
                    "modelDir" to modelDirPath,
                    "modelAvailable" to modelAvailable
                ))
            }
            "setLocalTranslationConfig" -> {
                val useLocal = call.argument<Boolean>("useLocalTranslation") ?: false
                val modelDir = call.argument<String>("modelDir") ?: ""
                val s = SettingsStore(context)
                s.saveUseLocalTranslation(useLocal)
                s.saveLocalTranslationModelDir(modelDir)
                result.success(true)
            }
            else -> result.notImplemented()
        }
    }

    private fun handleTranslate(call: MethodCall, result: MethodChannel.Result) {
        val imageBytes = call.argument<ByteArray>("imageBytes")
        if (imageBytes == null) {
            result.error("BAD_ARGS", "imageBytes required", null)
            return
        }
        val languageName = call.argument<String>("language") ?: "JA_TO_ZH"
        val forceOcr = call.argument<Boolean>("forceOcr") ?: false
        val renderImage = call.argument<Boolean>("renderImage") ?: true
        executor.execute {
            var inputFile: File? = null
            try {
                val extension = detectImageExtension(imageBytes)
                inputFile = File(context.cacheDir, "translate_in_${System.currentTimeMillis()}.$extension")
                FileOutputStream(inputFile).use { it.write(imageBytes) }
                val lang = TranslationLanguage.fromPref(languageName)
                val translation = runBlocking {
                    pipeline.translateImage(
                        imageFile = inputFile,
                        glossary = mutableMapOf(),
                        forceOcr = forceOcr,
                        language = lang,
                        onProgress = { }
                    )
                }
                if (translation == null) {
                    result.error("TRANSLATE_FAILED", "pipeline returned null (LLM not configured?)", null)
                    return@execute
                }
                // renderImage = false：仅返回气泡结构化数据（供上层自行渲染/编辑）
                if (!renderImage) {
                    val bubbles = translation.bubbles.map { b ->
                        val r: RectF = b.rect
                        mapOf(
                            "id" to b.id,
                            "text" to b.text,
                            "rect" to mapOf(
                                "left" to r.left,
                                "top" to r.top,
                                "right" to r.right,
                                "bottom" to r.bottom
                            ),
                            "source" to b.source.name
                        )
                    }
                    result.success(
                        mapOf(
                            "name" to translation.imageName,
                            "width" to translation.width,
                            "height" to translation.height,
                            "bubbles" to bubbles
                        )
                    )
                    return@execute
                }
                // renderImage = true（默认）：用端上 BubbleRenderer 把译文重绘进原图，
                // 返回合成后的 PNG 字节（与 colorize 通道返回 PNG 的契约一致，
                // Dart 侧可直接当作图片显示，无需在 Flutter 层再做 CJK 文本排版）。
                val srcBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                if (srcBitmap == null) {
                    result.error("TRANSLATE_RENDER_FAILED", "failed to decode source image", null)
                    return@execute
                }
                try {
                    val renderer = BubbleRenderer(context)
                    val outBitmap = runBlocking { renderer.render(srcBitmap, translation, false) }
                    val pngBytes = ByteArrayOutputStream().use { os ->
                        if (outBitmap.compress(Bitmap.CompressFormat.PNG, 100, os)) {
                            os.toByteArray()
                        } else {
                            null
                        }
                    }
                    srcBitmap.recycle()
                    if (outBitmap !== srcBitmap) {
                        outBitmap.recycle()
                    }
                    val png = pngBytes ?: run {
                        result.error("TRANSLATE_RENDER_FAILED", "failed to encode PNG", null)
                        return@execute
                    }
                    result.success(png)
                } catch (e: Throwable) {
                    result.error("TRANSLATE_RENDER_FAILED", e.message ?: "render failed", null)
                }
            } catch (e: Throwable) {
                result.error("TRANSLATE_FAILED", e.message ?: "unknown error", null)
            } finally {
                inputFile?.delete()
            }
        }
    }

    private fun detectImageExtension(bytes: ByteArray): String {
        if (bytes.size < 12) return "png"
        val b0 = bytes[0].toInt() and 0xFF
        val b1 = bytes[1].toInt() and 0xFF
        val b2 = bytes[2].toInt() and 0xFF
        val b3 = bytes[3].toInt() and 0xFF

        return when {
            // JPEG: FF D8 FF
            b0 == 0xFF && b1 == 0xD8 && b2 == 0xFF -> "jpg"
            // PNG: 89 50 4E 47
            b0 == 0x89 && b1 == 0x50 && b2 == 0x4E && b3 == 0x47 -> "png"
            // WebP: RIFF .... WEBP
            b0 == 0x52 && b1 == 0x49 && b2 == 0x46 && b3 == 0x46 -> {
                val b8 = bytes[8].toInt() and 0xFF
                val b9 = bytes[9].toInt() and 0xFF
                val b10 = bytes[10].toInt() and 0xFF
                val b11 = bytes[11].toInt() and 0xFF
                if (b8 == 0x57 && b9 == 0x45 && b10 == 0x42 && b11 == 0x50) "webp" else "png"
            }
            // AVIF: ....ftypavif
            b4IsFtyp(bytes) && isAvifBrand(bytes) -> "avif"
            // GIF: GIF8
            b0 == 0x47 && b1 == 0x49 && b2 == 0x46 && b3 == 0x38 -> "gif"
            // BMP: BM
            b0 == 0x42 && b1 == 0x4D -> "bmp"
            else -> "png"
        }
    }

    private fun b4IsFtyp(bytes: ByteArray): Boolean {
        if (bytes.size < 8) return false
        return bytes[4].toInt() == 0x66 && bytes[5].toInt() == 0x74 &&
                bytes[6].toInt() == 0x79 && bytes[7].toInt() == 0x70
    }

    private fun isAvifBrand(bytes: ByteArray): Boolean {
        if (bytes.size < 12) return false
        val brand = String(bytes, 8, 4)
        return brand == "avif" || brand == "avis"
    }
}

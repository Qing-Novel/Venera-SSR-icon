package com.github.kiastr.venera_ssr.translate

import android.content.Context
import android.graphics.RectF
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import com.manga.translate.TranslationLanguage
import com.manga.translate.TranslationPipeline
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
 *   args:   imageBytes: ByteArray, language: String? ("JA_TO_ZH" 等), forceOcr: Boolean?
 *   result: Map { name, width, height, bubbles: [ { id, text, rect:{left,top,right,bottom}, source } ] }
 *           未配置 LLM 时返回 TRANSLATE_FAILED（pipeline 返回 null）。
 */
class TranslatePlugin : MethodChannel.MethodCallHandler {

    companion object {
        const val CHANNEL = "com.github.kiastr.venera_ssr/translate"

        fun registerWith(context: Context, messenger: BinaryMessenger) {
            val channel = MethodChannel(messenger, CHANNEL)
            channel.setMethodCallHandler(TranslatePlugin(context))
        }
    }

    private val context: Context
    private val pipeline = TranslationPipeline(context)
    private val executor = Executors.newSingleThreadExecutor()

    constructor(context: Context) {
        this.context = context
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method != "translateImage") {
            result.notImplemented()
            return
        }
        val imageBytes = call.argument<ByteArray>("imageBytes")
        if (imageBytes == null) {
            result.error("BAD_ARGS", "imageBytes required", null)
            return
        }
        val languageName = call.argument<String>("language") ?: "JA_TO_ZH"
        val forceOcr = call.argument<Boolean>("forceOcr") ?: false
        executor.execute {
            var inputFile: File? = null
            try {
                inputFile = File(context.cacheDir, "translate_in_${System.currentTimeMillis()}.png")
                FileOutputStream(inputFile).use { it.write(imageBytes) }
                val lang = try {
                    TranslationLanguage.valueOf(languageName)
                } catch (_: Exception) {
                    TranslationLanguage.JA_TO_ZH
                }
                val translation = pipeline.translateImage(
                    imageFile = inputFile,
                    glossary = mutableMapOf(),
                    forceOcr = forceOcr,
                    language = lang,
                    onProgress = { }
                )
                if (translation == null) {
                    result.error("TRANSLATE_FAILED", "pipeline returned null (LLM not configured?)", null)
                    return@execute
                }
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
            } catch (e: Throwable) {
                result.error("TRANSLATE_FAILED", e.message ?: "unknown error", null)
            } finally {
                inputFile?.delete()
            }
        }
    }
}

package com.github.wgh136.venera.colorize

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import org.opencv.android.OpenCVLoader
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

/**
 * MethodChannel 桥接：Dart 侧通过 'colorize' 方法调用原生计算核心。
 * 单次调用完成：输入图片字节 -> 预处理 -> ONNX 推理 -> 后处理 -> 返回 PNG 字节。
 *
 * 移植自 AiColorize（com.kiastr.aicolorize.ColorizePlugin），改为接收/返回内存字节，
 * 避免中间落盘，更契合 Venera 的缓存式图片处理管线。
 */
class ColorizePlugin : MethodChannel.MethodCallHandler {

    companion object {
        const val CHANNEL = "com.github.wgh136.venera/colorize"

        fun registerWith(messenger: BinaryMessenger) {
            val channel = MethodChannel(messenger, CHANNEL)
            channel.setMethodCallHandler(ColorizePlugin())
        }
    }

    private val engine = ColorizeEngine()
    private val executor = Executors.newSingleThreadExecutor()

    @Volatile
    private var openCvReady = false

    /** OpenCV 4.x Maven AAR 通过 OpenCVLoader.initDebug() 加载内置 .so（debug/release 均可用）。 */
    private fun ensureOpenCv() {
        if (!openCvReady) {
            synchronized(this) {
                if (!openCvReady) {
                    openCvReady = OpenCVLoader.initDebug()
                }
            }
        }
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method != "colorize") {
            result.notImplemented()
            return
        }
        val imageBytes = call.argument<ByteArray>("imageBytes")
        val modelPath = call.argument<String>("modelPath")
        val type = call.argument<String>("type") ?: "deoldify"
        val useNnapi = call.argument<Boolean>("useNnapi") ?: false
        val intensity = call.argument<Double>("intensity")?.toFloat() ?: 1.0f

        if (imageBytes == null || modelPath == null) {
            result.error("BAD_ARGS", "imageBytes / modelPath 不能为空", null)
            return
        }

        // 推理在后台线程执行，避免阻塞 UI（MethodChannel 结果回主线程派发）
        executor.execute {
            try {
                ensureOpenCv()
                val inputBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                    ?: run {
                        result.error("DECODE_FAILED", "无法解码输入图片", null)
                        return@execute
                    }
                val outBitmap = engine.colorize(inputBitmap, modelPath, type, useNnapi, intensity)
                val baos = ByteArrayOutputStream()
                outBitmap.compress(Bitmap.CompressFormat.PNG, 100, baos)
                inputBitmap.recycle()
                outBitmap.recycle()
                result.success(baos.toByteArray())
            } catch (e: Exception) {
                result.error("COLORIZE_FAILED", e.message ?: "unknown error", null)
            }
        }
    }
}

package com.manga.translate

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * 自由文本检测器（气泡外的招牌 / 章节标题 / 拟声词等）。
 *
 * 移植自 jedzqer/manga-translator-android（MIT）。上游使用独立的
 * `yolo11n-text.onnx`（YOLO 输出 cx/cy/w/h + 类别分）做自由文本检测，
 * 与气泡检测器 [BubbleDetector]（yolov8m_seg）是两个不同模型。
 *
 * 推理时把 [suppressionRects]（气泡框，外扩后）以白色涂到 letterbox 画布上，
 * 使模型在气泡内部不输出文字框；最终输出再整体外扩 [FreeTextOutputExpand*]，
 * 经 [PageRegionDetector.detectUnifiedRegions] 合并进 `freeTextRects`。
 *
 * 注意：模型文件 `models/detection/yolo11n-text.onnx` 需随构建打包或通过下载框架获取；
 * 缺失时 [PageRegionDetector.getTextDetector] 会优雅降级（自由文本检测不生效，气泡检测照常）。
 */
class TextDetector(
    private val context: Context,
    private val modelAssetName: String = DEFAULT_MODEL_ASSET,
    private val threadProfile: OnnxThreadProfile = OnnxThreadProfile.LIGHT,
    private val settingsStore: SettingsStore = SettingsStore(context.applicationContext)
) {
    private val env = OnnxRuntimeSupport.environment()
    private val session: OrtSession = createSession()
    private val inputName: String
    private val inputWidth: Int
    private val inputHeight: Int

    init {
        val input = session.inputInfo.entries.first()
        inputName = input.key
        val inputShape = (input.value.info as TensorInfo).shape
        inputHeight = inputShape.getOrNull(2)?.takeIf { it > 0 }?.toInt()
            ?: TranslationCoreDefaults.DefaultDetectionInputSize
        inputWidth = inputShape.getOrNull(3)?.takeIf { it > 0 }?.toInt()
            ?: TranslationCoreDefaults.DefaultDetectionInputSize
    }

    @Synchronized
    fun detect(
        bitmap: Bitmap,
        suppressionRects: List<RectF> = emptyList()
    ): List<RectF> {
        if (bitmap.width <= 1 || bitmap.height <= 1) return emptyList()
        val preprocessed = OnnxImagePreprocessor.letterbox(
            bitmap = bitmap,
            inputWidth = inputWidth,
            inputHeight = inputHeight
        ) { canvas, gain, padX, padY ->
            if (suppressionRects.isEmpty()) return@letterbox
            val paint = Paint().apply {
                color = Color.WHITE
                style = Paint.Style.FILL
            }
            for (rect in suppressionRects) {
                canvas.drawRect(
                    rect.left * gain + padX,
                    rect.top * gain + padY,
                    rect.right * gain + padX,
                    rect.bottom * gain + padY,
                    paint
                )
            }
        }
        val inputBuffer = OnnxImagePreprocessor.bitmapToRgbChwFloat(preprocessed.bitmap)
        preprocessed.bitmap.recycle()
        val tensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(inputBuffer),
            longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
        )
        tensor.use {
            session.run(mapOf(inputName to tensor)).use { outputs ->
                val output = outputs[0] as OnnxTensor
                val shape = (output.info as TensorInfo).shape
                val rawDetections = parseDetections(output.value, shape)
                val filtered = filterByNms(
                    detections = rawDetections,
                    confidenceThreshold = TranslationCoreDefaults.TextDetectorConfThreshold,
                    iouThreshold = TranslationCoreDefaults.TextDetectorNmsIouThreshold,
                    preprocessed = preprocessed,
                    originalWidth = bitmap.width,
                    originalHeight = bitmap.height
                )
                val expanded = filtered.map { detection ->
                    expandRect(
                        rect = detection.toRect(preprocessed, bitmap.width, bitmap.height),
                        ratio = TranslationCoreDefaults.FreeTextOutputExpandRatio,
                        minExpand = TranslationCoreDefaults.FreeTextOutputExpandMin,
                        bitmap = bitmap
                    )
                }
                if (settingsStore.loadModelIoLogging()) {
                    val maxConfidence = rawDetections.maxOfOrNull { it.confidence } ?: 0f
                    AppLogger.log(
                        "TextDetector",
                        "Input ${bitmap.width}x${bitmap.height}, suppressed=${suppressionRects.size}, " +
                            "raw=${rawDetections.size}, max=${"%.3f".format(maxConfidence)}, " +
                            "kept=${expanded.size}"
                    )
                }
                return expanded
            }
        }
    }

    private fun parseDetections(raw: Any, shape: LongArray): List<RawTextDetection> {
        if (shape.size != 3) return emptyList()
        val batch = raw as? Array<*> ?: return emptyList()
        val first = batch.firstOrNull() as? Array<*> ?: return emptyList()
        val rows = first.mapNotNull { it as? FloatArray }
        if (rows.size != first.size) return emptyList()

        val dim1 = shape[1].toInt()
        val dim2 = shape[2].toInt()
        if (dim1 <= 0 || dim2 <= 0) return emptyList()
        val channelsFirst = dim1 <= dim2
        val featureCount = if (channelsFirst) dim1 else dim2
        val detectionCount = if (channelsFirst) dim2 else dim1
        if (featureCount < 5) return emptyList()

        val result = ArrayList<RawTextDetection>(detectionCount)
        if (channelsFirst) {
            if (rows.size < featureCount || rows.take(featureCount).any { it.size < detectionCount }) {
                return emptyList()
            }
            for (index in 0 until detectionCount) {
                val featureRow = FloatArray(featureCount) { channel -> rows[channel][index] }
                parseFeatureRow(featureRow)?.let(result::add)
            }
        } else {
            if (rows.size < detectionCount) return emptyList()
            for (index in 0 until detectionCount) {
                parseFeatureRow(rows[index])?.let(result::add)
            }
        }
        return result
    }

    private fun parseFeatureRow(row: FloatArray): RawTextDetection? {
        if (row.size < 5) return null
        val classScore = bestYoloClassScore(row) ?: return null
        val cx = row[0]
        val cy = row[1]
        val width = row[2]
        val height = row[3]
        if (
            !cx.isFinite() || !cy.isFinite() ||
            !width.isFinite() || !height.isFinite() ||
            width <= 0f || height <= 0f
        ) {
            return null
        }
        return RawTextDetection(
            cx = cx,
            cy = cy,
            width = width,
            height = height,
            confidence = classScore.confidence
        )
    }

    private fun filterByNms(
        detections: List<RawTextDetection>,
        confidenceThreshold: Float,
        iouThreshold: Float,
        preprocessed: LetterboxResult,
        originalWidth: Int,
        originalHeight: Int
    ): List<RawTextDetection> {
        val filtered = detections.filter { it.confidence >= confidenceThreshold }
            .sortedByDescending { it.confidence }
        val selected = ArrayList<RawTextDetection>()
        val suppressed = BooleanArray(filtered.size)
        for (index in filtered.indices) {
            if (suppressed[index]) continue
            val detection = filtered[index]
            val rect = detection.toRect(preprocessed, originalWidth, originalHeight)
            selected.add(detection)
            for (otherIndex in index + 1 until filtered.size) {
                if (suppressed[otherIndex]) continue
                val overlap = iou(
                    rect,
                    filtered[otherIndex].toRect(preprocessed, originalWidth, originalHeight)
                )
                if (overlap > iouThreshold) {
                    suppressed[otherIndex] = true
                }
            }
        }
        return selected
    }

    private fun expandRect(
        rect: RectF,
        ratio: Float,
        minExpand: Float,
        bitmap: Bitmap
    ): RectF {
        val pad = max(minExpand, max(1f, rect.height()) * ratio)
        return RectF(
            (rect.left - pad).coerceIn(0f, bitmap.width.toFloat()),
            (rect.top - pad).coerceIn(0f, bitmap.height.toFloat()),
            (rect.right + pad).coerceIn(0f, bitmap.width.toFloat()),
            (rect.bottom + pad).coerceIn(0f, bitmap.height.toFloat())
        )
    }

    private fun iou(first: RectF, second: RectF): Float {
        val left = max(first.left, second.left)
        val top = max(first.top, second.top)
        val right = min(first.right, second.right)
        val bottom = min(first.bottom, second.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val firstArea = max(0f, first.width()) * max(0f, first.height())
        val secondArea = max(0f, second.width()) * max(0f, second.height())
        val union = firstArea + secondArea - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun createSession(): OrtSession {
        return OnnxRuntimeSupport.getOrCreateSession(
            cacheDir = context.cacheDir,
            assetProvider = context.assets::open,
            assetName = modelAssetName,
            threadProfile = threadProfile
        )
    }

    companion object {
        const val DEFAULT_MODEL_ASSET = "models/detection/yolo11n-text.onnx"
    }
}

private data class RawTextDetection(
    val cx: Float,
    val cy: Float,
    val width: Float,
    val height: Float,
    val confidence: Float
) {
    fun toRect(
        preprocessed: LetterboxResult,
        originalWidth: Int,
        originalHeight: Int
    ): RectF {
        val normalized = width <= 1.5f && height <= 1.5f && cx <= 1.5f && cy <= 1.5f
        val inputCenterX = if (normalized) cx * preprocessed.inputWidth else cx
        val inputCenterY = if (normalized) cy * preprocessed.inputHeight else cy
        val inputBoxWidth = if (normalized) width * preprocessed.inputWidth else width
        val inputBoxHeight = if (normalized) height * preprocessed.inputHeight else height
        val left = (inputCenterX - inputBoxWidth / 2f - preprocessed.padX) / preprocessed.gain
        val top = (inputCenterY - inputBoxHeight / 2f - preprocessed.padY) / preprocessed.gain
        val right = (inputCenterX + inputBoxWidth / 2f - preprocessed.padX) / preprocessed.gain
        val bottom = (inputCenterY + inputBoxHeight / 2f - preprocessed.padY) / preprocessed.gain
        return RectF(
            left.coerceIn(0f, max(0f, originalWidth - 1f)),
            top.coerceIn(0f, max(0f, originalHeight - 1f)),
            right.coerceIn(0f, max(0f, originalWidth - 1f)),
            bottom.coerceIn(0f, max(0f, originalHeight - 1f))
        )
    }
}

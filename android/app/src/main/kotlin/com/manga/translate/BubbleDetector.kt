package com.manga.translate

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.TensorInfo
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

data class BubbleDetection(
    val rect: RectF,
    val confidence: Float,
    val classId: Int,
    val maskContour: FloatArray? = null
)

data class UnifiedRegionDetection(
    val balloons: List<BubbleDetection>,
    val freeTextRects: List<RectF>
)

/**
 * Ultralytics YOLO-seg speech-bubble detector.
 *
 * The bundled model is the upstream `manga109-segmentation-bubble.onnx`
 * (YOLO11n-seg, trained on the Manga109 corpus). It has one class (`balloon`)
 * and emits the normal Ultralytics segmentation head: [1, 37, N]
 * (`xywh`, confidence, 32 mask coefficients) plus [1, 32, H, W] mask
 * prototypes. The detection boxes and confidence are already decoded by the
 * ONNX graph; only NMS and mask reconstruction remain.
 *
 * The previous Venera model (`yolov8m_seg-speech-bubble.onnx`) used
 * `tensor(float16)` I/O; the upstream `manga109` export uses `tensor(float32)`.
 * The loader reads the graph's declared input precision and matches it, so both
 * exports work without changing call sites. Outputs are unpacked from FLOAT16
 * to float32 when needed (the speech-bubble model may emit half precision),
 * falling back to native float32 otherwise.
 *
 * Parsing restored to the upstream (jedzqer) channels-first mask-prototype format.
 * Venera had rewritten this to an end2end multi-class format that the real model
 * does not produce, which yielded zero balloons and silently returned the original
 * image for every translation backend.
 */
class BubbleDetector(
    private val context: Context,
    private val modelAssetName: String = DEFAULT_MODEL_ASSET,
    private val threadProfile: OnnxThreadProfile = OnnxThreadProfile.LIGHT,
    private val settingsStore: SettingsStore = SettingsStore(context.applicationContext)
) {
    private val env = OnnxRuntimeSupport.environment()
    private val session: OrtSession = createSession()
    private val inputName: String
    private val inputShape: LongArray
    private val inputType: OnnxJavaType

    init {
        val input = session.inputInfo.entries.first()
        inputName = input.key
        val tensorInfo = input.value.info as TensorInfo
        inputShape = tensorInfo.shape
        inputType = tensorInfo.type
    }

    @Synchronized
    fun detectRegions(bitmap: Bitmap): UnifiedRegionDetection {
        if (bitmap.width <= 1 || bitmap.height <= 1) {
            return UnifiedRegionDetection(emptyList(), emptyList())
        }

        val inputHeight = inputShape.getOrNull(2)?.takeIf { it > 0 }?.toInt()
            ?: DEFAULT_INPUT_SIZE
        val inputWidth = inputShape.getOrNull(3)?.takeIf { it > 0 }?.toInt()
            ?: DEFAULT_INPUT_SIZE
        val preprocessed = OnnxImagePreprocessor.letterbox(bitmap, inputWidth, inputHeight)
        // Ultralytics ONNX exports expect RGB values normalized to 0..1;
        // normalization is not embedded in this model graph.
        val inputBuffer = OnnxImagePreprocessor.bitmapToRgbChwFloat(preprocessed.bitmap)
        preprocessed.bitmap.recycle()

        val inputTensor = createInputTensor(
            inputBuffer,
            longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
        )
        inputTensor.use { tensor ->
            session.run(mapOf(inputName to tensor)).use { outputs ->
                val output0 = outputs[0] as? OnnxTensor
                    ?: return UnifiedRegionDetection(emptyList(), emptyList())
                val output0Shape = (output0.info as TensorInfo).shape
                val configuredThreshold = settingsStore.loadBubbleConfThresholdPercent() / 100f
                val rawDetections = parseDetections(
                    buffer = FloatBuffer.wrap(getTensorFloatArray(output0)),
                    shape = output0Shape,
                    configuredThreshold = configuredThreshold
                )

                val prototypes = if (outputs.size() >= 2) {
                    val output1 = outputs[1] as? OnnxTensor
                    output1?.let { tensor ->
                        val shape = (tensor.info as TensorInfo).shape
                        parsePrototypes(FloatBuffer.wrap(getTensorFloatArray(tensor)), shape)
                    }
                } else {
                    null
                }

                if (settingsStore.loadModelIoLogging()) {
                    val maxConfidence = rawDetections.maxOfOrNull { it.confidence } ?: 0f
                    AppLogger.log(
                        "BubbleDetector",
                        "Raw detections=${rawDetections.size}, configured=$configuredThreshold, " +
                            "max balloon=${formatConfidence(maxConfidence)}, " +
                            "input=${inputWidth}x$inputHeight"
                    )
                }

                val filtered = filterByNms(
                    detections = rawDetections,
                    iouThreshold = TranslationCoreDefaults.BubbleDetectorNmsIouThreshold,
                    preprocessed = preprocessed,
                    originalWidth = bitmap.width,
                    originalHeight = bitmap.height
                )
                val balloons = ArrayList<BubbleDetection>(filtered.size)
                for (raw in filtered) {
                    val rect = raw.toRect(preprocessed, bitmap.width, bitmap.height)
                    if (rect.width() <= 1f || rect.height() <= 1f) continue
                    val contour = prototypes?.let { proto ->
                        computeMaskContour(
                            detection = raw,
                            prototypes = proto.data,
                            protoHeight = proto.height,
                            protoWidth = proto.width,
                            preprocessed = preprocessed,
                            originalWidth = bitmap.width,
                            originalHeight = bitmap.height,
                            inputWidth = inputWidth,
                            inputHeight = inputHeight
                        )
                    }
                    balloons.add(
                        BubbleDetection(
                            rect = rect,
                            confidence = raw.confidence,
                            classId = CLASS_BALLOON,
                            maskContour = contour
                        )
                    )
                }
                if (settingsStore.loadModelIoLogging()) {
                    AppLogger.log(
                        "BubbleDetector",
                        "Balloons kept=${balloons.size}; maskContours=${balloons.count { it.maskContour != null }}"
                    )
                }
                return UnifiedRegionDetection(
                    balloons = balloons,
                    freeTextRects = emptyList()
                )
            }
        }
    }

    @Synchronized
    fun detect(bitmap: Bitmap): List<BubbleDetection> = detectRegions(bitmap).balloons

    private fun parseDetections(
        buffer: FloatBuffer,
        shape: LongArray,
        configuredThreshold: Float
    ): List<RawDetection> {
        if (shape.size != 3 || shape[0] != 1L) return emptyList()
        val dim1 = shape[1].toInt()
        val dim2 = shape[2].toInt()
        if (dim1 <= 0 || dim2 <= 0) return emptyList()

        // Ultralytics exports channels-first [1, 37, N]. Accept the
        // transposed form as well so a future re-export remains loadable.
        val channelsFirst = dim1 <= dim2
        val featureCount = if (channelsFirst) dim1 else dim2
        val detectionCount = if (channelsFirst) dim2 else dim1
        if (featureCount < 5 + MASK_COEFFICIENT_COUNT) return emptyList()

        val threshold = effectiveDetectionConfidenceThreshold(
            classId = CLASS_BALLOON,
            configuredThreshold = configuredThreshold
        )
        val values = buffer.duplicate()
        values.rewind()
        val result = ArrayList<RawDetection>()
        for (index in 0 until detectionCount) {
            val base = if (channelsFirst) 0 else index * featureCount
            val cx = readFeature(values, base, index, featureCount, detectionCount, channelsFirst, 0)
            val cy = readFeature(values, base, index, featureCount, detectionCount, channelsFirst, 1)
            val width = readFeature(values, base, index, featureCount, detectionCount, channelsFirst, 2)
            val height = readFeature(values, base, index, featureCount, detectionCount, channelsFirst, 3)
            val confidence = readFeature(values, base, index, featureCount, detectionCount, channelsFirst, 4)
            if (
                !cx.isFinite() || !cy.isFinite() || !width.isFinite() || !height.isFinite() ||
                !confidence.isFinite() || width <= 0f || height <= 0f || confidence < threshold
            ) {
                continue
            }
            val coefficients = FloatArray(MASK_COEFFICIENT_COUNT)
            for (coefficient in coefficients.indices) {
                coefficients[coefficient] = readFeature(
                    values,
                    base,
                    index,
                    featureCount,
                    detectionCount,
                    channelsFirst,
                    5 + coefficient
                )
            }
            result.add(
                RawDetection(
                    cx = cx,
                    cy = cy,
                    width = width,
                    height = height,
                    confidence = confidence.coerceIn(0f, 1f),
                    classId = CLASS_BALLOON,
                    maskCoefficients = coefficients
                )
            )
        }
        return result
    }

    private fun readFeature(
        values: FloatBuffer,
        base: Int,
        detectionIndex: Int,
        featureCount: Int,
        detectionCount: Int,
        channelsFirst: Boolean,
        featureIndex: Int
    ): Float {
        val offset = if (channelsFirst) {
            featureIndex * detectionCount + detectionIndex
        } else {
            base + featureIndex
        }
        return values.get(offset)
    }

    private fun parsePrototypes(buffer: FloatBuffer, shape: LongArray): PrototypeData? {
        if (shape.size != 4 || shape[0] != 1L) return null
        val channels = shape[1].toInt()
        val height = shape[2].toInt()
        val width = shape[3].toInt()
        if (channels != MASK_COEFFICIENT_COUNT || height <= 0 || width <= 0) return null
        val expected = channels * height * width
        if (buffer.remaining() < expected && buffer.capacity() < expected) return null
        val values = buffer.duplicate()
        values.rewind()
        val data = FloatArray(expected)
        values.get(data)
        return PrototypeData(data = data, height = height, width = width)
    }

    private fun filterByNms(
        detections: List<RawDetection>,
        iouThreshold: Float,
        preprocessed: LetterboxResult,
        originalWidth: Int,
        originalHeight: Int
    ): List<RawDetection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = ArrayList<RawDetection>()
        val suppressed = BooleanArray(sorted.size)
        for (index in sorted.indices) {
            if (suppressed[index]) continue
            val detection = sorted[index]
            val rect = detection.toRect(preprocessed, originalWidth, originalHeight)
            selected.add(detection)
            for (otherIndex in index + 1 until sorted.size) {
                if (suppressed[otherIndex]) continue
                val overlap = iou(
                    rect,
                    sorted[otherIndex].toRect(preprocessed, originalWidth, originalHeight)
                )
                if (overlap > iouThreshold) suppressed[otherIndex] = true
            }
        }
        return selected
    }

    /**
     * Reconstruct a compact outer polygon from the prototype mask. Sampling
     * scanlines keeps the Android overlay lightweight while preserving the
     * useful non-rectangular speech-bubble shape.
     */
    private fun computeMaskContour(
        detection: RawDetection,
        prototypes: FloatArray,
        protoHeight: Int,
        protoWidth: Int,
        preprocessed: LetterboxResult,
        originalWidth: Int,
        originalHeight: Int,
        inputWidth: Int,
        inputHeight: Int
    ): FloatArray? {
        val inputLeft = (detection.cx - detection.width / 2f).coerceIn(0f, inputWidth.toFloat())
        val inputTop = (detection.cy - detection.height / 2f).coerceIn(0f, inputHeight.toFloat())
        val inputRight = (detection.cx + detection.width / 2f).coerceIn(0f, inputWidth.toFloat())
        val inputBottom = (detection.cy + detection.height / 2f).coerceIn(0f, inputHeight.toFloat())
        val x1 = floor(inputLeft / inputWidth * protoWidth).toInt().coerceIn(0, protoWidth - 1)
        val y1 = floor(inputTop / inputHeight * protoHeight).toInt().coerceIn(0, protoHeight - 1)
        val x2 = floor(inputRight / inputWidth * protoWidth).toInt().coerceIn(x1 + 1, protoWidth)
        val y2 = floor(inputBottom / inputHeight * protoHeight).toInt().coerceIn(y1 + 1, protoHeight)
        if (x2 <= x1 || y2 <= y1) return null

        val sampleCount = (y2 - y1).coerceIn(4, MAX_CONTOUR_SAMPLES)
        val leftEdge = ArrayList<Float>(sampleCount * 2)
        val rightEdge = ArrayList<Float>(sampleCount * 2)
        for (sample in 0 until sampleCount) {
            val fraction = if (sampleCount == 1) 0f else sample / (sampleCount - 1f)
            val y = (y1 + ((y2 - 1 - y1) * fraction).toInt()).coerceIn(y1, y2 - 1)
            var leftX = -1
            var rightX = -1
            for (x in x1 until x2) {
                var score = 0f
                val protoOffset = y * protoWidth + x
                for (coefficient in detection.maskCoefficients.indices) {
                    score += detection.maskCoefficients[coefficient] *
                        prototypes[coefficient * protoHeight * protoWidth + protoOffset]
                }
                if (sigmoid(score) >= MASK_THRESHOLD) {
                    if (leftX < 0) leftX = x
                    rightX = x
                }
            }
            if (leftX >= 0) {
                val leftPoint = mapMaskPointToNormalized(
                    leftX.toFloat(), y.toFloat(), protoWidth, protoHeight,
                    preprocessed, originalWidth, originalHeight
                )
                val rightPoint = mapMaskPointToNormalized(
                    (rightX + 1).toFloat(), y.toFloat(), protoWidth, protoHeight,
                    preprocessed, originalWidth, originalHeight
                )
                leftEdge.add(leftPoint.first)
                leftEdge.add(leftPoint.second)
                rightEdge.add(rightPoint.first)
                rightEdge.add(rightPoint.second)
            }
        }
        if (leftEdge.size < 6) return null

        val polygon = FloatArray(leftEdge.size + rightEdge.size)
        leftEdge.toFloatArray().copyInto(polygon, 0)
        var outputIndex = leftEdge.size
        for (index in rightEdge.size - 2 downTo 0 step 2) {
            polygon[outputIndex] = rightEdge[index]
            polygon[outputIndex + 1] = rightEdge[index + 1]
            outputIndex += 2
        }
        return polygon
    }

    private fun mapMaskPointToNormalized(
        x: Float,
        y: Float,
        maskWidth: Int,
        maskHeight: Int,
        preprocessed: LetterboxResult,
        originalWidth: Int,
        originalHeight: Int
    ): Pair<Float, Float> {
        val inputX = x / maskWidth * preprocessed.inputWidth
        val inputY = y / maskHeight * preprocessed.inputHeight
        val originalX = ((inputX - preprocessed.padX) / preprocessed.gain)
            .coerceIn(0f, max(0f, originalWidth - 1f))
        val originalY = ((inputY - preprocessed.padY) / preprocessed.gain)
            .coerceIn(0f, max(0f, originalHeight - 1f))
        return (
            if (originalWidth > 0) originalX / originalWidth else 0f
        ) to (
            if (originalHeight > 0) originalY / originalHeight else 0f
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
        // Try the canonical upstream asset name first, then fall back to the
        // legacy name Venera previously referenced, so existing setups that
        // dropped jedzqer's model under the old filename still load.
        val candidates = listOf(modelAssetName, LEGACY_MODEL_ASSET, LEGACY_MODEL_ASSET_2).distinct()
        var lastError: Exception? = null
        for (name in candidates) {
            try {
                return OnnxRuntimeSupport.getOrCreateSession(
                    cacheDir = context.cacheDir,
                    assetProvider = context.assets::open,
                    assetName = name,
                    threadProfile = threadProfile,
                    useXnnpack = false
                )
            } catch (e: Exception) {
                lastError = e
                AppLogger.log("BubbleDetector", "Detection model '$name' unavailable", e)
            }
        }
        throw lastError ?: IllegalStateException("Bubble detection model not found: ${candidates.joinToString()}")
    }

    private fun formatConfidence(value: Float): String = "%.3f".format(value)

    /**
     * Build the input tensor using the precision the model graph declares.
     * The previous `yolov8m_seg-speech-bubble.onnx` export used
     * `tensor(float16)` I/O, while the upstream `manga109-segmentation-bubble.onnx`
     * (YOLO11n-seg) export uses `tensor(float32)`. Feeding the wrong precision
     * makes ONNXRuntime reject the input (or return zero detections), so we
     * read `inputType` from the loaded graph and match it.
     */
    private fun createInputTensor(data: FloatArray, shape: LongArray): OnnxTensor {
        return if (inputType == OnnxJavaType.FLOAT16) {
            createFloat16Tensor(data, shape)
        } else {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
        }
    }

    /**
     * Build a FLOAT16 input tensor (used when the model graph declares
     * `tensor(float16)` I/O). A float32 input would be rejected by
     * ONNXRuntime and yield zero detections, so each float is packed into
     * 2 bytes in little-endian order to match the compiled graph.
     */
    private fun createFloat16Tensor(data: FloatArray, shape: LongArray): OnnxTensor {
        val buf = ByteBuffer.allocateDirect(data.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        val sb = buf.asShortBuffer()
        for (v in data) sb.put(floatToFloat16(v))
        buf.rewind()
        return OnnxTensor.createTensor(env, buf, shape, OnnxJavaType.FLOAT16)
    }

    /**
     * Read a tensor's raw values into a flat FloatArray, transparently
     * converting from FLOAT16 when the graph emits half-precision output
     * (the speech-bubble model does). Falls back to the native float32 path
     * for any future full-precision re-export.
     */
    private fun getTensorFloatArray(tensor: OnnxTensor): FloatArray {
        val typeInfo = tensor.info as TensorInfo
        val shape = typeInfo.shape
        val total = shape.fold(1L) { acc, v -> acc * v }.toInt()
        return if (typeInfo.type == OnnxJavaType.FLOAT16) {
            val sb = tensor.byteBuffer.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            FloatArray(total) { float16ToFloat32(sb.get()) }
        } else {
            val fb = tensor.floatBuffer
            val arr = FloatArray(total)
            fb.get(arr)
            arr
        }
    }

    private fun floatToFloat16(v: Float): Short {
        val bits = java.lang.Float.floatToRawIntBits(v)
        val sign = (bits ushr 16) and 0x8000
        val exp = ((bits ushr 23) and 0xFF) - 112
        val mantissa = bits and 0x7FFFFF
        return when {
            exp >= 31 -> (sign or 0x7C00).toShort()
            exp <= 0 -> sign.toShort()
            else -> (sign or (exp shl 10) or (mantissa ushr 13)).toShort()
        }
    }

    private fun float16ToFloat32(half: Short): Float {
        val h = half.toInt() and 0xFFFF
        val sign = (h and 0x8000) shl 16
        val exp = (h and 0x7C00) ushr 10
        val mantissa = h and 0x03FF
        return when {
            exp == 0x1F -> java.lang.Float.intBitsToFloat(sign or 0x7F800000 or (mantissa shl 13))
            exp == 0 -> java.lang.Float.intBitsToFloat(sign or (mantissa shl 13))
            else -> java.lang.Float.intBitsToFloat(sign or ((exp + 112) shl 23) or (mantissa shl 13))
        }
    }

    companion object {
        // Venera now bundles the upstream Manga109 speech-bubble detector
        // (YOLO11n-seg, float32 I/O, native 1600x1600 input, single class).
        // It is ~4.4x smaller than the previous yolov8m_seg model while being
        // trained on the Manga109 corpus, giving better cost-performance for
        // manga translation.
        const val DEFAULT_MODEL_ASSET = "models/detection/manga109-segmentation-bubble.onnx"
        // Previous Venera model, kept as a fallback for existing setups.
        private const val LEGACY_MODEL_ASSET = "models/detection/yolov8m_seg-speech-bubble.onnx"
        private const val LEGACY_MODEL_ASSET_2 = "models/detection/manga109-seg.onnx"
        const val CLASS_BALLOON = 0
        // Kept for the shared confidence helper and existing unit tests. The
        // segmentation model itself only emits CLASS_BALLOON.
        const val CLASS_TEXT = 1
        // The input size is read from the model graph, so this is only a
        // fallback when the graph omits it. The upstream manga109 model
        // declares a native 1600x1600 input; the previous yolov8m model
        // preferred 1024.
        private const val DEFAULT_INPUT_SIZE = 1024
        private const val MASK_COEFFICIENT_COUNT = 32
        private const val MASK_THRESHOLD = 0.5f
        private const val MAX_CONTOUR_SAMPLES = 48
    }
}

private data class PrototypeData(
    val data: FloatArray,
    val height: Int,
    val width: Int
)

private data class RawDetection(
    val cx: Float,
    val cy: Float,
    val width: Float,
    val height: Float,
    val confidence: Float,
    val classId: Int,
    val maskCoefficients: FloatArray
) {
    fun toRect(
        preprocessed: LetterboxResult,
        originalWidth: Int,
        originalHeight: Int
    ): RectF {
        val left = (cx - width / 2f - preprocessed.padX) / preprocessed.gain
        val top = (cy - height / 2f - preprocessed.padY) / preprocessed.gain
        val right = (cx + width / 2f - preprocessed.padX) / preprocessed.gain
        val bottom = (cy + height / 2f - preprocessed.padY) / preprocessed.gain
        val maxX = max(0f, originalWidth - 1f)
        val maxY = max(0f, originalHeight - 1f)
        return RectF(
            left.coerceIn(0f, maxX),
            top.coerceIn(0f, maxY),
            right.coerceIn(0f, maxX),
            bottom.coerceIn(0f, maxY)
        )
    }
}

internal data class YoloClassScore(
    val classId: Int,
    val confidence: Float
)

internal fun bestYoloClassScore(
    featureRow: FloatArray,
    firstClassIndex: Int = 4
): YoloClassScore? {
    if (firstClassIndex !in featureRow.indices) return null
    var bestClassId = -1
    var bestConfidence = Float.NEGATIVE_INFINITY
    for (index in firstClassIndex until featureRow.size) {
        val confidence = featureRow[index]
        if (!confidence.isFinite()) continue
        if (confidence > bestConfidence) {
            bestConfidence = confidence
            bestClassId = index - firstClassIndex
        }
    }
    if (bestClassId < 0) return null
    return YoloClassScore(bestClassId, bestConfidence)
}

internal fun effectiveDetectionConfidenceThreshold(
    classId: Int,
    configuredThreshold: Float
): Float {
    val normalized = configuredThreshold.coerceIn(0f, 1f)
    return if (classId == BubbleDetector.CLASS_BALLOON) {
        max(normalized, TranslationCoreDefaults.MinBalloonConfidence)
    } else {
        normalized
    }
}

private fun sigmoid(value: Float): Float {
    return if (value >= 0f) {
        1f / (1f + exp(-value))
    } else {
        val expValue = exp(value)
        expValue / (1f + expValue)
    }
}

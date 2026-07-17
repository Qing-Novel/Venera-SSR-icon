package com.manga.translate

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.exp
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
 * Manga109 YOLO26-seg (exported ONNX) one-shot detector.
 * Classes: 0=frame (ignored), 1=text (free-floating), 2=balloon (speech bubble).
 * Output is Ultralytics end2end: output0 [1,N,4+1+1+32] xyxy/conf/cls/mask, output1 protos.
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
    private val hasSegOutput: Boolean

    init {
        val input = session.inputInfo.entries.first()
        inputName = input.key
        val tensorInfo = input.value.info as TensorInfo
        inputShape = tensorInfo.shape
        inputType = tensorInfo.type
        hasSegOutput = session.outputInfo.size >= 2
    }

    @Synchronized
    fun detectRegions(bitmap: Bitmap): UnifiedRegionDetection {
        val inputHeight = inputShape.getOrNull(2)?.takeIf { it > 0 }?.toInt()
            ?: TranslationCoreDefaults.DefaultDetectionInputSize
        val inputWidth = inputShape.getOrNull(3)?.takeIf { it > 0 }?.toInt()
            ?: TranslationCoreDefaults.DefaultDetectionInputSize
        val pre = preprocess(bitmap, inputWidth, inputHeight)

        val shape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
        val tensor = if (inputType == OnnxJavaType.FLOAT16) {
            createFloat16Tensor(pre.inputBuffer, shape)
        } else {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(pre.inputBuffer), shape)
        }
        tensor.use {
            val outputs = session.run(mapOf(inputName to tensor))
            outputs.use {
                val output0 = outputs[0] as OnnxTensor
                val output0Shape = (output0.info as TensorInfo).shape

                var protoData: FloatArray? = null
                var protoH = 160
                var protoW = 160
                if (hasSegOutput) {
                    val output1 = outputs[1] as OnnxTensor
                    val output1Shape = (output1.info as TensorInfo).shape
                    protoH = output1Shape.getOrNull(2)?.toInt() ?: 160
                    protoW = output1Shape.getOrNull(3)?.toInt() ?: 160
                    protoData = parseProtos(getTensorValue(output1), output1Shape)
                }

                val rawDetections = parseEnd2EndDetections(getTensorValue(output0), output0Shape)
                val confThreshold = settingsStore.loadBubbleConfThresholdPercent() / 100f
                if (settingsStore.loadModelIoLogging()) {
                    val maxConf = rawDetections.maxOfOrNull { it.confidence } ?: 0f
                    val aboveThreshold = rawDetections.count {
                        it.confidence >= effectiveDetectionConfidenceThreshold(it.classId, confThreshold)
                    }
                    AppLogger.log(
                        "BubbleDetector",
                        "Raw detections: ${rawDetections.size}, above configured=$confThreshold " +
                            "(balloon floor=${TranslationCoreDefaults.MinBalloonConfidence}): " +
                            "$aboveThreshold, max conf: ${"%.3f".format(maxConf)}"
                    )
                }

                val balloons = ArrayList<BubbleDetection>()
                val freeTexts = ArrayList<RectF>()
                val filtered = filterByNms(
                    rawDetections,
                    confThreshold,
                    TranslationCoreDefaults.BubbleDetectorNmsIouThreshold,
                    pre,
                    bitmap.width,
                    bitmap.height
                )
                for (raw in filtered) {
                    if (raw.classId == CLASS_FRAME) continue
                    val rect = raw.toRect(pre, bitmap.width, bitmap.height)
                    if (rect.width() <= 1f || rect.height() <= 1f) continue
                    when (raw.classId) {
                        CLASS_BALLOON -> {
                            val contour = if (protoData != null && raw.maskCoeffs.isNotEmpty()) {
                                computeMaskContour(
                                    raw, protoData, protoH, protoW, pre, bitmap.width, bitmap.height
                                )
                            } else {
                                null
                            }
                            balloons.add(
                                BubbleDetection(rect, raw.confidence, raw.classId, contour)
                            )
                        }
                        CLASS_TEXT -> {
                            freeTexts.add(
                                expandTextRect(
                                    rect,
                                    TranslationCoreDefaults.FreeTextOutputExpandRatio,
                                    TranslationCoreDefaults.FreeTextOutputExpandMin,
                                    bitmap
                                )
                            )
                        }
                    }
                }
                if (settingsStore.loadModelIoLogging()) {
                    AppLogger.log(
                        "BubbleDetector",
                        "Input ${bitmap.width}x${bitmap.height}, balloons=${balloons.size}, freeText=${freeTexts.size}"
                    )
                }
                return UnifiedRegionDetection(balloons = balloons, freeTextRects = freeTexts)
            }
        }
    }

    /** Balloon-only convenience for callers that do not need free-text boxes. */
    @Synchronized
    fun detect(bitmap: Bitmap): List<BubbleDetection> = detectRegions(bitmap).balloons

    private fun preprocess(bitmap: Bitmap, inputWidth: Int, inputHeight: Int): PreprocessResult {
        val letterboxed = OnnxImagePreprocessor.letterbox(bitmap, inputWidth, inputHeight)
        val inputBuffer = OnnxImagePreprocessor.bitmapToRgbChwFloat(letterboxed.bitmap)
        letterboxed.bitmap.recycle()
        return PreprocessResult(
            inputBuffer = inputBuffer,
            gain = letterboxed.gain,
            padX = letterboxed.padX,
            padY = letterboxed.padY,
            inputWidth = letterboxed.inputWidth,
            inputHeight = letterboxed.inputHeight
        )
    }

    private fun parseProtos(raw: Any, shape: LongArray): FloatArray? {
        if (shape.size != 4) return null
        val nm = shape[1].toInt()
        val ph = shape[2].toInt()
        val pw = shape[3].toInt()
        val batch = raw as? Array<*> ?: return null
        val first = batch.firstOrNull() as? Array<*> ?: return null
        val result = FloatArray(nm * ph * pw)
        for (k in 0 until min(first.size, nm)) {
            val rows = first[k] as? Array<*> ?: return null
            for (y in 0 until min(rows.size, ph)) {
                val row = rows[y] as? FloatArray ?: return null
                row.copyInto(result, k * ph * pw + y * pw, 0, min(row.size, pw))
            }
        }
        return result
    }

    /**
     * End2end layout: [1, N, C] with C = 4 (xyxy) + 1 conf + 1 class + 32 mask coeffs.
     * Also accepts legacy channels-first [1, C, N] and cxcywh formats for robustness.
     */
    private fun parseEnd2EndDetections(raw: Any, shape: LongArray): List<RawDetection> {
        if (shape.size != 3) return emptyList()
        val batch = raw as? Array<*> ?: return emptyList()
        val first = batch.firstOrNull() as? Array<*> ?: return emptyList()
        val data = first.mapNotNull { it as? FloatArray }
        if (data.size != first.size) return emptyList()

        val dim1 = shape[1].toInt()
        val dim2 = shape[2].toInt()
        // Prefer anchors-first when last dim looks like feature channels (e.g. 38).
        val anchorsFirst = dim2 in 6..80 || dim1 > dim2
        return if (anchorsFirst) {
            parseAnchorsFirst(data, dim1.coerceAtMost(data.size))
        } else {
            parseChannelsFirst(data, dim2, dim1)
        }
    }

    private fun parseAnchorsFirst(data: List<FloatArray>, n: Int): List<RawDetection> {
        val results = ArrayList<RawDetection>(n)
        for (i in 0 until n) {
            val row = data.getOrNull(i) ?: continue
            if (row.size < 6) continue
            val detection = parseFeatureRow(row) ?: continue
            results.add(detection)
        }
        return results
    }

    private fun parseChannelsFirst(data: List<FloatArray>, n: Int, c: Int): List<RawDetection> {
        if (data.any { it.size < n }) return emptyList()
        if (c < 6) return emptyList()
        val results = ArrayList<RawDetection>(n)
        for (i in 0 until n) {
            val row = FloatArray(c) { ch -> data[ch][i] }
            val detection = parseFeatureRow(row) ?: continue
            results.add(detection)
        }
        return results
    }

    private fun parseFeatureRow(row: FloatArray): RawDetection? {
        val hasMask = row.size >= 4 + 1 + 1 + MASK_COEFFS
        val conf: Float
        val classId: Int
        val maskCoeffs: FloatArray
        val boxIsXyxy: Boolean

        when {
            // End2end: xyxy, conf, cls, mask32
            hasMask && row.size == 4 + 1 + 1 + MASK_COEFFS -> {
                conf = row[4]
                classId = row[5].toInt()
                maskCoeffs = FloatArray(MASK_COEFFS) { k -> row[6 + k] }
                boxIsXyxy = true
            }
            // Multi-class raw YOLO: cxcywh + class scores + mask32
            row.size >= 4 + 3 + MASK_COEFFS -> {
                val numClasses = row.size - 4 - MASK_COEFFS
                if (numClasses <= 0) return null
                var best = 0f
                var bestId = 0
                for (c in 0 until numClasses) {
                    val v = row[4 + c]
                    if (v > best) {
                        best = v
                        bestId = c
                    }
                }
                conf = best
                classId = bestId
                maskCoeffs = FloatArray(MASK_COEFFS) { k -> row[4 + numClasses + k] }
                boxIsXyxy = false
            }
            row.size >= 6 -> {
                conf = row[4]
                classId = row[5].toInt()
                maskCoeffs = FloatArray(0)
                boxIsXyxy = row[2] > row[0] && row[3] > row[1]
            }
            else -> return null
        }

        return if (boxIsXyxy && row[2] > row[0] && row[3] > row[1]) {
            val x1 = row[0]
            val y1 = row[1]
            val x2 = row[2]
            val y2 = row[3]
            RawDetection(
                cx = (x1 + x2) * 0.5f,
                cy = (y1 + y2) * 0.5f,
                w = (x2 - x1).coerceAtLeast(0f),
                h = (y2 - y1).coerceAtLeast(0f),
                confidence = conf,
                classId = classId,
                maskCoeffs = maskCoeffs
            )
        } else {
            RawDetection(
                cx = row[0],
                cy = row[1],
                w = row[2],
                h = row[3],
                confidence = conf,
                classId = classId,
                maskCoeffs = maskCoeffs
            )
        }
    }

    private fun filterByNms(
        detections: List<RawDetection>,
        confThreshold: Float,
        iouThreshold: Float,
        pre: PreprocessResult,
        originalWidth: Int,
        originalHeight: Int
    ): List<RawDetection> {
        val filtered = detections.filter {
            it.confidence >= effectiveDetectionConfidenceThreshold(it.classId, confThreshold)
        }
            .sortedByDescending { it.confidence }
        val selected = ArrayList<RawDetection>()
        val taken = BooleanArray(filtered.size)

        for (i in filtered.indices) {
            if (taken[i]) continue
            val det = filtered[i]
            val rect = det.toRect(pre, originalWidth, originalHeight)
            selected.add(det)
            for (j in i + 1 until filtered.size) {
                if (taken[j]) continue
                // Class-aware NMS: only suppress same class.
                if (filtered[j].classId != det.classId) continue
                val iou = iou(rect, filtered[j].toRect(pre, originalWidth, originalHeight))
                if (iou > iouThreshold) taken[j] = true
            }
        }
        return selected
    }

    private fun computeMaskContour(
        det: RawDetection,
        protos: FloatArray,
        protoH: Int,
        protoW: Int,
        pre: PreprocessResult,
        originalWidth: Int,
        originalHeight: Int
    ): FloatArray? {
        val nm = det.maskCoeffs.size
        val totalProto = protoH * protoW
        val rawMask = FloatArray(totalProto)
        for (i in 0 until totalProto) {
            var sum = 0f
            for (k in 0 until nm) {
                sum += det.maskCoeffs[k] * protos[k * totalProto + i]
            }
            rawMask[i] = sigmoid(sum)
        }

        val normalized = det.w <= 1.5f && det.h <= 1.5f && det.cx <= 1.5f && det.cy <= 1.5f
        val xc = if (normalized) det.cx * pre.inputWidth else det.cx
        val yc = if (normalized) det.cy * pre.inputHeight else det.cy
        val bw = if (normalized) det.w * pre.inputWidth else det.w
        val bh = if (normalized) det.h * pre.inputHeight else det.h

        val x1p = ((xc - bw / 2f) / pre.inputWidth * protoW).toInt().coerceIn(0, protoW)
        val y1p = ((yc - bh / 2f) / pre.inputHeight * protoH).toInt().coerceIn(0, protoH)
        val x2p = ((xc + bw / 2f) / pre.inputWidth * protoW).toInt().coerceIn(0, protoW)
        val y2p = ((yc + bh / 2f) / pre.inputHeight * protoH).toInt().coerceIn(0, protoH)
        if (x2p <= x1p || y2p <= y1p) return null

        return extractContourPolygon(
            rawMask, protoW, protoH, x1p, y1p, x2p, y2p, pre, originalWidth, originalHeight
        )
    }

    private fun extractContourPolygon(
        mask: FloatArray,
        maskW: Int,
        maskH: Int,
        x1: Int,
        y1: Int,
        x2: Int,
        y2: Int,
        pre: PreprocessResult,
        originalWidth: Int,
        originalHeight: Int
    ): FloatArray? {
        val numSamples = (y2 - y1).coerceIn(2, 40)
        val leftEdge = ArrayList<Float>(numSamples * 2)
        val rightEdge = ArrayList<Float>(numSamples * 2)

        for (s in 0 until numSamples) {
            val y = y1 + (y2 - y1) * s / numSamples
            val rowOff = y.coerceIn(0, maskH - 1) * maskW
            var leftX = -1
            var rightX = -1
            for (x in x1 until x2) {
                if (mask[rowOff + x] >= 0.5f) {
                    if (leftX < 0) leftX = x
                    rightX = x
                }
            }
            if (leftX >= 0) {
                val leftPoint = mapMaskPointToOriginal(
                    leftX.toFloat(), y.toFloat(), maskW, maskH, pre, originalWidth, originalHeight
                )
                val rightPoint = mapMaskPointToOriginal(
                    (rightX + 1).toFloat(), y.toFloat(), maskW, maskH, pre, originalWidth, originalHeight
                )
                leftEdge.add(leftPoint.first)
                leftEdge.add(leftPoint.second)
                rightEdge.add(rightPoint.first)
                rightEdge.add(rightPoint.second)
            }
        }
        if (leftEdge.isEmpty()) return null

        val polygon = FloatArray(leftEdge.size + rightEdge.size)
        leftEdge.toFloatArray().copyInto(polygon, 0)
        var outIdx = leftEdge.size
        for (i in rightEdge.size - 2 downTo 0 step 2) {
            polygon[outIdx] = rightEdge[i]
            polygon[outIdx + 1] = rightEdge[i + 1]
            outIdx += 2
        }
        return polygon
    }

    private fun mapMaskPointToOriginal(
        x: Float,
        y: Float,
        maskW: Int,
        maskH: Int,
        pre: PreprocessResult,
        originalWidth: Int,
        originalHeight: Int
    ): Pair<Float, Float> {
        val inputX = x / maskW * pre.inputWidth
        val inputY = y / maskH * pre.inputHeight
        val originalX = ((inputX - pre.padX) / pre.gain).coerceIn(0f, max(0f, originalWidth - 1f))
        val originalY = ((inputY - pre.padY) / pre.gain).coerceIn(0f, max(0f, originalHeight - 1f))
        val normalizedX = if (originalWidth > 0) originalX / originalWidth else 0f
        val normalizedY = if (originalHeight > 0) originalY / originalHeight else 0f
        return normalizedX to normalizedY
    }

    private fun expandTextRect(rect: RectF, ratio: Float, minExpand: Float, bitmap: Bitmap): RectF {
        val h = max(1f, rect.height())
        val pad = max(minExpand, ratio * h)
        return RectF(
            (rect.left - pad).coerceIn(0f, bitmap.width.toFloat()),
            (rect.top - pad).coerceIn(0f, bitmap.height.toFloat()),
            (rect.right + pad).coerceIn(0f, bitmap.width.toFloat()),
            (rect.bottom + pad).coerceIn(0f, bitmap.height.toFloat())
        )
    }

    private fun createFloat16Tensor(data: FloatArray, shape: LongArray): OnnxTensor {
        val buf = ByteBuffer.allocateDirect(data.size * 2).order(ByteOrder.nativeOrder())
        val sb = buf.asShortBuffer()
        for (v in data) sb.put(floatToFloat16(v))
        buf.rewind()
        return OnnxTensor.createTensor(env, buf, shape, OnnxJavaType.FLOAT16)
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

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val inter = max(0f, right - left) * max(0f, bottom - top)
        val areaA = max(0f, a.width()) * max(0f, a.height())
        val areaB = max(0f, b.width()) * max(0f, b.height())
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))

    private fun getTensorValue(tensor: OnnxTensor): Any {
        val typeInfo = tensor.info as TensorInfo
        if (typeInfo.type != OnnxJavaType.FLOAT16) return tensor.value
        val shape = typeInfo.shape
        val total = shape.fold(1L) { acc, v -> acc * v }.toInt()
        val sb = tensor.byteBuffer.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val floatData = FloatArray(total) { float16ToFloat32(sb.get()) }
        return reshapeToNestedArray(floatData, shape, 0)
    }

    private fun reshapeToNestedArray(data: FloatArray, shape: LongArray, dim: Int): Any {
        val size = shape[dim].toInt()
        if (dim == shape.size - 1) return data.copyOf(size)
        val subSize = shape.drop(dim + 1).fold(1L) { acc, v -> acc * v }.toInt()
        return Array(size) { i ->
            val start = i * subSize
            reshapeToNestedArray(data.copyOfRange(start, minOf(start + subSize, data.size)), shape, dim + 1)
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

    private fun createSession(): OrtSession {
        // YOLO-seg + XNNPACK has crashed natively on some devices during session create.
        // Match PP-OCR / line-detector: plain CPU EP only.
        return OnnxRuntimeSupport.getOrCreateSession(
            cacheDir = context.cacheDir,
            assetProvider = context.assets::open,
            assetName = modelAssetName,
            threadProfile = threadProfile,
            useXnnpack = false
        )
    }

    companion object {
        const val DEFAULT_MODEL_ASSET = "models/detection/manga109-seg.onnx"
        const val CLASS_FRAME = 0
        const val CLASS_TEXT = 1
        const val CLASS_BALLOON = 2
        private const val MASK_COEFFS = 32
    }
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

private data class RawDetection(
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val confidence: Float,
    val classId: Int,
    val maskCoeffs: FloatArray = FloatArray(0)
) {
    fun toRect(
        pre: PreprocessResult,
        originalWidth: Int,
        originalHeight: Int
    ): RectF {
        val normalized = w <= 1.5f && h <= 1.5f && cx <= 1.5f && cy <= 1.5f
        val xCenter = if (normalized) cx * pre.inputWidth else cx
        val yCenter = if (normalized) cy * pre.inputHeight else cy
        val width = if (normalized) w * pre.inputWidth else w
        val height = if (normalized) h * pre.inputHeight else h
        val left = ((xCenter - width / 2f) - pre.padX) / pre.gain
        val top = ((yCenter - height / 2f) - pre.padY) / pre.gain
        val right = ((xCenter + width / 2f) - pre.padX) / pre.gain
        val bottom = ((yCenter + height / 2f) - pre.padY) / pre.gain
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

private data class PreprocessResult(
    val inputBuffer: FloatArray,
    val gain: Float,
    val padX: Float,
    val padY: Float,
    val inputWidth: Int,
    val inputHeight: Int
)

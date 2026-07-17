package com.manga.translate

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.core.graphics.scale
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

private typealias DecoderTokenCache = Array<Array<Array<Array<FloatArray>>>>

class MangaOcrMobile(
    private val context: Context,
    private val settingsStore: SettingsStore = SettingsStore(context.applicationContext),
    private val numThreads: Int = 2
) : OcrEngine {
    @Volatile private var closed = false
    private val config = loadConfig()
    private val tokenizer = loadTokenizer()
    private val encoderInterpreter = createInterpreter(ENCODER_ASSET_NAME)
    private val decoderInterpreter = createInterpreter(DECODER_ASSET_NAME)

    @Synchronized
    override fun recognize(bitmap: Bitmap): String {
        val encoderInput = preprocess(bitmap)
        val encoderOutput = Array(1) { Array(config.encoderSeqLen) { FloatArray(config.encoderHiddenSize) } }
        encoderInterpreter.run(encoderInput, encoderOutput)

        val initInputs = mutableMapOf<String, Any>(
            "args_0" to encoderOutput,
            "args_1" to arrayOf(longArrayOf(config.decoderStartTokenId.toLong()))
        )
        val initOutputs = mutableMapOf<String, Any>(
            "output_0" to Array(1) { FloatArray(config.vocabSize) },
            "output_1" to Array(config.numDecoderLayers) {
                Array(1) { Array(config.numHeads) { Array(1) { FloatArray(config.headDim) } } }
            },
            "output_2" to Array(config.numDecoderLayers) {
                Array(1) { Array(config.numHeads) { Array(1) { FloatArray(config.headDim) } } }
            },
            "output_3" to Array(config.numDecoderLayers) {
                Array(1) {
                    Array(config.numHeads) {
                        Array(config.encoderSeqLen) { FloatArray(config.headDim) }
                    }
                }
            },
            "output_4" to Array(config.numDecoderLayers) {
                Array(1) {
                    Array(config.numHeads) {
                        Array(config.encoderSeqLen) { FloatArray(config.headDim) }
                    }
                }
            }
        )
        decoderInterpreter.runSignature(initInputs, initOutputs, DECODER_INIT_SIGNATURE)

        val tokenIds = ArrayList<Int>(config.maxLength)
        var logits = outputLogits("output_0", initOutputs)
        val selfKeys = initializeSelfCache(initOutputs.getValue("output_1"))
        val selfValues = initializeSelfCache(initOutputs.getValue("output_2"))
        val crossKeys = initOutputs.getValue("output_3")
        val crossValues = initOutputs.getValue("output_4")

        var nextToken = argmax(logits[0])
        var position = 1L
        while (tokenIds.size < config.maxLength && nextToken != config.eosTokenId) {
            tokenIds.add(nextToken)
            if (tokenIds.size >= config.maxLength) {
                break
            }
            val stepInputs = mutableMapOf<String, Any>(
                "args_0" to encoderOutput,
                "args_1" to arrayOf(longArrayOf(nextToken.toLong())),
                "args_2" to arrayOf(longArrayOf(position)),
                "args_3" to selfKeys,
                "args_4" to selfValues,
                "args_5" to crossKeys,
                "args_6" to crossValues
            )
            val stepOutputs = mutableMapOf<String, Any>(
                "output_0" to Array(1) { FloatArray(config.vocabSize) },
                "output_1" to Array(config.numDecoderLayers) {
                    Array(1) { Array(config.numHeads) { Array(1) { FloatArray(config.headDim) } } }
                },
                "output_2" to Array(config.numDecoderLayers) {
                    Array(1) { Array(config.numHeads) { Array(1) { FloatArray(config.headDim) } } }
                }
            )
            decoderInterpreter.runSignature(stepInputs, stepOutputs, DECODER_STEP_SIGNATURE)
            logits = outputLogits("output_0", stepOutputs)
            appendSelfCache(
                currentCache = selfKeys,
                stepCache = stepOutputs.getValue("output_1"),
                position = tokenIds.size
            )
            appendSelfCache(
                currentCache = selfValues,
                stepCache = stepOutputs.getValue("output_2"),
                position = tokenIds.size
            )
            nextToken = argmax(logits[0])
            position++
        }

        val text = tokenizer.decode(tokenIds)
        if (settingsStore.loadModelIoLogging()) {
            AppLogger.log("MangaOcrMobile", "Input ${bitmap.width}x${bitmap.height}, output: $text")
        }
        return text
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val canvasBitmap = Bitmap.createBitmap(config.imageSize, config.imageSize, Bitmap.Config.ARGB_8888)
        var scaled: Bitmap? = null
        try {
            val canvas = Canvas(canvasBitmap)
            canvas.drawColor(Color.WHITE)

            val scale = min(
                config.imageSize.toFloat() / bitmap.width.toFloat(),
                config.imageSize.toFloat() / bitmap.height.toFloat()
            )
            val scaledWidth = max(1, (bitmap.width * scale).toInt())
            val scaledHeight = max(1, (bitmap.height * scale).toInt())
            scaled = bitmap.scale(scaledWidth, scaledHeight)
            val left = (config.imageSize - scaledWidth) / 2f
            val top = (config.imageSize - scaledHeight) / 2f
            canvas.drawBitmap(scaled, left, top, Paint(Paint.FILTER_BITMAP_FLAG))

            val buffer = ByteBuffer.allocateDirect(4 * 3 * config.imageSize * config.imageSize)
            buffer.order(ByteOrder.nativeOrder())
            val pixels = IntArray(config.imageSize * config.imageSize)
            canvasBitmap.getPixels(
                pixels,
                0,
                config.imageSize,
                0,
                0,
                config.imageSize,
                config.imageSize
            )
            for (channel in 0 until 3) {
                for (pixel in pixels) {
                    val component = when (channel) {
                        0 -> (pixel shr 16) and 0xFF
                        1 -> (pixel shr 8) and 0xFF
                        else -> pixel and 0xFF
                    }
                    buffer.putFloat(component / 255f)
                }
            }
            buffer.rewind()
            return buffer
        } finally {
            if (scaled != null && scaled !== bitmap && !scaled.isRecycled) {
                scaled.recycle()
            }
            if (!canvasBitmap.isRecycled) {
                canvasBitmap.recycle()
            }
        }
    }

    private fun initializeSelfCache(initCache: Any): DecoderTokenCache {
        val firstStep = requireDecoderTokenCache(initCache, "decoder init cache")
        val cache = allocateSelfCache()
        for (layer in 0 until config.numDecoderLayers) {
            for (head in 0 until config.numHeads) {
                System.arraycopy(
                    firstStep[layer][0][head][0],
                    0,
                    cache[layer][0][head][0],
                    0,
                    config.headDim
                )
            }
        }
        return cache
    }

    private fun appendSelfCache(
        currentCache: DecoderTokenCache,
        stepCache: Any,
        position: Int
    ) {
        val step = requireDecoderTokenCache(stepCache, "decoder step cache")
        for (layer in 0 until config.numDecoderLayers) {
            for (head in 0 until config.numHeads) {
                System.arraycopy(
                    step[layer][0][head][0], 0,
                    currentCache[layer][0][head][position - 1], 0,
                    config.headDim
                )
            }
        }
    }

    private fun allocateSelfCache(): DecoderTokenCache {
        return Array(config.numDecoderLayers) {
            Array(1) {
                Array(config.numHeads) {
                    Array(config.maxLength) { FloatArray(config.headDim) }
                }
            }
        }
    }

    private fun argmax(values: FloatArray): Int {
        var bestIndex = 0
        var bestValue = Float.NEGATIVE_INFINITY
        for (i in values.indices) {
            if (values[i] > bestValue) {
                bestValue = values[i]
                bestIndex = i
            }
        }
        return bestIndex
    }

    private fun outputLogits(
        key: String,
        outputs: Map<String, Any>
    ): Array<FloatArray> {
        val value = outputs[key]
        require(value is Array<*>) { "Expected logits array for $key" }
        @Suppress("UNCHECKED_CAST")
        return value as Array<FloatArray>
    }

    private fun requireDecoderTokenCache(
        value: Any,
        label: String
    ): DecoderTokenCache {
        require(value is Array<*>) { "Expected decoder token cache for $label" }
        @Suppress("UNCHECKED_CAST")
        return value as DecoderTokenCache
    }

    fun close() {
        if (closed) return
        closed = true
        encoderInterpreter.close()
        decoderInterpreter.close()
    }

    private fun createInterpreter(assetName: String): Interpreter {
        val model = OnnxRuntimeSupport.copyAssetToCacheIfMissing(
            cacheDir = context.cacheDir,
            assetProvider = context.assets::open,
            assetName = assetName
        )
        return Interpreter(
            model,
            Interpreter.Options().apply {
                setNumThreads(numThreads)
                setUseXNNPACK(true)
            }
        )
    }

    private fun loadConfig(): MobileConfig {
        val json = JSONObject(readAsset(CONFIG_ASSET_NAME))
        return MobileConfig(
            imageSize = json.getInt("image_size"),
            maxLength = json.getInt("max_length"),
            decoderStartTokenId = json.getInt("decoder_start_token_id"),
            eosTokenId = json.getInt("eos_token_id"),
            padTokenId = json.getInt("pad_token_id"),
            encoderSeqLen = json.getInt("encoder_seq_len"),
            encoderHiddenSize = json.getInt("encoder_hidden_size"),
            numHeads = json.getInt("num_heads"),
            headDim = json.getInt("head_dim"),
            numDecoderLayers = json.getInt("num_decoder_layers"),
            vocabSize = tokenizerVocabSize()
        )
    }

    private fun tokenizerVocabSize(): Int {
        val tokenizerJson = JSONObject(readAsset(TOKENIZER_ASSET_NAME))
        val vocab = tokenizerJson.getJSONObject("model").getJSONObject("vocab")
        return vocab.length()
    }

    private fun loadTokenizer(): MobileTokenizer {
        val tokenizerJson = JSONObject(readAsset(TOKENIZER_ASSET_NAME))
        val modelJson = tokenizerJson.getJSONObject("model")
        val vocabJson = modelJson.getJSONObject("vocab")
        val keys = vocabJson.keys()
        val entries = ArrayList<Pair<String, Int>>(vocabJson.length())
        var maxId = 0
        while (keys.hasNext()) {
            val token = keys.next()
            val id = vocabJson.getInt(token)
            maxId = max(maxId, id)
            entries.add(token to id)
        }
        val idToToken = Array(maxId + 1) { "" }
        for ((token, id) in entries) {
            idToToken[id] = token
        }

        val specialTokensJson = JSONObject(readAsset(SPECIAL_TOKENS_ASSET_NAME))
        val specialTokens = HashSet<String>()
        val specialKeys = specialTokensJson.keys()
        while (specialKeys.hasNext()) {
            val key = specialKeys.next()
            specialTokens.add(
                specialTokensJson.getJSONObject(key).getString("content")
            )
        }
        return MobileTokenizer(idToToken, specialTokens)
    }

    private fun readAsset(name: String): String {
        return context.assets.open(name).bufferedReader().use { it.readText() }
    }

    companion object {
        private const val CONFIG_ASSET_NAME = "models/ocr/manga_ocr_mobile/config.json"
        private const val ENCODER_ASSET_NAME = "models/ocr/manga_ocr_mobile/encoder.tflite"
        private const val DECODER_ASSET_NAME = "models/ocr/manga_ocr_mobile/decoder.tflite"
        private const val TOKENIZER_ASSET_NAME = "models/ocr/manga_ocr_mobile/tokenizer/tokenizer.json"
        private const val SPECIAL_TOKENS_ASSET_NAME = "models/ocr/manga_ocr_mobile/tokenizer/special_tokens_map.json"
        private const val DECODER_INIT_SIGNATURE = "init"
        private const val DECODER_STEP_SIGNATURE = "step"
    }
}

private data class MobileConfig(
    val imageSize: Int,
    val maxLength: Int,
    val decoderStartTokenId: Int,
    val eosTokenId: Int,
    val padTokenId: Int,
    val encoderSeqLen: Int,
    val encoderHiddenSize: Int,
    val numHeads: Int,
    val headDim: Int,
    val numDecoderLayers: Int,
    val vocabSize: Int
)

private class MobileTokenizer(
    private val idToToken: Array<String>,
    private val specialTokens: Set<String>
) {
    fun decode(ids: List<Int>): String {
        val builder = StringBuilder()
        for (id in ids) {
            if (id !in idToToken.indices) continue
            val token = idToToken[id]
            if (token.isEmpty() || token in specialTokens) continue
            builder.append(token)
        }
        return builder.toString()
    }
}

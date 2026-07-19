package com.manga.translate

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.io.File
import java.nio.LongBuffer

class MarianMtEngine(
    private val context: Context,
    private val modelDir: File
) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var tokenizer: SentencePieceTokenizer? = null
    private var session: OrtSession? = null

    private fun initIfNeeded() {
        if (session != null && tokenizer != null) return
        val vocabFile = File(modelDir, "vocab.json")
        val spmFile = File(modelDir, "source.spm")
        if (vocabFile.exists() && spmFile.exists()) {
            tokenizer = SentencePieceTokenizer(spmFile, vocabFile)
        }
        val modelFile = File(modelDir, "model.onnx").takeIf { it.exists() }
            ?: File(modelDir, "encoder_model.onnx").takeIf { it.exists() }
        if (modelFile != null) {
            val options = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setInterOpNumThreads(1)
            }
            session = env.createSession(modelFile.absolutePath, options)
        }
    }

    fun isAvailable(): Boolean {
        return (File(modelDir, "model.onnx").exists() || File(modelDir, "encoder_model.onnx").exists())
            && File(modelDir, "vocab.json").exists()
            && File(modelDir, "source.spm").exists()
    }

    fun translate(text: String, language: TranslationLanguage): String {
        try {
            initIfNeeded()
            val currentTokenizer = tokenizer ?: return text
            val currentSession = session ?: return text
            val inputIds = currentTokenizer.encode(text)
            if (inputIds.isEmpty()) return text
            val seqLen = inputIds.size.toLong()
            val inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), longArrayOf(1, seqLen))
            val attentionMask = LongArray(inputIds.size) { 1L }
            val attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), longArrayOf(1, seqLen))
            return inputIdsTensor.use { ids ->
                attentionMaskTensor.use { mask ->
                    val inputs = mutableMapOf<String, OnnxTensor>()
                    inputs["input_ids"] = ids
                    inputs["attention_mask"] = mask
                    val outputs = currentSession.run(inputs)
                    outputs.use {
                        val outputTensor = outputs.get(0) as OnnxTensor
                        val outputIds = outputTensor.longBuffer.array().map { it.toInt() }.toIntArray()
                        currentTokenizer.decode(outputIds)
                    }
                }
            }
        } catch (e: Exception) {
            AppLogger.log("MarianMtEngine", "Translation failed", e)
            return text
        }
    }

    fun close() {
        runCatching { session?.close() }
        session = null
        tokenizer = null
    }
}

package com.manga.translate

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.io.File
import java.nio.LongBuffer

fun TranslationLanguage.toMarianSourceTag(): String = when (this) {
    TranslationLanguage.JA_TO_ZH -> ">>zh<<"
    TranslationLanguage.EN_TO_ZH -> ">>zh<<"
    TranslationLanguage.KO_TO_ZH -> ">>zh<<"
    TranslationLanguage.FR_TO_ZH -> ">>zh<<"
    TranslationLanguage.ES_TO_ZH -> ">>zh<<"
    TranslationLanguage.PT_TO_ZH -> ">>zh<<"
    TranslationLanguage.DE_TO_ZH -> ">>zh<<"
    TranslationLanguage.IT_TO_ZH -> ">>zh<<"
    TranslationLanguage.RU_TO_ZH -> ">>zh<<"
    else -> ">>zh<<"
}

class MarianMtEngine(
    private val context: Context,
    private val modelDir: File
) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var tokenizer: MarianTokenizer? = null
    private var session: OrtSession? = null

    private fun initIfNeeded() {
        if (session != null && tokenizer != null) return
        val vocabFile = File(modelDir, "vocab.json")
        if (vocabFile.exists()) {
            tokenizer = MarianTokenizer(vocabFile)
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
    }

    fun translate(text: String, language: TranslationLanguage): String {
        try {
            initIfNeeded()
            val currentTokenizer = tokenizer ?: return text
            val currentSession = session ?: return text
            val taggedText = "${language.toMarianSourceTag()} $text"
            val inputIds = currentTokenizer.encode(taggedText)
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
                        val outputIds = outputTensor.longBuffer.array().map { it.toInt() }
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

class MarianTokenizer(private val vocabFile: File) {
    private val tokenToId: Map<String, Int>
    private val idToToken: Array<String>
    val eosTokenId: Int
    val padTokenId: Int
    val unkTokenId: Int

    init {
        val json = JSONObject(vocabFile.readText())
        val map = mutableMapOf<String, Int>()
        var maxId = 0
        val keys = json.keys()
        while (keys.hasNext()) {
            val token = keys.next()
            val id = json.getInt(token)
            map[token] = id
            if (id > maxId) maxId = id
        }
        tokenToId = map
        idToToken = Array(maxId + 1) { "" }
        map.forEach { (token, id) -> idToToken[id] = token }
        eosTokenId = map["</s>"] ?: 0
        padTokenId = map["<pad>"] ?: 1
        unkTokenId = map["<unk>"] ?: 3
    }

    fun encode(text: String): LongArray {
        val spPrefix = "\u2581"
        val tokens = mutableListOf<Long>()
        val words = text.split(Regex("\\s+"))
        for ((idx, word) in words.withIndex()) {
            if (word.isEmpty()) continue
            val prefixed = if (idx == 0) word else "$spPrefix$word"
            val id = tokenToId[prefixed] ?: tokenToId[word] ?: unkTokenId
            tokens.add(id.toLong())
        }
        tokens.add(eosTokenId.toLong())
        return tokens.toLongArray()
    }

    fun decode(ids: List<Int>): String {
        val spPrefix = "\u2581"
        val sb = StringBuilder()
        for (id in ids) {
            if (id == eosTokenId || id == padTokenId) break
            if (id !in idToToken.indices) continue
            val token = idToToken[id]
            if (token.isEmpty() || token.startsWith("<")) continue
            if (token.startsWith(spPrefix)) {
                if (sb.isNotEmpty()) sb.append(" ")
                sb.append(token.substring(spPrefix.length))
            } else {
                sb.append(token)
            }
        }
        return sb.toString().trim()
    }
}

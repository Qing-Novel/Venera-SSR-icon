package com.manga.translate

import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * SentencePiece Unigram 分词器（Kotlin 移植），用于端上 MarianMT 推理。
 *
 * 为什么需要它：之前手写的 [MarianTokenizer]（空格切分 + 整词查表）对 "Yes!" / "manga" /
 * "reading" 这类「标点粘连词」或「子词」完全无法处理——整词不在词表里就退化成 <unk>，
 * 模型输出空译文，协调器把这些气泡判空丢弃，最终原图英文残留但界面报「翻译成功」。
 * 这里改为解析随模型打包的 `source.spm`（protobuf），用 Viterbi（带 SentencePiece 同款的
 * 单字惩罚）做子词切分，使产出的 token id 与参考 MarianTokenizer 完全一致
 * （导出的 ONNX 模型正是按该分词训练的）。
 */
class SentencePieceTokenizer(
    spmFile: File,
    vocabFile: File
) {
    data class Piece(val text: String, val score: Float, val type: Int)

    private val pieces: List<Piece>
    private val pieceToId: Map<String, Int>
    private val piecesByFirstChar: Map<Char, List<Piece>>
    private val bytePieces: Map<Int, Piece>
    private val unkId: Int
    private val eosId: Int
    private val padId: Int
    private val spPrefix = "\u2581"

    init {
        val vocab = JSONObject(vocabFile.readText())
        val idMap = mutableMapOf<String, Int>()
        val keys = vocab.keys()
        while (keys.hasNext()) {
            val token = keys.next()
            idMap[token] = vocab.getInt(token)
        }
        pieceToId = idMap
        unkId = idMap["<unk>"] ?: 3
        eosId = idMap["</s>"] ?: 0
        padId = idMap["<pad>"] ?: 65000

        val parsed = parseSpam(spmFile.readBytes())
        pieces = parsed
        piecesByFirstChar = parsed.groupBy { if (it.text.isEmpty()) ' ' else it.text.first() }
        bytePieces = parsed.filter { it.type == 5 }.mapNotNull { piece ->
            val hex = piece.text.removePrefix("<0x").removeSuffix(">")
            val cp = hex.toIntOrNull(16) ?: return@mapNotNull null
            cp to piece
        }.toMap()
    }

    fun encode(text: String): LongArray {
        val norm = spPrefix + text.replace(Regex("\\s+"), spPrefix)
        val n = norm.length
        val inf = Double.MAX_VALUE
        val best = DoubleArray(n + 1) { inf }
        best[0] = 0.0
        val back = IntArray(n + 1) { -1 }
        val backId = IntArray(n + 1) { -1 }
        for (pos in 0 until n) {
            if (best[pos] >= inf) continue
            val candidates = piecesByFirstChar[norm[pos]] ?: emptyList()
            for (piece in candidates) {
                val len = piece.text.length
                if (len == 0 || pos + len > n) continue
                if (norm.regionMatches(pos, piece.text, 0, len)) {
                    var cost = best[pos] + (-piece.score)
                    if (len == 1 && piece.type != 5) cost += 10.0
                    if (cost < best[pos + len]) {
                        best[pos + len] = cost
                        back[pos + len] = pos
                        backId[pos + len] = pieceToId[piece.text] ?: unkId
                    }
                }
            }
            // 单字兜底：本位置若仍不可达，用 byte piece 或 unk 吞掉一个码点
            if (best[pos + 1] >= inf) {
                val cp = norm[pos].code
                val bp = bytePieces[cp]
                val cost = if (bp != null) best[pos] + (-bp.score) + 10.0 else best[pos] + 10.0
                if (cost < best[pos + 1]) {
                    best[pos + 1] = cost
                    back[pos + 1] = pos
                    backId[pos + 1] = if (bp != null) (pieceToId[bp.text] ?: unkId) else unkId
                }
            }
        }
        if (best[n] >= inf) return longArrayOf(unkId.toLong(), eosId.toLong())
        val ids = mutableListOf<Int>()
        var pos = n
        while (pos > 0) {
            val id = backId[pos]
            if (id < 0) break
            ids.add(id)
            pos = back[pos]
        }
        ids.reverse()
        ids.add(eosId)
        return ids.map { it.toLong() }.toLongArray()
    }

    fun decode(ids: IntArray): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (id == eosId || id == padId) break
            val token = idToToken(id) ?: continue
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

    private val idToTokenCache = mutableMapOf<Int, String>()

    private fun idToToken(id: Int): String? {
        idToTokenCache[id]?.let { return it }
        val found = pieceToId.entries.firstOrNull { it.value == id }?.key
        if (found != null) idToTokenCache[id] = found
        return found
    }

    private fun parseSpam(bytes: ByteArray): List<Piece> {
        val result = mutableListOf<Piece>()
        var i = 0
        while (i < bytes.size) {
            val (tag, ni) = readVarint(bytes, i)
            i = ni
            val fn = tag shr 3
            val wt = tag and 7
            if (fn == 1 && wt == 2) {
                val (len, ni2) = readVarint(bytes, i)
                i = ni2
                val msg = bytes.copyOfRange(i, i + len)
                i += len
                var p = ""
                var sc = 0f
                var ty = 0
                var j = 0
                while (j < msg.size) {
                    val (t2, nj) = readVarint(msg, j)
                    j = nj
                    val f2 = t2 shr 3
                    val w2 = t2 and 7
                    when {
                        f2 == 1 && w2 == 2 -> {
                            val (l2, nj2) = readVarint(msg, j)
                            j = nj2
                            p = String(msg, j, l2, Charsets.UTF_8)
                            j += l2
                        }
                        f2 == 2 && w2 == 5 -> {
                            sc = ByteBuffer.wrap(msg, j, 4).order(ByteOrder.LITTLE_ENDIAN).float
                            j += 4
                        }
                        f2 == 3 && w2 == 0 -> {
                            val (v, nj3) = readVarint(msg, j)
                            j = nj3
                            ty = v
                        }
                        else -> {
                            when (w2) {
                                0 -> readVarint(msg, j)
                                2 -> { val (l, nn) = readVarint(msg, j); j = nn + l }
                                5 -> j += 4
                                1 -> j += 8
                                else -> j = msg.size
                            }
                        }
                    }
                }
                result.add(Piece(p, sc, ty))
            } else {
                when (wt) {
                    0 -> readVarint(bytes, i)
                    2 -> { val (l, nn) = readVarint(bytes, i); i = nn + l }
                    5 -> i += 4
                    1 -> i += 8
                    else -> i = bytes.size
                }
            }
        }
        return result
    }

    private fun readVarint(bytes: ByteArray, start: Int): Pair<Int, Int> {
        var i = start
        var result = 0L
        var shift = 0
        while (i < bytes.size) {
            val b = bytes[i].toInt() and 0xFF
            i++
            result = result or ((b.toLong() and 0x7FL) shl shift)
            if (b and 0x80 == 0) break
            shift += 7
        }
        return Pair(result.toInt(), i)
    }
}

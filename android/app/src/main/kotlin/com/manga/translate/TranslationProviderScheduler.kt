package com.manga.translate

import java.util.concurrent.atomic.AtomicInteger

data class AdditionalTranslationProvider(
    val name: String,
    val apiUrl: String,
    val apiKey: String,
    val modelName: String,
    val weight: Int,
    val enabled: Boolean = true
) {
    fun isConfigured(): Boolean {
        return apiUrl.isNotBlank() && apiKey.isNotBlank() && modelName.isNotBlank()
    }
}

data class WeightedProviderCandidate(
    val providerId: String,
    val displayName: String,
    val settings: ApiSettings,
    val weight: Int,
    val isPrimary: Boolean
)

data class PageTranslationProviderContext(
    val providerId: String,
    val displayName: String,
    val apiSettings: ApiSettings,
    val isPrimary: Boolean
)

class WeightedTranslationProviderScheduler(
    candidates: List<WeightedProviderCandidate>
) {
    private val normalizedCandidates = candidates.filter { it.weight > 0 }
    private val weightedSequence: List<WeightedProviderCandidate> =
        buildBalancedWeightedSequence(normalizedCandidates)
    private val nextIndex = AtomicInteger(0)

    fun orderedCandidatesForPage(): List<PageTranslationProviderContext> {
        if (weightedSequence.isEmpty()) return emptyList()
        // 每页推进一格起点(原子、跨页线程安全)，让加权序列在并发翻译的各页之间整体轮转，实现负载均衡。
        val start = nextIndex.getAndUpdate { current ->
            val next = current + 1
            if (next >= weightedSequence.size) 0 else next
        }
        // 从起点环形展开整条序列，但每个供应商只取首次出现：返回的是“本页的完整故障转移顺序”
        // （首选供应商失败后按此顺序回退），而权重只决定谁更可能排在前面，不产生重复条目。
        val ordered = ArrayList<PageTranslationProviderContext>(normalizedCandidates.size)
        val seen = LinkedHashSet<String>(normalizedCandidates.size)
        for (offset in weightedSequence.indices) {
            val candidate = weightedSequence[(start + offset) % weightedSequence.size]
            if (!seen.add(candidate.providerId)) continue
            ordered += candidate.toContext()
            if (seen.size == normalizedCandidates.size) {
                break
            }
        }
        return ordered
    }

    private fun WeightedProviderCandidate.toContext(): PageTranslationProviderContext {
        return PageTranslationProviderContext(
            providerId = providerId,
            displayName = displayName,
            apiSettings = settings,
            isPrimary = isPrimary
        )
    }

    private fun buildBalancedWeightedSequence(
        candidates: List<WeightedProviderCandidate>
    ): List<WeightedProviderCandidate> {
        if (candidates.isEmpty()) return emptyList()
        // 用“最大余额法”把各供应商按权重铺成一条长度=总权重的序列，使同一供应商的出现尽量均匀散开，
        // 而不是简单地 [A,A,A,B,B] 连续堆叠——后者会让前几页全压在 A 上，失去均衡意义。
        val remaining = candidates.associateWith { it.weight }.toMutableMap()
        val assigned = candidates.associateWith { 0 }.toMutableMap()
        val totalWeight = candidates.sumOf { it.weight }
        val sequence = ArrayList<WeightedProviderCandidate>(totalWeight)
        repeat(totalWeight) { step ->
            val progress = step + 1
            // 每一步选“理想累计配额(weight*进度/总权重)与已分配数差距最大”的供应商，即当前最该被补的那个；
            // 权重相同时优先权重大的以保持确定性。
            val selected = candidates
                .filter { (remaining[it] ?: 0) > 0 }
                .maxWithOrNull(
                    compareBy<WeightedProviderCandidate> { candidate ->
                        val target = candidate.weight.toDouble() * progress / totalWeight.toDouble()
                        target - (assigned[candidate] ?: 0).toDouble()
                    }.thenByDescending { it.weight }
                )
                ?: return@repeat
            sequence += selected
            remaining[selected] = (remaining[selected] ?: 0) - 1
            assigned[selected] = (assigned[selected] ?: 0) + 1
        }
        return sequence
    }
}

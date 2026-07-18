package com.manga.translate

internal class TextBubbleTranslationCoordinator(
    private val llmClient: LlmGateway,
    private val localTranslationEngine: MarianMtEngine? = null
) {

    suspend fun translateBubbles(
        bubbles: List<BubbleTranslation>,
        glossary: Map<String, String>,
        promptAsset: String,
        requestTimeoutMs: Int? = null,
        retryCount: Int = 3,
        apiSettings: ApiSettings? = null,
        language: TranslationLanguage = TranslationLanguage.JA_TO_ZH,
        logTag: String,
        translationMode: String
    ): TextBubbleTranslationBatchResult? {
        if (bubbles.isEmpty()) {
            return TextBubbleTranslationBatchResult(bubbles = bubbles, glossaryUsed = emptyMap())
        }
        val translatable = bubbles.filter { it.sourceText.isNotBlank() }
        if (translatable.isEmpty()) {
            AppLogger.log(logTag, "Skip translate: no translatable text")
            return TextBubbleTranslationBatchResult(bubbles = bubbles, glossaryUsed = emptyMap())
        }

        val resolvedApiSettings = apiSettings
        if (localTranslationEngine != null && localTranslationEngine.isAvailable()) {
            val results = mutableListOf<BubbleTranslation>()
            val removedIds = mutableSetOf<Int>()
            for (bubble in translatable) {
                val translated = localTranslationEngine.translate(bubble.sourceText, language)
                if (translated.isBlank() || translated == bubble.sourceText) {
                    removedIds.add(bubble.id)
                } else {
                    results.add(bubble.withTranslationResult(translated))
                }
            }
            // 合并未翻译的（空白的）气泡
            val finalBubbles = bubbles.map { bubble ->
                results.find { it.id == bubble.id } ?: bubble
            }.filterNot { it.id in removedIds }
            
            return TextBubbleTranslationBatchResult(
                bubbles = finalBubbles,
                glossaryUsed = emptyMap(),
                removedBubbleIds = removedIds
            )
        }
        if (!llmClient.isConfigured(resolvedApiSettings)) {
            AppLogger.log(logTag, "Skip translate: LLM client not configured")
            return null
        }

        val translatedMap = HashMap<Int, String>(translatable.size)
        val removedBubbleIds = LinkedHashSet<Int>()
        val cacheMisses = ArrayList<BubbleTranslation>(translatable.size)
        cacheMisses.addAll(translatable)

        fun merge(): List<BubbleTranslation> {
            return bubbles.filterNot { it.id in removedBubbleIds }.map { bubble ->
                translatedMap[bubble.id]?.let { translated ->
                    bubble.withTranslationResult(translated)
                } ?: bubble
            }
        }

        if (cacheMisses.isEmpty()) {
            return TextBubbleTranslationBatchResult(
                bubbles = merge(),
                glossaryUsed = emptyMap()
            )
        }

        AppLogger.log(logTag, "Translate request segments=${cacheMisses.size}")
        val requestItems = cacheMisses
            .sortedWith(compareBy({ it.rect.top }, { it.rect.left }, { it.id }))
            .map {
                LlmBubbleTranslationRequestItem(
                    id = it.id,
                    text = normalizeOcrText(it.sourceText, language)
                )
            }
        val translated = llmClient.translateBubbleItems(
            items = requestItems,
            glossary = glossary,
            promptAsset = promptAsset,
            requestTimeoutMs = requestTimeoutMs,
            retryCount = retryCount,
            apiSettings = resolvedApiSettings
        ) ?: return null

        val translationById = LinkedHashMap<Int, String>(translated.items.size)
        val duplicateIds = LinkedHashSet<Int>()
        for (item in translated.items) {
            val normalizedTranslation = item.translation.trim()
            if (translationById.putIfAbsent(item.id, normalizedTranslation) != null) {
                duplicateIds.add(item.id)
            }
        }
        val requestedIds = requestItems.map { it.id }
        val requestedIdSet = requestedIds.toSet()
        val unexpectedIds = translationById.keys.filter { it !in requestedIdSet }
        val missingIds = requestedIds.filterNot { translationById.containsKey(it) }
        if (
            duplicateIds.isNotEmpty() ||
            unexpectedIds.isNotEmpty() ||
            missingIds.isNotEmpty()
        ) {
            val error = buildStructuredTranslationErrorLog(
                mode = translationMode,
                requestedIds = requestedIds,
                duplicateIds = duplicateIds.toList(),
                unexpectedIds = unexpectedIds,
                missingIds = missingIds
            )
            AppLogger.log(logTag, error)
            throw LlmResponseException(LlmErrorCode.MissingTranslationItems, error)
        }

        for (source in cacheMisses) {
            val translatedText = translationById[source.id].orEmpty()
            if (translatedText.isBlank()) {
                removedBubbleIds.add(source.id)
            } else {
                translatedMap[source.id] = translatedText
            }
        }

        return TextBubbleTranslationBatchResult(
            bubbles = merge(),
            glossaryUsed = translated.glossaryUsed,
            removedBubbleIds = removedBubbleIds
        )
    }
}

internal data class TextBubbleTranslationBatchResult(
    val bubbles: List<BubbleTranslation>,
    val glossaryUsed: Map<String, String>,
    val removedBubbleIds: Set<Int> = emptySet()
)

private fun buildStructuredTranslationErrorLog(
    mode: String,
    requestedIds: List<Int>,
    duplicateIds: List<Int>,
    unexpectedIds: List<Int>,
    missingIds: List<Int>
): String {
    return buildString {
        append("Structured translation partial in ")
        append(mode)
        append(": requested=")
        append(summarizeIdsForLog(requestedIds))
        if (duplicateIds.isNotEmpty()) {
            append(", duplicate=")
            append(summarizeIdsForLog(duplicateIds))
        }
        if (unexpectedIds.isNotEmpty()) {
            append(", unexpected=")
            append(summarizeIdsForLog(unexpectedIds))
        }
        if (missingIds.isNotEmpty()) {
            append(", missing=")
            append(summarizeIdsForLog(missingIds))
        }
    }
}

private fun summarizeIdsForLog(ids: List<Int>, limit: Int = 12): String {
    if (ids.isEmpty()) return "[]"
    val normalized = ids.distinct()
    val shown = normalized.take(limit).joinToString(prefix = "[", postfix = "]")
    return if (normalized.size <= limit) shown else "$shown...(${normalized.size} total)"
}

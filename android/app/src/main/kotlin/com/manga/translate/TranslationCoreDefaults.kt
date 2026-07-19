package com.manga.translate

object TranslationCoreDefaults {
    const val DefaultDetectionInputSize = 640
    const val DefaultLineDetectionInputSize = 960

    // Aligned with upstream (jedzqer): 0.5 (not 0.7). A higher NMS IoU lets
    // overlapping duplicate detections survive, which split a single speech
    // bubble into multiple boxes. 0.5 matches upstream and keeps one box per bubble.
    const val BubbleDetectorNmsIouThreshold = 0.5f
    const val MinBalloonConfidence = 0.15f
    const val FreeTextOutputExpandRatio = 0.08f
    const val FreeTextOutputExpandMin = 1.0f

    const val PageRegionTextIouThreshold = 0.2f
    const val TinyBubbleShortSideMinPx = 12f
    const val TinyBubbleLongSideMinPx = 28f
    // Aligned with upstream (jedzqer): 0.02 (not 0.015). The lower Venera value
    // kept smaller bubble false positives (faces/clothing fragments, page-number
    // artifacts), surfacing as "text boxes" in non-text areas. 0.02 is upstream.
    const val TinyBubbleShortSideRatio = 0.02f
    const val TinyBubbleLongSideRatio = 0.035f
    const val TinyBubbleMaxAreaRatio = 0.0008f
    const val BubbleDedupIouThreshold = 0.65f

    const val VlBubbleExpandRatio = 0.1f
    const val VlBubbleExpandMin = 4f
}

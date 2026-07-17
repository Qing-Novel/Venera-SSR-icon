package com.manga.translate

object TranslationCoreDefaults {
    const val DefaultDetectionInputSize = 640
    const val DefaultLineDetectionInputSize = 960

    const val BubbleDetectorNmsIouThreshold = 0.7f
    const val MinBalloonConfidence = 0.15f
    const val FreeTextOutputExpandRatio = 0.08f
    const val FreeTextOutputExpandMin = 1.0f

    const val PageRegionTextIouThreshold = 0.2f
    const val TinyBubbleShortSideMinPx = 12f
    const val TinyBubbleLongSideMinPx = 28f
    const val TinyBubbleShortSideRatio = 0.015f
    const val TinyBubbleLongSideRatio = 0.035f
    const val TinyBubbleMaxAreaRatio = 0.0008f
    const val BubbleDedupIouThreshold = 0.65f

    const val VlBubbleExpandRatio = 0.1f
    const val VlBubbleExpandMin = 4f
}

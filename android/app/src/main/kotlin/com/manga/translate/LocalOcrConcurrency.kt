package com.manga.translate

import kotlin.math.min

internal object LocalOcrConcurrency {
    private const val MEMORY_PER_INSTANCE_BYTES = 45L * 1024L * 1024L
    private const val HARD_CAP = 3
    private const val THREADS_PER_INSTANCE = 1

    fun compute(): Int {
        val rt = Runtime.getRuntime()
        val byCpu = rt.availableProcessors() / THREADS_PER_INSTANCE
        val byMemory = (rt.maxMemory() / MEMORY_PER_INSTANCE_BYTES).toInt()
        return min(min(byCpu, byMemory), HARD_CAP).coerceAtLeast(1)
    }

    // 0 means auto; positive values override the computed result
    fun resolve(override: Int): Int = if (override > 0) override else compute()
}

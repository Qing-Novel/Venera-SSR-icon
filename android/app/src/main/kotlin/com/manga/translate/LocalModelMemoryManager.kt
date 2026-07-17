package com.manga.translate

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.CopyOnWriteArrayList

internal class LocalModelMemoryManager(
    private val releaseLoadedModels: () -> Unit
) {
    private val activeUsers = AtomicInteger(0)
    private val appInForeground = AtomicBoolean(false)
    private val releasedWhileBackground = AtomicBoolean(false)
    private val releaseCallbacks = CopyOnWriteArrayList<() -> Unit>()

    fun acquire(ownerTag: String): LocalModelLease {
        releasedWhileBackground.set(false)
        val count = activeUsers.incrementAndGet()
        AppLogger.log("LocalModelMemory", "Acquire $ownerTag, active=$count")
        return LocalModelLease(ownerTag, this)
    }

    fun setAppInForeground(foreground: Boolean) {
        val changed = appInForeground.getAndSet(foreground) != foreground
        if (changed) {
            AppLogger.log("LocalModelMemory", "App foreground=$foreground")
        }
        if (foreground) {
            releasedWhileBackground.set(false)
        }
        if (!foreground) {
            releaseIfIdleAndBackground()
        }
    }

    fun registerReleaseCallback(callback: () -> Unit): AutoCloseable {
        releaseCallbacks.add(callback)
        return AutoCloseable {
            releaseCallbacks.remove(callback)
        }
    }

    private fun release(ownerTag: String) {
        val count = activeUsers.decrementAndGet()
        if (count < 0) {
            activeUsers.set(0)
            AppLogger.log("LocalModelMemory", "Release count underflow from $ownerTag")
            return
        }
        AppLogger.log("LocalModelMemory", "Release $ownerTag, active=$count")
        releaseIfIdleAndBackground()
    }

    private fun releaseIfIdleAndBackground() {
        if (activeUsers.get() != 0 || appInForeground.get()) {
            return
        }
        if (!releasedWhileBackground.compareAndSet(false, true)) {
            return
        }
        AppLogger.log("LocalModelMemory", "Release loaded local models while app is background")
        releaseCallbacks.forEach { callback ->
            runCatching { callback() }
                .onFailure { AppLogger.log("LocalModelMemory", "Release callback failed", it) }
        }
        releaseLoadedModels()
    }

    internal class LocalModelLease(
        private val ownerTag: String,
        private val manager: LocalModelMemoryManager
    ) : AutoCloseable {
        private val closed = AtomicBoolean(false)

        override fun close() {
            if (closed.compareAndSet(false, true)) {
                manager.release(ownerTag)
            }
        }
    }
}

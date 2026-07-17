package com.manga.translate

import java.util.concurrent.atomic.AtomicReference

internal object TranslationCancellationRegistry {
    private val handlerRef = AtomicReference<(() -> Unit)?>(null)

    fun register(handler: () -> Unit) {
        handlerRef.set(handler)
    }

    fun clear() {
        handlerRef.set(null)
    }

    fun requestCancel(): Boolean {
        val handler = handlerRef.get() ?: return false
        handler.invoke()
        return true
    }
}

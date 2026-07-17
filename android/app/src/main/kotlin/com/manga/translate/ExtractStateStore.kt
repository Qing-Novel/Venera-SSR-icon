package com.manga.translate

import org.json.JSONArray
import java.io.File

class ExtractStateStore {
    fun load(folder: File, targetKey: String = DEFAULT_TARGET_KEY): MutableSet<String> {
        val file = stateFileFor(folder, targetKey)
        if (!file.exists()) return mutableSetOf()
        return try {
            val json = JSONArray(file.readText())
            val result = mutableSetOf<String>()
            for (i in 0 until json.length()) {
                val name = json.optString(i).trim()
                if (name.isNotBlank()) {
                    result.add(name)
                }
            }
            result
        } catch (e: Exception) {
            AppLogger.log("ExtractStateStore", "Failed to load extract state for ${folder.name}", e)
            mutableSetOf()
        }
    }

    fun save(folder: File, extracted: Set<String>, targetKey: String = DEFAULT_TARGET_KEY) {
        val json = JSONArray()
        for (name in extracted) {
            json.put(name)
        }
        stateFileFor(folder, targetKey).writeText(json.toString())
    }

    private fun stateFileFor(folder: File, targetKey: String): File {
        val normalizedTarget = targetKey.trim().lowercase()
        val fileName = when (normalizedTarget) {
            "", DEFAULT_TARGET_KEY -> ".extract-state.json"
            else -> ".extract-state_${normalizedTarget}.json"
        }
        return File(folder, fileName)
    }

    companion object {
        private const val DEFAULT_TARGET_KEY = "zh_hans"
    }
}

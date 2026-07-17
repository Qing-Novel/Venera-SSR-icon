package com.manga.translate

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.security.DigestInputStream
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap

enum class OnnxThreadProfile(
    val intraOpThreads: Int,
    val interOpThreads: Int
) {
    SINGLE(1, 1),
    LIGHT(2, 1);

    companion object {
        fun forCpuBoundWork(parallelism: Int): OnnxThreadProfile {
            return if (parallelism <= 1) LIGHT else SINGLE
        }
    }
}

object OnnxRuntimeSupport {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessionCache = ConcurrentHashMap<String, OrtSession>()

    fun environment(): OrtEnvironment = env

    fun getOrCreateSession(
        cacheDir: File,
        assetProvider: (String) -> java.io.InputStream,
        assetName: String,
        threadProfile: OnnxThreadProfile,
        useXnnpack: Boolean = true
    ): OrtSession {
        val cacheKey = "${cacheDir.absolutePath}|$assetName|${threadProfile.name}|xnnpack=$useXnnpack"
        sessionCache[cacheKey]?.let { return it }
        synchronized(cacheLock) {
            sessionCache[cacheKey]?.let { return it }
            val session = createSessionWithRecovery(
                cacheDir = cacheDir,
                assetProvider = assetProvider,
                assetName = assetName,
                threadProfile = threadProfile,
                useXnnpack = useXnnpack
            )
            sessionCache[cacheKey] = session
            return session
        }
    }

    private fun createSessionWithRecovery(
        cacheDir: File,
        assetProvider: (String) -> java.io.InputStream,
        assetName: String,
        threadProfile: OnnxThreadProfile,
        useXnnpack: Boolean
    ): OrtSession {
        val modelFile = copyAssetToCacheIfMissing(cacheDir, assetProvider, assetName)
        AppLogger.log(
            "OnnxRuntime",
            "Creating session for $assetName (${modelFile.length()} bytes), " +
                "threads=${threadProfile.name}, xnnpack=$useXnnpack"
        )
        return try {
            createSession(modelFile, threadProfile, useXnnpack)
        } catch (e: OrtException) {
            if (useXnnpack) {
                AppLogger.log(
                    "OnnxRuntime",
                    "Session create with XNNPACK failed for $assetName, retrying plain CPU",
                    e
                )
                return try {
                    createSession(modelFile, threadProfile, useXnnpack = false)
                } catch (cpuError: OrtException) {
                    if (!shouldRebuildCache(cpuError) && !shouldRebuildCache(e)) throw cpuError
                    rebuildAndCreateSession(cacheDir, assetProvider, assetName, threadProfile, useXnnpack = false)
                }
            }
            if (!shouldRebuildCache(e)) throw e
            rebuildAndCreateSession(cacheDir, assetProvider, assetName, threadProfile, useXnnpack = false)
        }.also {
            AppLogger.log("OnnxRuntime", "Session ready for $assetName")
        }
    }

    private fun rebuildAndCreateSession(
        cacheDir: File,
        assetProvider: (String) -> java.io.InputStream,
        assetName: String,
        threadProfile: OnnxThreadProfile,
        useXnnpack: Boolean
    ): OrtSession {
        AppLogger.log(
            "OnnxRuntime",
            "Model cache looks corrupted, rebuilding $assetName"
        )
        deleteCachedModel(cacheDir, assetName)
        val rebuiltFile = forceCopyAssetToCache(cacheDir, assetProvider, assetName)
        return createSession(rebuiltFile, threadProfile, useXnnpack)
    }

    private fun createSession(
        modelFile: File,
        threadProfile: OnnxThreadProfile,
        useXnnpack: Boolean
    ): OrtSession {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threadProfile.intraOpThreads)
            setInterOpNumThreads(threadProfile.interOpThreads)
            // Large YOLO-seg graphs can OOM / SIGSEGV under full graph opts + mem pattern.
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            setMemoryPatternOptimization(false)
            if (useXnnpack) {
                try {
                    addXnnpack(emptyMap())
                } catch (e: OrtException) {
                    AppLogger.log("OnnxRuntime", "XNNPACK unavailable, using plain CPU", e)
                }
            }
        }
        return options.use {
            env.createSession(modelFile.absolutePath, it)
        }
    }

    fun forceCopyAssetToCache(
        cacheDir: File,
        assetProvider: (String) -> java.io.InputStream,
        assetName: String
    ): File {
        val target = File(cacheDir, assetName)
        val expectedHash = computeAssetSha256(assetProvider, assetName)
        synchronized(cacheLock) {
            if (target.exists()) {
                target.delete()
            }
            writeAssetToCache(target, assetProvider, assetName)
            writeAssetHash(cacheDir, assetName, expectedHash)
        }
        return target
    }

    fun copyAssetToCacheIfMissing(
        cacheDir: File,
        assetProvider: (String) -> java.io.InputStream,
        assetName: String
    ): File {
        val target = File(cacheDir, assetName)
        val expectedHash = computeAssetSha256(assetProvider, assetName)
        synchronized(cacheLock) {
            if (
                target.exists() &&
                target.length() > 0L &&
                readCachedAssetHash(cacheDir, assetName) == expectedHash
            ) {
                return target
            }
            if (target.exists()) {
                target.delete()
            }
            deleteCachedAssetHash(cacheDir, assetName)
            AppLogger.log("OnnxRuntime", "Refreshing cached asset $assetName")
            writeAssetToCache(target, assetProvider, assetName)
            writeAssetHash(cacheDir, assetName, expectedHash)
        }
        return target
    }

    fun closeCachedSessions() {
        val sessions = synchronized(cacheLock) {
            val current = sessionCache.values.toList()
            sessionCache.clear()
            current
        }
        sessions.forEach { session ->
            runCatching { session.close() }
                .onFailure { AppLogger.log("OnnxRuntime", "Failed to close cached session", it) }
        }
        if (sessions.isNotEmpty()) {
            AppLogger.log("OnnxRuntime", "Closed ${sessions.size} cached sessions")
        }
    }

    private fun writeAssetToCache(
        target: File,
        assetProvider: (String) -> java.io.InputStream,
        assetName: String
    ) {
        target.parentFile?.mkdirs()
        val tempFile = File(target.parentFile, "${target.name}.tmp")
        if (tempFile.exists()) {
            tempFile.delete()
        }
        try {
            assetProvider(assetName).use { input ->
                FileOutputStream(tempFile).use { output ->
                    input.copyTo(output)
                    output.fd.sync()
                }
            }
            if (!tempFile.renameTo(target)) {
                tempFile.copyTo(target, overwrite = true)
                if (!tempFile.delete()) {
                    tempFile.deleteOnExit()
                }
            }
        } catch (t: Throwable) {
            if (tempFile.exists()) {
                tempFile.delete()
            }
            throw t
        }
    }

    private fun computeAssetSha256(
        assetProvider: (String) -> java.io.InputStream,
        assetName: String
    ): String {
        val digest = MessageDigest.getInstance("SHA-256")
        assetProvider(assetName).use { input ->
            DigestInputStream(input, digest).use { stream ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (stream.read(buffer) != -1) {
                    // DigestInputStream updates the digest as bytes are read.
                }
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun assetHashFile(cacheDir: File, assetName: String): File {
        return File(cacheDir, "$assetName.sha256")
    }

    private fun readCachedAssetHash(cacheDir: File, assetName: String): String? {
        val hashFile = assetHashFile(cacheDir, assetName)
        return runCatching { hashFile.readText().trim() }.getOrNull()?.takeIf { it.isNotEmpty() }
    }

    private fun writeAssetHash(cacheDir: File, assetName: String, hash: String) {
        val hashFile = assetHashFile(cacheDir, assetName)
        hashFile.parentFile?.mkdirs()
        val tempFile = File(hashFile.parentFile, "${hashFile.name}.tmp")
        tempFile.writeText(hash)
        if (!tempFile.renameTo(hashFile)) {
            hashFile.writeText(hash)
            tempFile.delete()
        }
    }

    private fun deleteCachedAssetHash(cacheDir: File, assetName: String) {
        val hashFile = assetHashFile(cacheDir, assetName)
        if (hashFile.exists() && !hashFile.delete()) {
            hashFile.deleteOnExit()
        }
    }

    private fun deleteCachedModel(cacheDir: File, assetName: String) {
        val target = File(cacheDir, assetName)
        if (target.exists() && !target.delete()) {
            target.deleteOnExit()
        }
        deleteCachedAssetHash(cacheDir, assetName)
        val tempFile = File(cacheDir, "$assetName.tmp")
        if (tempFile.exists() && !tempFile.delete()) {
            tempFile.deleteOnExit()
        }
    }

    private fun shouldRebuildCache(error: OrtException): Boolean {
        val message = error.message.orEmpty()
        return message.contains("ORT_INVALID_PROTOBUF", ignoreCase = true) ||
            message.contains("Protobuf parsing failed", ignoreCase = true) ||
            message.contains("Load model from", ignoreCase = true)
    }

    private val cacheLock = Any()
}

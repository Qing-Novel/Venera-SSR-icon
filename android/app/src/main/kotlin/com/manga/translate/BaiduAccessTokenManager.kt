package com.manga.translate

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class BaiduAccessTokenManager(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences(
        PREFS_NAME, Context.MODE_PRIVATE
    )
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()
    private val mutex = Mutex()

    companion object {
        private const val PREFS_NAME = "baidu_ocr_token"
        private const val KEY_TOKEN = "access_token"
        private const val KEY_EXPIRES_AT = "expires_at"
        private const val KEY_CREDENTIAL_HASH = "credential_hash"
        private const val TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
        // 提前 1 小时刷新，避免边界过期
        private const val REFRESH_BEFORE_MS = 3_600_000L
    }

    suspend fun getAccessToken(apiKey: String, secretKey: String): String? {
        if (apiKey.isBlank() || secretKey.isBlank()) return null
        mutex.withLock {
            val cached = loadFromCache(apiKey, secretKey)
            if (cached != null) return cached
            return fetchAndCache(apiKey, secretKey)
        }
    }

    private fun loadFromCache(apiKey: String, secretKey: String): String? {
        val storedHash = prefs.getString(KEY_CREDENTIAL_HASH, null) ?: return null
        val currentHash = credentialHash(apiKey, secretKey)
        if (storedHash != currentHash) {
            // Credentials changed; clear stale cache
            clearCache()
            return null
        }
        val token = prefs.getString(KEY_TOKEN, null) ?: return null
        val expiresAt = prefs.getLong(KEY_EXPIRES_AT, 0L)
        if (System.currentTimeMillis() + REFRESH_BEFORE_MS >= expiresAt) {
            return null
        }
        return token
    }

    private fun credentialHash(apiKey: String, secretKey: String): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        val hash = digest.digest("$apiKey:$secretKey".toByteArray(Charsets.UTF_8))
        return hash.joinToString("") { "%02x".format(it) }
    }

    private suspend fun fetchAndCache(apiKey: String, secretKey: String): String? =
        withContext(Dispatchers.IO) {
            try {
                val url = "$TOKEN_URL?grant_type=client_credentials" +
                    "&client_id=${java.net.URLEncoder.encode(apiKey, "UTF-8")}" +
                    "&client_secret=${java.net.URLEncoder.encode(secretKey, "UTF-8")}"
                val request = Request.Builder()
                    .url(url)
                    .get()
                    .build()
                val response = httpClient.newCall(request).execute()
                val body = response.body?.string().orEmpty()
                if (response.code !in 200..299) {
                    AppLogger.log(
                        "BaiduToken",
                        "Token fetch failed HTTP ${response.code}: ${summarizeBody(body)}"
                    )
                    return@withContext null
                }
                val json = JSONObject(body)
                val token = json.optString("access_token").takeIf { it.isNotBlank() }
                    ?: return@withContext null
                val expiresIn = json.optLong("expires_in", 0L)
                val expiresAt = if (expiresIn > 0) {
                    System.currentTimeMillis() + expiresIn * 1000
                } else {
                    // 默认 30 天
                    System.currentTimeMillis() + 30L * 24 * 3600 * 1000
                }
                val currentHash = credentialHash(apiKey, secretKey)
                prefs.edit()
                    .putString(KEY_CREDENTIAL_HASH, currentHash)
                    .putString(KEY_TOKEN, token)
                    .putLong(KEY_EXPIRES_AT, expiresAt)
                    .apply()
                token
            } catch (e: Exception) {
                AppLogger.log("BaiduToken", "Token fetch error", e)
                null
            }
        }

    fun clearCache() {
        prefs.edit().clear().apply()
    }

    private fun summarizeBody(body: String): String {
        return if (body.length > 200) body.take(200) + "..." else body
    }
}

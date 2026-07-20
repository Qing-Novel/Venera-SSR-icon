package com.github.kiastr.venera_ssr.translate

import android.app.Activity
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.os.Build
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

/**
 * 下载后批量翻译的进度通知（通知栏）。仅 Android 生效。
 *
 * 通过 MethodChannel 接收 Dart 侧 [com.github.kiastr.venera_ssr/translation_notify] 的调用：
 *   start(title, total)           初始化通道 + 申请权限（API33+）+ 显示 0/total
 *   update(title, done, total)    更新进度条
 *   complete(title, done, total, ok)  用完成摘要替换进度通知（可点击关闭）
 *   cancel()                      移除通知
 *
 * 全程不抛异常到 Dart 侧：通知能力缺失（如权限被拒、通道创建失败）时静默降级，
 * 真实翻译任务不受影响。
 */
class TranslationNotifyPlugin(private val context: Context) : MethodChannel.MethodCallHandler {

    companion object {
        const val CHANNEL = "com.github.kiastr.venera_ssr/translation_notify"
        private const val CHANNEL_ID = "venera_translation_progress"
        private const val NOTIFY_ID = 778899
        private const val PERM_REQUEST_CODE = 9001

        fun registerWith(context: Context, messenger: BinaryMessenger) {
            MethodChannel(messenger, CHANNEL).setMethodCallHandler(TranslationNotifyPlugin(context))
        }
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        try {
            when (call.method) {
                "start" -> {
                    val title = call.argument<String>("title") ?: "Translation"
                    val total = call.argument<Int>("total") ?: 0
                    ensureChannel()
                    requestPermissionIfNeeded()
                    showProgress(title, "0 / $total", 0, total)
                    result.success(null)
                }
                "update" -> {
                    val title = call.argument<String>("title") ?: "Translation"
                    val done = call.argument<Int>("done") ?: 0
                    val total = call.argument<Int>("total") ?: 0
                    showProgress(title, "$done / $total", done, total)
                    result.success(null)
                }
                "complete" -> {
                    val title = call.argument<String>("title") ?: "Translation"
                    val done = call.argument<Int>("done") ?: 0
                    val total = call.argument<Int>("total") ?: 0
                    val ok = call.argument<Boolean>("ok") ?: true
                    val body = if (ok) {
                        "Completed: $done / $total pages"
                    } else {
                        "Finished with errors: $done / $total"
                    }
                    showComplete(title, body)
                    result.success(null)
                }
                "cancel" -> {
                    NotificationManagerCompat.from(context).cancel(NOTIFY_ID)
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        } catch (_: Throwable) {
            // 通知失败绝不应中断翻译流程
            result.success(null)
        }
    }

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val mgr = context.getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager ?: return
        if (mgr.getNotificationChannel(CHANNEL_ID) != null) return
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Translation Progress",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Shows progress of background manga translation"
            setShowBadge(false)
        }
        mgr.createNotificationChannel(channel)
    }

    private fun requestPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val mgr = NotificationManagerCompat.from(context)
        if (mgr.areNotificationsEnabled()) return
        val activity = context as? Activity ?: return
        ActivityCompat.requestPermissions(
            activity,
            arrayOf(android.Manifest.permission.POST_NOTIFICATIONS),
            PERM_REQUEST_CODE
        )
    }

    private fun contentIntent(): PendingIntent? {
        val launch = context.packageManager.getLaunchIntentForPackage(context.packageName) ?: return null
        return PendingIntent.getActivity(context, 0, launch, PendingIntent.FLAG_IMMUTABLE)
    }

    private fun showProgress(title: String, body: String, done: Int, total: Int) {
        ensureChannel()
        val builder = NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(body)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setProgress(total, done, false)
        val intent = contentIntent()
        if (intent != null) builder.setContentIntent(intent)
        NotificationManagerCompat.from(context).notify(NOTIFY_ID, builder.build())
    }

    private fun showComplete(title: String, body: String) {
        ensureChannel()
        val builder = NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(body)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(false)
            .setAutoCancel(true)
            .setProgress(0, 0, false)
        val intent = contentIntent()
        if (intent != null) builder.setContentIntent(intent)
        NotificationManagerCompat.from(context).notify(NOTIFY_ID, builder.build())
    }
}

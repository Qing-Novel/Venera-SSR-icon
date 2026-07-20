import 'dart:io';

import 'package:flutter/services.dart';

/// 下载后批量翻译的进度通知（通知栏）。
///
/// 仅 Android 生效：通过 MethodChannel [com.github.kiastr.venera_ssr/translation_notify]
/// 调用 Kotlin 侧 [TranslationNotifyPlugin]，由原生 NotificationManager 显示带进度条的
/// 通知。其它平台（iOS/Linux/macOS/Windows）调用均为 no-op，不影响翻译流程与平台构建。
///
/// 设计原则：
/// - 全程 try/catch，任何异常都不影响真实的翻译流程。
/// - 仅 Android 真正发起 MethodChannel 调用；其余平台静默跳过。
class TranslationNotification {
  static const MethodChannel _channel =
      MethodChannel('com.github.kiastr.venera_ssr/translation_notify');

  static Future<void> start(String title, int total) async {
    if (!Platform.isAndroid) return;
    try {
      await _channel.invokeMethod('start', {'title': title, 'total': total});
    } catch (_) {
      // 通知能力缺失时静默降级
    }
  }

  /// 每翻译完成一页调用：更新进度条。
  static Future<void> update(String title, int done, int total) async {
    if (!Platform.isAndroid) return;
    try {
      await _channel.invokeMethod(
          'update', {'title': title, 'done': done, 'total': total});
    } catch (_) {
      // 忽略：进度通知失败不应中断翻译
    }
  }

  /// 翻译结束时调用：用一份完成摘要替换进度通知。
  static Future<void> complete(String title, int done, int total,
      {bool ok = true}) async {
    if (!Platform.isAndroid) return;
    try {
      await _channel.invokeMethod('complete',
          {'title': title, 'done': done, 'total': total, 'ok': ok});
    } catch (_) {
      // 忽略
    }
  }

  static Future<void> cancel() async {
    if (!Platform.isAndroid) return;
    try {
      await _channel.invokeMethod('cancel');
    } catch (_) {
      // 忽略
    }
  }
}

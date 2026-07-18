import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:venera/utils/file_type.dart';
import 'package:venera/utils/log.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:venera/foundation/log.dart';

/// 漫画翻译服务（下载后批量 / 阅读时实时）
///
/// 移植自 jedzqer/manga-translator-android 的检测 + OCR 端上能力，文本翻译走远程 LLM。
/// 端上完成「气泡检测 + OCR + 译文重绘」后返回合成 PNG 字节，Dart 侧直接当图片显示
/// （与原生 colorize 通道返回 PNG 的契约一致）。
///
/// 与 [ColorizationService] 同构：单例 + MethodChannel
/// [com.github.kiastr.venera_ssr/translate]（method `translateImage`）+
/// 磁盘缓存 + 并发受限的任务队列。
///
/// 注意：翻译依赖用户在设置中配置的远程 LLM 与端上检测/OCR 模型（仓库内不含模型）。
/// 未配置 LLM 时原生管道返回 null，本服务对应当页返回 null，阅读器回退显示原图。
class TranslationService {
  TranslationService._internal();

  static final TranslationService _instance = TranslationService._internal();

  factory TranslationService() => _instance;

  static TranslationService get instance => _instance;

  /// 与原生端通信的 MethodChannel
  static const MethodChannel _channel =
      MethodChannel('com.github.kiastr.venera_ssr/translate');

  /// 缓存目录路径
  String? _cacheDir;

  /// 正在处理的任务集合（避免重复处理同一图片）
  final Set<String> _processingKeys = {};

  /// 会话级负缓存：记录本次运行内翻译失败/未配置的 key，避免阅读时反复打远程 LLM
  final Set<String> _negativeCache = {};

  /// 最大并发处理数（与 BubbleRenderer 内部的渲染信号量互补）
  static const int _maxConcurrentTasks = 2;

  /// 当前正在运行的任务数
  int _runningTasks = 0;

  /// 任务队列
  final List<Function> _taskQueue = [];

  /// 初始化缓存目录
  Future<void> init() async {
    try {
      final dir = await getTemporaryDirectory();
      _cacheDir = path.join(dir.path, 'translation_cache');
      final cacheDirectory = Directory(_cacheDir!);
      if (!await cacheDirectory.exists()) {
        await cacheDirectory.create(recursive: true);
      }
    } catch (e) {
      Log.error('Translation', 'Translation init error: $e');
    }
  }

  /// 获取缓存文件路径
  String? _getCachePath(String key) {
    if (_cacheDir == null) return null;
    return path.join(_cacheDir!, '${key.hashCode.abs()}.png');
  }

  /// 检查是否有缓存，有则返回缓存数据
  Future<Uint8List?> _getFromCache(String key) async {
    final cachePath = _getCachePath(key);
    if (cachePath == null) return null;

    final file = File(cachePath);
    if (await file.exists()) {
      try {
        return await file.readAsBytes();
      } catch (e) {
        return null;
      }
    }
    return null;
  }

  /// 保存处理结果到缓存
  Future<void> _saveToCache(String key, Uint8List data) async {
    final cachePath = _getCachePath(key);
    if (cachePath == null) return;

    try {
      final file = File(cachePath);
      await file.writeAsBytes(data);
    } catch (e) {
      Log.error('Translation', 'Translation cache save error: $e');
    }
  }

  /// 调用原生端完成翻译并重绘
  ///
  /// 返回合成后的翻译 PNG 字节；失败（未配置 LLM / 渲染失败）返回 null（不抛异常）。
  Future<Uint8List?> _translateOnNative(
    Uint8List imageBytes,
    String language,
    bool forceOcr,
  ) async {
    // 检查图片格式是否受支持。
    // 原生端的 TranslationPipeline 依赖文件名扩展名或特定的解码器路由，
    // 对于不支持的格式（如 SVG）或无法识别的字节流，直接在 Dart 侧拦截以避免原生崩溃。
    final fileType = detectFileType(imageBytes);
    final mime = fileType.mime;
    final isSupported = mime.startsWith('image/') &&
        !mime.contains('svg') &&
        !mime.contains('gif'); // 翻译流水线目前主要处理静态图

    if (!isSupported) {
      Log.warning(
        'Translation',
        'unsupported image format ($mime), skipping native translation',
      );
      return null;
    }

    try {
      final result = await _channel.invokeMethod<Uint8List>('translateImage', {
        'imageBytes': imageBytes,
        'language': language,
        'forceOcr': forceOcr,
        'renderImage': true,
      });
      return result;
    } on PlatformException catch (e) {
      Log.warning(
        'Translation',
        'native translate failed (${e.code}): ${e.message}',
      );
      return null;
    } catch (e, s) {
      Log.error('Translation', 'native translate failed: $e\n$s');
      return null;
    }
  }

  /// 处理图片字节数据，返回翻译后（重绘译文）的 PNG 字节。
  ///
  /// 与 [ColorizationService.processImage] 同构：
  /// [imageBytes] 原始图片字节，[cacheKey] 缓存键（通常用图片路径或 provider.key），
  /// [language] 目标语言 pref 值（如 `ja_to_zh`），[forceOcr] 是否强制 OCR。
  Future<Uint8List?> processImage({
    required Uint8List imageBytes,
    required String cacheKey,
    String language = 'ja_to_zh',
    bool forceOcr = false,
  }) async {
    final fullKey = '${cacheKey}_${language}_${forceOcr ? 1 : 0}';

    // 会话级负缓存：本次运行内已确认翻译不可用（未配置 LLM 等），直接跳过
    if (_negativeCache.contains(fullKey)) {
      return null;
    }

    final cached = await _getFromCache(fullKey);
    if (cached != null) {
      Log.info('Translation', 'cache hit for $cacheKey');
      return cached;
    }

    // 防止重复处理同一图片
    if (_processingKeys.contains(fullKey)) {
      Log.info('Translation', 'already processing $cacheKey');
      return null;
    }

    _processingKeys.add(fullKey);

    return _enqueueTask(() async {
      try {
        Log.info('Translation', 'translating image $cacheKey (lang=$language)');

        final result = await _translateOnNative(imageBytes, language, forceOcr);

        if (result != null) {
          await _saveToCache(fullKey, result);
          Log.info('Translation', 'translation complete for $cacheKey');
        } else {
          // 翻译不可用（未配置 LLM 等）：记入负缓存，本次运行内不再重试
          _negativeCache.add(fullKey);
          Log.warning('Translation', 'translation returned null for $cacheKey');
        }

        return result;
      } catch (e, s) {
        Log.error('Translation', 'Translation processing error: $e\n$s');
        return null;
      } finally {
        _processingKeys.remove(fullKey);
      }
    });
  }

  /// 将任务加入队列并按并发上限执行
  Future<T?> _enqueueTask<T>(Future<T?> Function() task) async {
    final completer = Completer<T?>();

    _taskQueue.add(() async {
      _runningTasks++;
      try {
        final result = await task();
        completer.complete(result);
      } catch (e) {
        completer.completeError(e);
      } finally {
        _runningTasks--;
        _nextTask();
      }
    });

    _nextTask();
    return completer.future;
  }

  /// 执行下一个任务
  void _nextTask() {
    if (_runningTasks < _maxConcurrentTasks && _taskQueue.isNotEmpty) {
      final task = _taskQueue.removeAt(0);
      task();
    }
  }

  /// 处理本地图片文件，返回翻译后的 PNG 字节（用于下载后批量翻译）。
  Future<Uint8List?> processFile({
    required String filePath,
    String language = 'ja_to_zh',
    bool forceOcr = false,
  }) async {
    try {
      final file = File(filePath);
      if (!await file.exists()) return null;

      final imageBytes = await file.readAsBytes();
      return processImage(
        imageBytes: imageBytes,
        cacheKey: filePath,
        language: language,
        forceOcr: forceOcr,
      );
    } catch (e) {
      Log.error('Translation', 'Translation file processing error: $e');
      return null;
    }
  }

  /// 读取原生端保存的远程 LLM 配置（apiUrl / apiKey / modelName / apiFormat / configured）。
  Future<Map<String, dynamic>?> getLlmConfig() async {
    try {
      final res = await _channel.invokeMethod<Map>('getLlmConfig');
      return res?.cast<String, dynamic>();
    } catch (e, s) {
      Log.error('Translation', 'getLlmConfig failed: $e\n$s');
      return null;
    }
  }

  /// 远程 LLM 是否已配置（apiUrl + apiKey + modelName 均非空）。
  Future<bool> isLlmConfigured() async {
    final cfg = await getLlmConfig();
    return cfg?['configured'] == true;
  }

  /// 保存远程 LLM 配置到原生端（jedzqer SettingsStore）。
  /// 配置变更后清空会话负缓存，使此前因"未配置"被跳过的页可重新翻译。
  Future<bool> setLlmConfig({
    required String apiUrl,
    required String apiKey,
    required String modelName,
    required String apiFormat,
  }) async {
    try {
      await _channel.invokeMethod<bool>('setLlmConfig', {
        'apiUrl': apiUrl,
        'apiKey': apiKey,
        'modelName': modelName,
        'apiFormat': apiFormat,
      });
      _negativeCache.clear();
      return true;
    } catch (e, s) {
      Log.error('Translation', 'setLlmConfig failed: $e\n$s');
      return false;
    }
  }

  /// 清除所有翻译缓存（含负缓存）
  Future<void> clearCache() async {
    _negativeCache.clear();
    if (_cacheDir == null) return;
    try {
      final dir = Directory(_cacheDir!);
      if (await dir.exists()) {
        await dir.delete(recursive: true);
        await dir.create(recursive: true);
      }
      Log.info('Translation', 'Translation: cache cleared');
    } catch (e) {
      Log.error('Translation', 'Translation cache clear error: $e');
    }
  }

  /// 获取缓存占用的磁盘大小（字节）
  Future<int> getCacheSize() async {
    if (_cacheDir == null) return 0;
    try {
      final dir = Directory(_cacheDir!);
      if (!await dir.exists()) return 0;

      int totalSize = 0;
      await for (final entity in dir.list(recursive: true)) {
        if (entity is File) {
          totalSize += await entity.length();
        }
      }
      return totalSize;
    } catch (e) {
      return 0;
    }
  }
}

import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:venera/foundation/log.dart';

/// 模型管理器：负责从网络下载 DeOldify 模型到应用目录。
///
/// 不使用 Flutter assets 打包模型（模型文件 ~243MB 会导致 Android APK 构建失败），
/// 改为在设置页由用户手动触发下载，避免首次启动即下载大文件。
///
/// 实际的推理已移至原生端（Kotlin + OpenCV + ONNX Runtime），
/// 见 android/app/src/main/kotlin/com/github/kiastr/venera_ssr/colorize/，
/// 通过 MethodChannel 调用。本文件只负责模型的下载与本地路径解析。
class ColorizationModelManager {
  static const modelFileName = 'deoldify_artistic.onnx';
  static const expectedFileSize = 243 * 1024 * 1024; // ~243MB

  /// 默认镜像源（按稳定性排序），用户可在设置页增删
  static const String _modelUrlsKey = 'colorization_model_urls';
  static const String _customModelPathKey = 'colorization_custom_model_path';
  static const List<String> _defaultModelUrls = [
    'https://mirror.ghproxy.com/https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
    'https://ghp.ci/https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
    'https://ghproxy.net/https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
    'https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
  ];

  /// 持久化的镜像 URL 列表（用户可编辑）；首次运行初始化为默认列表
  static List<String> _modelUrls = [];
  static bool _urlsLoaded = false;

  /// 用户自选的本地模型文件路径（优先于下载模型）；null 表示未使用
  static String? _customModelPath;

  /// 获取当前生效的镜像 URL 列表（懒加载 + 持久化）
  static Future<List<String>> getModelUrls() async {
    if (!_urlsLoaded) {
      final prefs = await SharedPreferences.getInstance();
      final saved = prefs.getStringList(_modelUrlsKey);
      _modelUrls = (saved != null && saved.isNotEmpty)
          ? List.from(saved)
          : List.from(_defaultModelUrls);
      _urlsLoaded = true;
    }
    return List.from(_modelUrls);
  }

  /// 添加一个自定义镜像 URL（去重）
  static Future<void> addModelUrl(String url) async {
    await getModelUrls();
    final trimmed = url.trim();
    if (trimmed.isEmpty || _modelUrls.contains(trimmed)) return;
    _modelUrls.add(trimmed);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_modelUrlsKey, _modelUrls);
  }

  /// 删除指定下标的镜像 URL
  static Future<void> removeModelUrlAt(int index) async {
    await getModelUrls();
    if (index < 0 || index >= _modelUrls.length) return;
    _modelUrls.removeAt(index);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_modelUrlsKey, _modelUrls);
  }

  /// 恢复默认镜像列表
  static Future<void> resetModelUrls() async {
    _modelUrls = List.from(_defaultModelUrls);
    _urlsLoaded = true;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_modelUrlsKey, _modelUrls);
  }

  /// 获取用户自选的本地模型路径
  static Future<String?> getCustomModelPath() async {
    if (_customModelPath != null) return _customModelPath;
    final prefs = await SharedPreferences.getInstance();
    _customModelPath = prefs.getString(_customModelPathKey);
    return _customModelPath;
  }

  /// 设置用户自选的本地模型文件路径（传 null 清除）
  static Future<void> setCustomModelPath(String? path) async {
    _customModelPath = path;
    final prefs = await SharedPreferences.getInstance();
    if (path == null) {
      await prefs.remove(_customModelPathKey);
    } else {
      await prefs.setString(_customModelPathKey, path);
    }
  }

  /// 从用户选定的文件字节导入为自定义模型。
  ///
  /// 物化到应用私有目录（真实文件路径），确保 Android 上的 content URI
  /// 与原生 ONNX Runtime 都能正确读取。返回最终存储路径。
  static Future<String> importCustomModel(Uint8List bytes) async {
    if (bytes.length < 1024 * 1024) {
      throw Exception('File too small, invalid model');
    }
    final dir = await getApplicationSupportDirectory();
    final dest = path.join(dir.path, 'custom_colorization_model.onnx');
    await File(dest).writeAsBytes(bytes, flush: true);
    _customModelPath = dest;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_customModelPathKey, dest);
    return dest;
  }

  static String? _cachedModelPath;
  static bool _isDownloading = false;
  static double _downloadProgress = 0.0;
  static String? _currentStatus;

  /// 获取模型文件路径。优先返回用户自选的本地模型，其次已下载的默认模型。
  /// 两者都不可用则返回 null（不自动下载）。
  static Future<String?> ensureModelAvailable() async {
    // 1) 用户自选的本地模型文件（优先）
    final custom = await getCustomModelPath();
    if (custom != null) {
      final f = File(custom);
      if (await f.exists()) {
        final size = await f.length();
        // 自定义模型大小未知，仅做基本有效性校验
        if (size > 1024 * 1024) {
          _cachedModelPath = custom;
          return custom;
        }
      }
      // 自选路径失效，清除
      await setCustomModelPath(null);
    }

    // 2) 已下载的默认模型
    if (_cachedModelPath != null) return _cachedModelPath!;

    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final targetFile = File(targetPath);

    // 检查是否已存在且大小合理
    if (await targetFile.exists()) {
      final size = await targetFile.length();
      if (size > expectedFileSize ~/ 2) {
        _cachedModelPath = targetPath;
        return targetPath;
      }
      // 文件不完整，删除
      await targetFile.delete();
    }

    return null;
  }

  /// 手动触发模型下载，支持进度回调和断点续传
  static Future<void> downloadModel({
    void Function(double progress)? onProgress,
    void Function(String status)? onStatus,
  }) async {
    if (_isDownloading) {
      throw Exception('Model is already downloading');
    }

    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final tempPath = '$targetPath.tmp';
    final tempFile = File(tempPath);

    _isDownloading = true;
    _downloadProgress = 0.0;

    void reportStatus(String status) {
      _currentStatus = status;
      onStatus?.call(status);
    }

    try {
      // 删除旧的临时文件
      if (await tempFile.exists()) {
        await tempFile.delete();
      }

      Exception? lastError;
      final urls = await getModelUrls();
      for (int i = 0; i < urls.length; i++) {
        final url = urls[i];
        reportStatus('Trying mirror ${i + 1}/${urls.length}...');
        try {
          await _downloadWithResume(url, tempPath, (received, total) {
            _downloadProgress = total > 0 ? received / total : 0;
            onProgress?.call(_downloadProgress);
          });
          final downloaded = await File(tempPath).length();
          if (downloaded > expectedFileSize ~/ 2) {
            await File(tempPath).rename(targetPath);
            _cachedModelPath = targetPath;
            _downloadProgress = 1.0;
            reportStatus('Download complete');
            return;
          }
          await File(tempPath).delete();
        } catch (e) {
          lastError = e is Exception ? e : Exception(e.toString());
          reportStatus('Mirror ${i + 1} failed: $e');
          // 清理临时文件
          if (await tempFile.exists()) {
            await tempFile.delete();
          }
        }
      }
      throw lastError ?? Exception('All model download URLs failed');
    } finally {
      _isDownloading = false;
    }
  }

  /// 使用 HttpClient 下载，支持 Range 断点续传和实时进度
  static Future<void> _downloadWithResume(
    String url,
    String targetPath,
    void Function(int received, int total) onProgress,
  ) async {
    final client = HttpClient();
    final file = File(targetPath);
    int startByte = 0;
    if (await file.exists()) {
      startByte = await file.length();
    }

    IOSink? sink;
    try {
      final uri = Uri.parse(url);
      final request = await client.getUrl(uri);
      request.followRedirects = true;
      request.headers.set('User-Agent', 'Venera/1.0');
      request.headers.set('Accept', '*/*');
      request.headers.set('Connection', 'keep-alive');
      if (startByte > 0) {
        request.headers.set('Range', 'bytes=$startByte-');
      }

      final response = await request.close();
      if (response.statusCode != HttpStatus.ok &&
          response.statusCode != HttpStatus.partialContent) {
        throw Exception('HTTP ${response.statusCode}');
      }

      final contentLength = response.contentLength;
      int received = startByte;
      final total = contentLength > 0 ? contentLength + startByte : 0;

      sink = file.openWrite(
        mode: startByte > 0 ? FileMode.append : FileMode.write,
      );

      await for (final chunk in response) {
        sink.add(chunk);
        received += chunk.length;
        onProgress(received, total);
      }
      await sink.close();
    } catch (e) {
      sink?.close();
      rethrow;
    } finally {
      client.close();
    }
  }

  /// 模型是否已下载且完整
  static Future<bool> get isModelDownloaded async {
    // 用户自选的本地模型也算就绪
    final custom = await getCustomModelPath();
    if (custom != null) {
      final f = File(custom);
      if (await f.exists()) {
        final size = await f.length();
        if (size > 1024 * 1024) return true;
      }
    }
    if (_cachedModelPath != null) return true;
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final targetFile = File(targetPath);
    if (!await targetFile.exists()) return false;
    final size = await targetFile.length();
    return size > expectedFileSize ~/ 2;
  }

  /// 获取已下载模型大小（字节）
  static Future<int> getDownloadedSize() async {
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final targetFile = File(targetPath);
    if (!await targetFile.exists()) return 0;
    return await targetFile.length();
  }

  /// 获取下载进度（0.0 - 1.0）
  static double get downloadProgress => _downloadProgress;

  /// 是否正在下载
  static bool get isDownloading => _isDownloading;

  /// 当前状态文本
  static String? get currentStatus => _currentStatus;

  /// 清除模型文件
  static Future<void> clearModel() async {
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final tempPath = '$targetPath.tmp';
    final targetFile = File(targetPath);
    final tempFile = File(tempPath);
    if (await targetFile.exists()) {
      await targetFile.delete();
    }
    if (await tempFile.exists()) {
      await tempFile.delete();
    }
    _cachedModelPath = null;
  }
}

import 'dart:io';

import 'package:archive/archive.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:venera/foundation/log.dart';

/// 翻译模型管理器：参照 [ColorizationModelManager] 的下载框架。
///
/// 设计：检测 / OCR 类模型（气泡、文本检测、PP-OCR 检测·识别、日漫 OCR、韩文 OCR）
/// 已随 APK 打包（与上游 release 对齐，见 commit c8d0cc6），无需下载；
/// 仅**本地翻译模型 MarianMT (en_zh)** 因体积较大（~470MB fp32）改为按需下载。
///
/// 默认镜像指向 Venera-SSR 的 GitHub Release（`translation-models` tag），并附带 ghproxy
/// 镜像加速；模型文件由维护者在 Release 中托管。
class TranslationModelManager {
  static const _urlsKeyPrefix = 'translation_model_urls_';
  static const _releaseBase =
      'https://github.com/Kiastr/Venera-SSR/releases/download/translation-models';

  /// 单文件模型直接落到 [targetSubdir]/[fileName]；
  /// zip 模型下载后解压到 [targetSubdir]（[markerFile] 用于存在性校验）。
  /// 仅本地翻译模型（MarianMT en_zh）走按需下载；其余检测/OCR 模型已打包进 APK。
  static final List<TranslationModelEntry> models = [
    TranslationModelEntry(
      id: 'marianmt',
      label: '本地翻译 (MarianMT en_zh)',
      isZip: true,
      fileName: 'en_zh.zip',
      targetSubdir: 'translation_models/en_zh',
      markerFile: 'vocab.json',
      minSize: 300 * 1024 * 1024,
      defaultUrls: _mirrors('en_zh.zip'),
    ),
  ];

  static List<String> _mirrors(String file) => [
        'https://ghproxy.net/$_releaseBase/$file',
        'https://mirror.ghproxy.com/$_releaseBase/$file',
        _releaseBase,
      ];

  static TranslationModelEntry entryById(String id) =>
      models.firstWhere((e) => e.id == id);

  // ===== 镜像地址管理（按模型 id 分别持久化）=====

  static Future<List<String>> getUrls(String id) async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getStringList('$_urlsKeyPrefix$id');
    if (saved != null && saved.isNotEmpty) return List.from(saved);
    return List.from(entryById(id).defaultUrls);
  }

  static Future<void> addUrl(String id, String url) async {
    final urls = await getUrls(id);
    final trimmed = url.trim();
    if (trimmed.isEmpty || urls.contains(trimmed)) return;
    urls.add(trimmed);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('$_urlsKeyPrefix$id', urls);
  }

  static Future<void> removeUrlAt(String id, int index) async {
    final urls = await getUrls(id);
    if (index < 0 || index >= urls.length) return;
    urls.removeAt(index);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('$_urlsKeyPrefix$id', urls);
  }

  static Future<void> resetUrls(String id) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('$_urlsKeyPrefix$id',
        List.from(entryById(id).defaultUrls));
  }

  // ===== 下载状态 =====

  static bool _isDownloading = false;
  static double _progress = 0.0;
  static String? _status;
  static String? _activeId;

  static bool get isDownloading => _isDownloading;
  static double get progress => _progress;
  static String? get status => _status;
  static String? get activeId => _activeId;

  // ===== 路径 / 校验 =====

  static Future<String> _targetDirFor(TranslationModelEntry e) async {
    final base = await getApplicationSupportDirectory();
    return path.join(base.path, e.targetSubdir);
  }

  static Future<bool> isDownloaded(String id) async {
    final e = entryById(id);
    final dir = await _targetDirFor(e);
    if (e.isZip) {
      final marker = e.markerFile ?? e.fileName;
      return File(path.join(dir, marker)).exists();
    }
    final f = File(path.join(dir, e.fileName));
    if (!await f.exists()) return false;
    return await f.length() > e.minSize;
  }

  static Future<int> getDownloadedSize(String id) async {
    final e = entryById(id);
    final dir = Directory(await _targetDirFor(e));
    if (!await dir.exists()) return 0;
    int total = 0;
    try {
      await for (final entity in dir.list(recursive: true)) {
        if (entity is File) total += await entity.length();
      }
    } catch (_) {}
    return total;
  }

  static Future<void> deleteModel(String id) async {
    final e = entryById(id);
    final dir = Directory(await _targetDirFor(e));
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    }
    if (_activeId == id) {
      _isDownloading = false;
      _progress = 0;
      _status = null;
      _activeId = null;
    }
  }

  // ===== 下载主流程 =====

  static Future<void> download({
    required String id,
    void Function(double progress)? onProgress,
    void Function(String status)? onStatus,
  }) async {
    if (_isDownloading) throw Exception('A translation model is already downloading');
    final e = entryById(id);
    _isDownloading = true;
    _progress = 0.0;
    _activeId = id;
    final dir = await _targetDirFor(e);
    await Directory(dir).create(recursive: true);
    final tempPath = path.join(dir, '${e.fileName}.tmp');

    void report(String s) {
      _status = s;
      onStatus?.call(s);
    }

    try {
      final urls = await getUrls(id);
      Exception? lastError;
      for (int i = 0; i < urls.length; i++) {
        report('Trying mirror ${i + 1}/${urls.length}...');
        try {
          await _downloadWithResume(urls[i], tempPath, (received, total) {
            _progress = total > 0 ? received / total : 0;
            onProgress?.call(_progress);
          });
          final downloaded = await File(tempPath).length();
          if (downloaded > e.minSize) {
            if (e.isZip) {
              report('Extracting...');
              await _extractZip(tempPath, dir);
            } else {
              final finalPath = path.join(dir, e.fileName);
              await File(tempPath).rename(finalPath);
            }
            await File(tempPath).delete().catchError((_) {});
            _progress = 1.0;
            report('Download complete');
            return;
          }
          await File(tempPath).delete();
        } catch (err) {
          lastError = err is Exception ? err : Exception(err.toString());
          report('Mirror ${i + 1} failed: $err');
          await File(tempPath).delete().catchError((_) {});
        }
      }
      throw lastError ?? Exception('All translation model download URLs failed');
    } finally {
      _isDownloading = false;
      _activeId = null;
    }
  }

  /// 使用 HttpClient 下载，支持 Range 断点续传与实时进度（与 ColorizationModelManager 同构）
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

  static Future<void> _extractZip(String zipPath, String targetDir) async {
    final bytes = await File(zipPath).readAsBytes();
    final archive = ZipDecoder().decodeBytes(bytes);
    for (final file in archive) {
      final filename = file.name;
      if (file.isFile) {
        final data = file.content as List<int>;
        final outPath = path.join(targetDir, filename);
        final outFile = File(outPath);
        await outFile.parent.create(recursive: true);
        await outFile.writeAsBytes(data);
      }
    }
  }
}

class TranslationModelEntry {
  final String id;
  final String label;
  final bool isZip;
  final String fileName;
  final String targetSubdir;
  final String? markerFile;
  final int minSize;
  final List<String> defaultUrls;

  const TranslationModelEntry({
    required this.id,
    required this.label,
    required this.isZip,
    required this.fileName,
    required this.targetSubdir,
    this.markerFile,
    required this.minSize,
    required this.defaultUrls,
  });
}

import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

/// 图像上色参数
class ColorizationParams {
  final Uint8List imageBytes;
  final String modelPath;
  final double intensity;

  ColorizationParams({
    required this.imageBytes,
    required this.modelPath,
    this.intensity = 1.0,
  });
}

/// 用于 Isolate 的参数封装
class _IsolateParams {
  final Uint8List imageBytes;
  final String modelPath;
  final double intensity;

  _IsolateParams({
    required this.imageBytes,
    required this.modelPath,
    required this.intensity,
  });
}

/// 在 Isolate 中执行图像上色
///
/// 这是顶层函数，可以安全地被 compute() 调用。
Future<Uint8List?> colorizeImage(ColorizationParams params) async {
  return _colorizeImpl(_IsolateParams(
    imageBytes: params.imageBytes,
    modelPath: params.modelPath,
    intensity: params.intensity,
  ));
}

// ===== Sobel 边缘检测 =====

/// 使用 Sobel 算子在 256x256 灰度图上检测边缘，返回边缘强度图 (0-1)
List<double> _sobelEdgeMask(img.Image grayImage) {
  final w = grayImage.width;
  final h = grayImage.height;
  final mask = List<double>.filled(w * h, 0.0);

  // 先计算灰度
  final gray = List<double>.filled(w * h, 0.0);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final pixel = grayImage.getPixel(x, y);
      final r = pixel.r.toDouble();
      final g = pixel.g.toDouble();
      final b = pixel.b.toDouble();
      gray[y * w + x] = (r + g + g + g + b) / 6.0;
    }
  }

  // Sobel 核
  // Gx: -1 0 1   Gy: -1 -2 -1
  //     -2 0 2        0  0  0
  //     -1 0 1        1  2  1
  for (int y = 1; y < h - 1; y++) {
    for (int x = 1; x < w - 1; x++) {
      final tl = gray[(y - 1) * w + (x - 1)];
      final tr = gray[(y - 1) * w + (x + 1)];
      final ml = gray[y * w + (x - 1)];
      final mr = gray[y * w + (x + 1)];
      final bl = gray[(y + 1) * w + (x - 1)];
      final br = gray[(y + 1) * w + (x + 1)];
      final t = gray[(y - 1) * w + x];
      final b = gray[(y + 1) * w + x];

      final gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      final gy = -tl - 2 * t - tr + bl + 2 * b + br;

      final mag = math.sqrt(gx * gx + gy * gy);
      // 归一化到 0–1：300 以上视为强边缘
      mask[y * w + x] = (mag / 300.0).clamp(0.0, 1.0);
    }
  }

  return mask;
}

// ===== 从 OrtValue 中提取 Float32List =====

Float32List? _extractFloat32List(OrtValue? value) {
  if (value == null) return null;

  try {
    // OrtValue 对象通常会是一个包装对象，尝试多种方式访问数据
    // 方式1: 直接通过 value.value 访问
    final val = value.value;
    if (val is Float32List) return val;
    if (val is List<double>) return Float32List.fromList(val.cast<num>().map((e) => e.toDouble()).toList());
    if (val is List) return Float32List.fromList(val.cast<num>().map((e) => e.toDouble()).toList());
  } catch (_) {}

  try {
    // 方式2: 尝试访问 data 属性
    final data = (value as dynamic).data;
    if (data is Float32List) return data;
    if (data is List<double>) return Float32List.fromList(data);
    if (data is List) return Float32List.fromList(data.cast<num>().map((e) => e.toDouble()).toList());
  } catch (_) {}

  try {
    // 方式3: 尝试访问 tensorData 属性
    final tensorData = (value as dynamic).tensorData;
    if (tensorData is Float32List) return tensorData;
    if (tensorData is List<double>) return Float32List.fromList(tensorData);
  } catch (_) {}

  return null;
}

// ===== 实际的上色实现 =====
Future<Uint8List?> _colorizeImpl(_IsolateParams params) async {
  try {
    // 1. 解码输入图像
    final decoded = img.decodeImage(params.imageBytes);
    if (decoded == null) {
      debugPrint('Colorization: failed to decode image bytes');
      return null;
    }

    final originalWidth = decoded.width;
    final originalHeight = decoded.height;

    // 2. 缩放到 256x256（DeOldify 模型固定输入尺寸）
    final resized = img.copyResize(
      decoded,
      width: 256,
      height: 256,
      interpolation: img.Interpolation.linear,
    );

    // 3. 构造 float32 输入张量 [1, 3, 256, 256]，值范围 0-255
    final input = Float32List(1 * 3 * 256 * 256);
    int offset = 0;

    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
          final pixel = resized.getPixel(x, y);
          double value;
          if (c == 0) {
            value = pixel.r.toDouble();
          } else if (c == 1) {
            value = pixel.g.toDouble();
          } else {
            value = pixel.b.toDouble();
          }
          input[offset++] = value;
        }
      }
    }

    // 4. ONNX 模型推理
    final modelFile = File(params.modelPath);
    if (!modelFile.existsSync()) {
      debugPrint('Colorization: model file not found at ${params.modelPath}');
      return null;
    }

    OrtEnv.instance.init();

    final session = OrtSession.fromFile(
      modelFile,
      OrtSessionOptions(),
    );

    try {
      final runOptions = OrtRunOptions();
      final inputTensor = OrtValueTensor.createTensorWithDataList(
        input,
        [1, 3, 256, 256],
      );

      // 模型输入名称可能为 "input" 或其他名称，先尝试 "input"，
      // 失败时使用 session.inputNames 的第一个。
      Map<String, OrtValue> inputs;
      try {
        inputs = {'input': inputTensor};
        final outputs = session.run(runOptions, inputs);
        return _buildColorizedPng(
          outputs,
          resized,
          originalWidth,
          originalHeight,
          params.intensity,
        );
      } catch (e) {
        // 尝试使用模型实际的输入名称
        final inputNames = session.inputNames;
        if (inputNames.isEmpty) rethrow;
        final actualInputName = inputNames.first;
        debugPrint(
          'Colorization: input name "input" failed, retrying with "$actualInputName": $e',
        );
        inputs = {actualInputName: inputTensor};
        final outputs = session.run(runOptions, inputs);
        return _buildColorizedPng(
          outputs,
          resized,
          originalWidth,
          originalHeight,
          params.intensity,
        );
      }
    } finally {
      session.release();
    }
  } catch (e, s) {
    debugPrint('Colorization error: $e\n$s');
    return null;
  }
}

/// 根据模型输出构建上色后的 PNG 字节
Uint8List? _buildColorizedPng(
  dynamic outputs,
  img.Image resized,
  int originalWidth,
  int originalHeight,
  double intensity,
) {
  if (outputs is! Map<String, OrtValue> || outputs.isEmpty) {
    debugPrint('Colorization: empty model outputs');
    return null;
  }

  final outputValue = outputs.values.first;
  final outputFloats = _extractFloat32List(outputValue);
  if (outputFloats == null) {
    debugPrint('Colorization: failed to extract Float32List from output');
    return null;
  }

  // 6. 将模型输出 [1, 3, 256, 256] 解析为 RGB 图像
  final modelRgb = img.Image(width: 256, height: 256);
  for (int y = 0; y < 256; y++) {
    for (int x = 0; x < 256; x++) {
      final idx = y * 256 + x;
      final rIdx = 0 * 256 * 256 + idx;
      final gIdx = 1 * 256 * 256 + idx;
      final bIdx = 2 * 256 * 256 + idx;

      double r, g, b;
      if (rIdx < outputFloats.length &&
          gIdx < outputFloats.length &&
          bIdx < outputFloats.length) {
        r = outputFloats[rIdx].clamp(0.0, 255.0);
        g = outputFloats[gIdx].clamp(0.0, 255.0);
        b = outputFloats[bIdx].clamp(0.0, 255.0);
      } else {
        // 备用：NHWC 格式 [1, 256, 256, 3]
        final flatIdx = idx * 3;
        r = outputFloats[flatIdx].clamp(0.0, 255.0);
        g = outputFloats[flatIdx + 1].clamp(0.0, 255.0);
        b = outputFloats[flatIdx + 2].clamp(0.0, 255.0);
      }

      modelRgb.setPixelRgb(
        x,
        y,
        r.round().clamp(0, 255),
        g.round().clamp(0, 255),
        b.round().clamp(0, 255),
      );
    }
  }

  // ============== 以下为修正后的后处理（与 colorize.py 一致） ==============

  // 7. Sobel 边缘检测：识别漫画中的黑色线条区域
  final edgeMask = _sobelEdgeMask(resized);

  // 8. 亮度检测 + 边缘抑制：
  //    - 边缘区域（黑色线条）：抑制颜色，保留原图灰度
  //    - 非常亮的区域（白纸）：抑制颜色
  //    - 非常暗的区域（纯黑线条）：抑制颜色
  //    - 其他区域：保留模型输出的颜色
  final mixedRgb = img.Image(width: 256, height: 256);

  for (int y = 0; y < 256; y++) {
    for (int x = 0; x < 256; x++) {
      final origPixel = resized.getPixel(x, y);
      final origR = origPixel.r.toDouble();
      final origG = origPixel.g.toDouble();
      final origB = origPixel.b.toDouble();

      final origLuminance = (origR + origG * 2 + origB) / 4;

      final coloredPixel = modelRgb.getPixel(x, y);
      final coloredR = coloredPixel.r.toDouble();
      final coloredG = coloredPixel.g.toDouble();
      final coloredB = coloredPixel.b.toDouble();

      final edgeStrength = edgeMask[y * 256 + x];

      // 抑制权重 (0 = 完全保留颜色, 1 = 完全变回原图)
      double suppress = 0.0;

      // 8a. 边缘区域（漫画黑色线条）
      if (edgeStrength > 0.15) {
        suppress = math.min(1.0, edgeStrength * 2.5);
      }

      // 8b. 非常亮的区域（白纸背景）
      if (origLuminance > 235) {
        final brightFactor = ((origLuminance - 235) / 20.0).clamp(0.0, 1.0);
        suppress = math.max(suppress, brightFactor);
      }

      // 8c. 非常暗的区域（纯黑线条）
      if (origLuminance < 25) {
        final darkFactor = ((25 - origLuminance) / 25.0).clamp(0.0, 1.0);
        suppress = math.max(suppress, darkFactor);
      }

      // 混合：抑制区域回退到原图灰度
      final gray = origLuminance;
      double finalR = coloredR + (gray - coloredR) * suppress;
      double finalG = coloredG + (gray - coloredG) * suppress;
      double finalB = coloredB + (gray - coloredB) * suppress;

      // 8d. intensity 调整：控制整体上色强度
      if (intensity != 1.0) {
        finalR = gray + (finalR - gray) * intensity;
        finalG = gray + (finalG - gray) * intensity;
        finalB = gray + (finalB - gray) * intensity;
      }

      mixedRgb.setPixelRgb(
        x,
        y,
        finalR.round().clamp(0, 255),
        finalG.round().clamp(0, 255),
        finalB.round().clamp(0, 255),
      );
    }
  }

  // 9. 高斯模糊（半径 1）：柔化色彩边界，减少模型的高频噪点
  final blurred = img.gaussianBlur(mixedRgb, radius: 1);

  // 10. 缩放到原图尺寸
  final finalImage = img.copyResize(
    blurred,
    width: originalWidth,
    height: originalHeight,
    interpolation: img.Interpolation.linear,
  );

  // 11. 编码 PNG
  final pngBytes = img.encodePng(finalImage);
  return Uint8List.fromList(pngBytes);
}

/// 模型管理器：负责从网络下载模型到应用目录
///
/// 不使用 Flutter assets 打包模型（模型文件 243MB 会导致 Android APK 构建失败），
/// 改为在设置页由用户手动触发下载，避免首次启动即下载大文件。
class ColorizationModelManager {
  static const modelFileName = 'deoldify_artistic.onnx';
  static const expectedFileSize = 243 * 1024 * 1024; // ~243MB

  /// 多个镜像源，按稳定性排序
  static const modelDownloadUrls = [
    'https://mirror.ghproxy.com/https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
    'https://ghp.ci/https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
    'https://ghproxy.net/https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
    'https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
  ];

  static String? _cachedModelPath;
  static bool _isDownloading = false;
  static double _downloadProgress = 0.0;
  static String? _currentStatus;

  /// 获取模型文件路径。如果模型未下载，返回 null（不自动下载）
  static Future<String?> ensureModelAvailable() async {
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
      for (int i = 0; i < modelDownloadUrls.length; i++) {
        final url = modelDownloadUrls[i];
        reportStatus('Trying mirror ${i + 1}/${modelDownloadUrls.length}...');
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

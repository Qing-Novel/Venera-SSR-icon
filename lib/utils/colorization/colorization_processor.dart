import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:venera/foundation/log.dart';

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

// ===== OpenCV 风格的颜色空间转换 =====
//
// 参照 deoldify-onnx 官方实现（color/deoldify.py），后处理使用 LAB 颜色空间
// 合并原始 L 通道与模型输出的 A/B 通道。由于 Flutter/Dart 端无法直接引入
// Python 的 cv2，这里用 Dart 实现等价的 OpenCV RGB<->LAB 转换。

/// 参考白点 D65（OpenCV 使用值）
const double _labXn = 0.950456;
const double _labYn = 1.0;
const double _labZn = 1.088754;

/// 8-bit sRGB -> 线性 RGB [0, 1] 查找表，避免每像素重复 math.pow。
final Float64List _inverseGammaLut = _buildInverseGammaLut();

Float64List _buildInverseGammaLut() {
  final lut = Float64List(256);
  for (int i = 0; i < 256; i++) {
    final c = i.toDouble();
    if (c <= 0.04045 * 255.0) {
      lut[i] = c / (12.92 * 255.0);
    } else {
      lut[i] = math.pow((c / 255.0 + 0.055) / 1.055, 2.4).toDouble();
    }
  }
  return lut;
}

/// 将线性 RGB [0, 1] 做 sRGB gamma 校正为 8-bit。
double _srgbGamma(double c) {
  if (c <= 0.0031308) {
    return c * 12.92 * 255.0;
  }
  return ((math.pow(c, 1.0 / 2.4).toDouble() * 1.055) - 0.055) * 255.0;
}

/// OpenCV 风格的 RGB -> LAB。
///
/// 输入 r, g, b 为 [0, 255]，输出 (L, A, B) 为 OpenCV uint8 编码：
/// L in [0, 255]，A/B in [0, 255]（中性灰为 128）。
(double, double, double) rgbToLab(double r, double g, double b) {
  final ri = r.round().clamp(0, 255);
  final gi = g.round().clamp(0, 255);
  final bi = b.round().clamp(0, 255);
  final lr = _inverseGammaLut[ri];
  final lg = _inverseGammaLut[gi];
  final lb = _inverseGammaLut[bi];

  double x = lr * 0.412453 + lg * 0.357580 + lb * 0.180423;
  double y = lr * 0.212671 + lg * 0.715160 + lb * 0.072169;
  double z = lr * 0.019334 + lg * 0.119193 + lb * 0.950227;

  x /= _labXn;
  y /= _labYn;
  z /= _labZn;

  double f(double t) {
    if (t > 0.008856) {
      return math.pow(t, 1.0 / 3.0).toDouble();
    }
    return 7.787 * t + 16.0 / 116.0;
  }

  final fy = f(y);
  final l = 116.0 * fy - 16.0;
  final a = 500.0 * (f(x) - fy);
  final bb = 200.0 * (fy - f(z));

  // 映射到 OpenCV uint8 范围
  return (
    l.clamp(0.0, 100.0) * 2.55,
    (a + 128.0).clamp(0.0, 255.0),
    (bb + 128.0).clamp(0.0, 255.0),
  );
}

/// OpenCV 风格的 LAB -> RGB。
///
/// 输入 (L, A, B) 为 OpenCV uint8 编码，输出 (r, g, b) 为 [0, 255]。
(double, double, double) labToRgb(double l, double a, double b) {
  // 从 OpenCV uint8 范围还原到 CIE LAB 自然范围
  final ll = l / 2.55;
  final aa = a - 128.0;
  final bb = b - 128.0;

  final fy = (ll + 16.0) / 116.0;
  final fx = aa / 500.0 + fy;
  final fz = fy - bb / 200.0;

  double fInv(double t) {
    final t3 = t * t * t;
    if (t3 > 0.008856) {
      return t3;
    }
    return (t - 16.0 / 116.0) / 7.787;
  }

  double x = fInv(fx) * _labXn;
  double y = fInv(fy) * _labYn;
  double z = fInv(fz) * _labZn;

  // XYZ -> linear RGB
  final lr = x * 3.240481 + y * -1.537151 + z * -0.498536;
  final lg = x * -0.969254 + y * 1.875990 + z * 0.041556;
  final lb = x * 0.055646 + y * -0.203994 + z * 1.057069;

  return (
    _srgbGamma(lr).clamp(0.0, 255.0),
    _srgbGamma(lg).clamp(0.0, 255.0),
    _srgbGamma(lb).clamp(0.0, 255.0),
  );
}

/// 生成与 OpenCV cv2.GaussianBlur(ksize=(13,13), sigma=0) 等价的一维核。
Float64List gaussianKernel13() {
  const int k = 13;
  const int center = k ~/ 2;
  // OpenCV 在 sigma=0 时按 ksize 计算 sigma
  final sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8;
  final denom = 2.0 * sigma * sigma;
  final kernel = Float64List(k);
  double sum = 0.0;
  for (int i = 0; i < k; i++) {
    final v = math.exp(-((i - center) * (i - center)) / denom);
    kernel[i] = v;
    sum += v;
  }
  for (int i = 0; i < k; i++) {
    kernel[i] /= sum;
  }
  return kernel;
}

/// OpenCV BORDER_REFLECT_101 风格的边界插值。
///
/// 例如长度为 n 的序列：... 2 1 | 0 1 2 ... n-3 n-2 | n-1 n-2 n-3 ...
int _reflect101(int p, int len) {
  if (len <= 1) return 0;
  if (p < 0) {
    p = -p;
  } else if (p >= len) {
    p = len - 1 - (p - (len - 1));
  } else {
    return p;
  }
  // 极少数情况下反射后仍越界，递归处理
  return _reflect101(p, len);
}

/// 13x13 可分离高斯模糊（等价于 OpenCV (13,13), sigmaX=0）。
///
/// 对 [src] 的 RGB 三通道分别做水平+垂直卷积，结果写入 [dst]。
/// 两个图像必须同宽高、同通道数，且只处理 RGB 通道（若有 Alpha 则原样复制）。
void _gaussianBlur13({
  required Uint8List src,
  required Uint8List dst,
  required int width,
  required int height,
  required int channels,
}) {
  final kernel = gaussianKernel13();
  const int radius = 6;

  // 水平方向：src -> tmp
  final tmp = Uint8List(width * height * channels);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final base = (y * width + x) * channels;
      for (int c = 0; c < 3; c++) {
        double acc = 0.0;
        for (int k = 0; k < kernel.length; k++) {
          final sx = _reflect101(x + k - radius, width);
          acc += src[(y * width + sx) * channels + c] * kernel[k];
        }
        tmp[base + c] = acc.round().clamp(0, 255);
      }
      if (channels == 4) {
        tmp[base + 3] = src[base + 3];
      }
    }
  }

  // 垂直方向：tmp -> dst
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final base = (y * width + x) * channels;
      for (int c = 0; c < 3; c++) {
        double acc = 0.0;
        for (int k = 0; k < kernel.length; k++) {
          final sy = _reflect101(y + k - radius, height);
          acc += tmp[(sy * width + x) * channels + c] * kernel[k];
        }
        dst[base + c] = acc.round().clamp(0, 255);
      }
      if (channels == 4) {
        dst[base + 3] = src[base + 3];
      }
    }
  }
}

/// 对 [src] 图像应用 13x13 高斯模糊并返回新图像。
img.Image gaussianBlur13Image(img.Image src) {
  final srcBytes = src.getBytes();
  final dstBytes = Uint8List(srcBytes.length);
  _gaussianBlur13(
    src: srcBytes,
    dst: dstBytes,
    width: src.width,
    height: src.height,
    channels: src.numChannels,
  );
  return img.Image.fromBytes(
    width: src.width,
    height: src.height,
    bytes: dstBytes.buffer,
    numChannels: src.numChannels,
  );
}

// ===== 实际的上色实现 =====
Future<Uint8List?> _colorizeImpl(_IsolateParams params) async {
  try {
    // 1. 解码输入图像
    final decoded = img.decodeImage(params.imageBytes);
    if (decoded == null) {
      Log.error('Colorization', 'failed to decode image bytes');
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

    // 3. 构造 float32 输入张量 [1, 3, 256, 256]，值范围 0-255，使用灰度亮度。
    //
    // 官方实现：
    //   image = cv2.cvtColor(image, COLOR_BGR2GRAY)
    //   image = cv2.cvtColor(image, COLOR_GRAY2RGB)
    // 即三通道均为灰度亮度，不进行 /255 归一化。
    final input = Float32List(1 * 3 * 256 * 256);
    int offset = 0;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
          final pixel = resized.getPixel(x, y);
          // OpenCV BGR2GRAY 的等效亮度系数（在 RGB 空间）
          final luminance = 0.114 * pixel.r + 0.587 * pixel.g + 0.299 * pixel.b;
          input[offset++] = luminance;
        }
      }
    }

    // 4. ONNX 模型推理
    final modelFile = File(params.modelPath);
    if (!modelFile.existsSync()) {
      Log.error('Colorization', 'model file not found at ${params.modelPath}');
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
          decoded,
          originalWidth,
          originalHeight,
          params.intensity,
        );
      } catch (e) {
        // 尝试使用模型实际的输入名称
        final inputNames = session.inputNames;
        if (inputNames.isEmpty) rethrow;
        final actualInputName = inputNames.first;
        Log.error('Colorization', 'input name "input" failed, retrying with "$actualInputName": $e');
        inputs = {actualInputName: inputTensor};
        final outputs = session.run(runOptions, inputs);
        return _buildColorizedPng(
          outputs,
          decoded,
          originalWidth,
          originalHeight,
          params.intensity,
        );
      }
    } finally {
      session.release();
    }
  } catch (e, s) {
    Log.error('Colorization', 'Colorization error: $e\n$s');
    return null;
  }
}

/// 根据模型输出构建上色后的 PNG 字节
Uint8List? _buildColorizedPng(
  dynamic outputs,
  img.Image original,
  int originalWidth,
  int originalHeight,
  double intensity,
) {
  if (outputs is! Map<String, OrtValue> || outputs.isEmpty) {
    Log.error('Colorization', 'empty model outputs');
    return null;
  }

  final outputValue = outputs.values.first;
  final outputFloats = _extractFloat32List(outputValue);
  if (outputFloats == null) {
    Log.error('Colorization', 'failed to extract Float32List from output');
    return null;
  }

  // 5. 将模型输出 [1, 3, 256, 256] 解析为 RGB 图像。
  //
  // 官方实现把模型输出当作 BGR，再做 COLOR_BGR2RGB。
  // 因此在 Dart 端直接把通道 0 当 B、通道 2 当 R 写入 RGB 图像。
  // 同时按官方做法 clip 到 [0, 255] 并转为 uint8。
  final modelRgb = img.Image(width: 256, height: 256);
  for (int y = 0; y < 256; y++) {
    for (int x = 0; x < 256; x++) {
      final idx = y * 256 + x;
      final bIdx = 0 * 256 * 256 + idx;
      final gIdx = 1 * 256 * 256 + idx;
      final rIdx = 2 * 256 * 256 + idx;

      double r, g, b;
      if (rIdx < outputFloats.length &&
          gIdx < outputFloats.length &&
          bIdx < outputFloats.length) {
        r = outputFloats[rIdx].clamp(0.0, 255.0);
        g = outputFloats[gIdx].clamp(0.0, 255.0);
        b = outputFloats[bIdx].clamp(0.0, 255.0);
      } else {
        // 备用：NHWC 格式 [1, 256, 256, 3]，同样按 BGR 解释
        final flatIdx = idx * 3;
        b = outputFloats[flatIdx].clamp(0.0, 255.0);
        g = outputFloats[flatIdx + 1].clamp(0.0, 255.0);
        r = outputFloats[flatIdx + 2].clamp(0.0, 255.0);
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

  // 6. 缩放到原图尺寸（官方先 resize 再 blur）
  final scaled = img.copyResize(
    modelRgb,
    width: originalWidth,
    height: originalHeight,
    interpolation: img.Interpolation.linear,
  );

  // 7. 高斯模糊 (13, 13)，平滑模型输出的色块。
  //
  // 官方实现：
  //   colorized = cv2.GaussianBlur(colorized, (13, 13), 0)
  final blurred = gaussianBlur13Image(scaled);

  // 8. LAB 合并：原始图像的 L + 模糊后模型输出的 A/B。
  //
  // 官方实现：
  //   targetL = cv2.cvtColor(image, COLOR_BGR2LAB)[:, :, 0]
  //   colorizedLAB = cv2.cvtColor(colorized, COLOR_BGR2LAB)
  //   colorized = cv2.merge((targetL, colorizedLAB[:, :, 1:]))
  //   colorized = cv2.cvtColor(colorized, COLOR_LAB2BGR)
  final result = img.Image(width: originalWidth, height: originalHeight);
  for (int y = 0; y < originalHeight; y++) {
    for (int x = 0; x < originalWidth; x++) {
      final origPixel = original.getPixel(x, y);
      final (origL, _, _) = rgbToLab(
        origPixel.r.toDouble(),
        origPixel.g.toDouble(),
        origPixel.b.toDouble(),
      );

      final blurredPixel = blurred.getPixel(x, y);
      final (_, modelA, modelB) = rgbToLab(
        blurredPixel.r.toDouble(),
        blurredPixel.g.toDouble(),
        blurredPixel.b.toDouble(),
      );

      // intensity：在 AB 通道上围绕中性灰 128 做缩放，保留 L 不变
      final adjustedA = 128.0 + (modelA - 128.0) * intensity;
      final adjustedB = 128.0 + (modelB - 128.0) * intensity;

      final (r, g, b) = labToRgb(origL, adjustedA, adjustedB);
      result.setPixelRgb(
        x,
        y,
        r.round().clamp(0, 255),
        g.round().clamp(0, 255),
        b.round().clamp(0, 255),
      );
    }
  }

  // 9. 编码 PNG
  final pngBytes = img.encodePng(result);
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

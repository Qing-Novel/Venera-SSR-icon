import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
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

/// 实际的上色实现
Future<Uint8List?> _colorizeImpl(_IsolateParams params) async {
  try {
    // 1. 解码输入图像
    final decoded = img.decodeImage(params.imageBytes);
    if (decoded == null) {
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

    // 按通道顺序（R, G, B）填入
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
          final pixel = resized.getPixel(x, y);
          double value;
          if (c == 0) {
            value = img.getRed(pixel).toDouble();
          } else if (c == 1) {
            value = img.getGreen(pixel).toDouble();
          } else {
            value = img.getBlue(pixel).toDouble();
          }
          input[offset++] = value;
        }
      }
    }

    // 4. ONNX 模型推理
    final modelFile = File(params.modelPath);
    if (!modelFile.existsSync()) {
      return null;
    }

    // 确保环境已初始化
    OrtEnv.instance.init();

    final session = OrtSession.fromFile(
      params.modelPath,
      OrtSessionOptions(),
    );

    try {
      final runOptions = OrtRunOptions();

      // 构造 OrtValueTensor 输入
      final inputTensor = OrtValueTensor.createTensorWithDataList(
        input,
        [1, 3, 256, 256],
      );

      final inputs = {'input': inputTensor};

      // 执行推理
      final outputs = session.run(runOptions, inputs);

      // 5. 处理输出
      if (outputs.isEmpty) {
        return null;
      }

      // 从输出张量提取 Float32List 数据
      final outputValue = outputs.values.first;
      Float32List outputFloats;

      if (outputValue is Float32List) {
        outputFloats = outputValue;
      } else if (outputValue is List) {
        outputFloats = Float32List.fromList(
          outputValue.cast<num>().map((e) => e.toDouble()).toList(),
        );
      } else {
        // 尝试访问 data 属性（一些版本的 OrtValueTensor 有 data 属性）
        try {
          final data = (outputValue as dynamic).data;
          if (data is Float32List) {
            outputFloats = data;
          } else if (data is List) {
            outputFloats = Float32List.fromList(
              data.cast<num>().map((e) => e.toDouble()).toList(),
            );
          } else {
            return null;
          }
        } catch (e) {
          return null;
        }
      }

      // 6. 将输出 [1, 3, 256, 256] float32 转 uint8 RGB 图像
      // 裁剪到 [0, 255] 范围
      final colored256 = img.Image(width: 256, height: 256);
      for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
          final idx = y * 256 + x;
          // 输出格式: [1, 3, 256, 256] - 通道优先
          final rIdx = 0 * 256 * 256 + idx;
          final gIdx = 1 * 256 * 256 + idx;
          final bIdx = 2 * 256 * 256 + idx;

          double r;
          double g;
          double b;

          // 确保索引不越界
          if (rIdx < outputFloats.length &&
              gIdx < outputFloats.length &&
              bIdx < outputFloats.length) {
            r = outputFloats[rIdx];
            g = outputFloats[gIdx];
            b = outputFloats[bIdx];
          } else {
            // 备用: 可能输出是 [1, 256, 256, 3] 格式
            final flatIdx = idx * 3;
            if (flatIdx + 2 < outputFloats.length) {
              r = outputFloats[flatIdx];
              g = outputFloats[flatIdx + 1];
              b = outputFloats[flatIdx + 2];
            } else {
              r = g = b = 0.0;
            }
          }

          // 裁剪到 [0, 255] 范围
          r = r.clamp(0.0, 255.0);
          g = g.clamp(0.0, 255.0);
          b = b.clamp(0.0, 255.0);

          // 应用 intensity 调整（相对于原图灰度做混合）
          if (params.intensity != 1.0) {
            final origPixel = resized.getPixel(x, y);
            final gray = (img.getRed(origPixel) +
                    img.getGreen(origPixel) +
                    img.getBlue(origPixel)) /
                3.0;
            r = gray + (r - gray) * params.intensity;
            g = gray + (g - gray) * params.intensity;
            b = gray + (b - gray) * params.intensity;
          }

          colored256.setPixelRgb(
            x,
            y,
            r.round().clamp(0, 255),
            g.round().clamp(0, 255),
            b.round().clamp(0, 255),
          );
        }
      }

      // 7. 缩放到原图尺寸
      final finalImage = img.copyResize(
        colored256,
        width: originalWidth,
        height: originalHeight,
        interpolation: img.Interpolation.linear,
      );

      // 8. 编码 PNG
      final pngBytes = img.encodePng(finalImage);
      return Uint8List.fromList(pngBytes);
    } finally {
      session.release();
    }
  } catch (e) {
    return null;
  }
}

/// 模型管理器：负责从 assets 提取模型到应用目录
class ColorizationModelManager {
  static const assetPath = 'assets/deoldify_artistic.onnx';
  static String? _cachedModelPath;

  /// 获取模型文件路径（从 asset 复制到临时目录）
  static Future<String> ensureModelAvailable() async {
    if (_cachedModelPath != null) return _cachedModelPath!;

    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, 'deoldify_artistic.onnx');
    final targetFile = File(targetPath);

    // 如果文件已存在就不再复制
    if (!await targetFile.exists()) {
      final byteData = await rootBundle.load(assetPath);
      final bytes = byteData.buffer
          .asUint8List(byteData.offsetInBytes, byteData.lengthInBytes);
      await targetFile.writeAsBytes(bytes, flush: true);
    }

    _cachedModelPath = targetPath;
    return targetPath;
  }

  /// 清除模型文件
  static Future<void> clearModel() async {
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, 'deoldify_artistic.onnx');
    final targetFile = File(targetPath);
    if (await targetFile.exists()) {
      await targetFile.delete();
    }
    _cachedModelPath = null;
  }
}

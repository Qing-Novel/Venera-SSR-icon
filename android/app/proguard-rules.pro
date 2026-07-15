# ============================================================
# Venera 图像上色（Colorization）原生依赖混淆保留规则
# release 构建 minifyEnabled=true + shrinkResources=true，
# 必须保留 ONNX Runtime / OpenCV 的 Java 类与 JNI 入口，否则运行期
# 会 NoClassDefFoundError / UnsatisfiedLinkError。
# ============================================================

# ONNX Runtime Android
-keep class ai.onnxruntime.** { *; }
-keep interface ai.onnxruntime.** { *; }
-keep class ai.onnxruntime.providers.** { *; }
-dontwarn ai.onnxruntime.**

# OpenCV (Maven AAR)
-keep class org.opencv.** { *; }
-keep interface org.opencv.** { *; }
-dontwarn org.opencv.**

# 上色插件与方法通道（防止 R8 误删未被直接引用的入口）
-keep class com.github.wgh136.venera.colorize.** { *; }

# 通用：MethodChannel 回调与 Flutter 嵌入层
-keep class io.flutter.** { *; }
-keep class * extends io.flutter.plugin.common.MethodCallHandler { *; }

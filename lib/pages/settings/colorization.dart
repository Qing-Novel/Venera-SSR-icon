part of 'settings_page.dart';

/// 图像上色设置页
///
/// 参考 Anime4K 设置页的模式。提供开关、强度调节和模型管理。
class ColorizationSettings extends StatefulWidget {
  const ColorizationSettings({super.key});

  @override
  State<ColorizationSettings> createState() => _ColorizationSettingsState();
}

class _ColorizationSettingsState extends State<ColorizationSettings> {
  bool _isModelDownloaded = false;
  bool _isDownloading = false;
  double _downloadProgress = 0.0;
  String _status = '';
  String? _customModelPath;
  List<String> _modelUrls = [];
  bool _usingCustom = false;

  @override
  void initState() {
    super.initState();
    _refreshModelStatus();
  }

  Future<void> _refreshModelStatus() async {
    final custom = await ColorizationModelManager.getCustomModelPath();
    final urls = await ColorizationModelManager.getModelUrls();
    final downloaded = await ColorizationModelManager.isModelDownloaded;
    if (mounted) {
      setState(() {
        _customModelPath = custom;
        _modelUrls = urls;
        _isModelDownloaded = downloaded;
        _usingCustom = custom != null;
      });
    }
  }

  Future<void> _downloadModel() async {
    if (_isDownloading) return;
    setState(() {
      _isDownloading = true;
      _downloadProgress = 0.0;
      _status = 'Preparing...';
    });

    try {
      await ColorizationModelManager.downloadModel(
        onProgress: (progress) {
          if (mounted) {
            setState(() {
              _downloadProgress = progress;
              _status = 'Downloading ${(progress * 100).toStringAsFixed(1)}%';
            });
          }
        },
        onStatus: (status) {
          if (mounted) {
            setState(() {
              _status = status;
            });
          }
        },
      );
      if (mounted) {
        context.showMessage(message: "Model downloaded".tl);
      }
    } catch (e) {
      if (mounted) {
        context.showMessage(message: "Download failed: $e".tl);
      }
    } finally {
      _isDownloading = false;
      // 主动刷新服务的模型路径缓存，使后续上色处理无需重启即可生效
      await ColorizationService.instance.checkModelAvailable();
      await _refreshModelStatus();
      if (mounted) {
        setState(() {
          if (_isModelDownloaded) {
            _status = 'Ready';
          } else {
            _status = '';
          }
        });
      }
    }
  }

  Future<void> _deleteModel() async {
    await ColorizationModelManager.clearModel();
    await ColorizationService.instance.clearCache();
    // 让服务感知模型已删除（重置 _modelPath，校验文件不存在）
    await ColorizationService.instance.checkModelAvailable();
    await _refreshModelStatus();
    if (mounted) {
      context.showMessage(message: "Model deleted".tl);
      setState(() {
        _status = '';
        _downloadProgress = 0.0;
      });
    }
  }

  /// 选择本地 .onnx 模型文件（优先级高于内置下载模型）
  Future<void> _pickLocalModel() async {
    try {
      final xFile = await file_selector.openFile(
        acceptedTypeGroups: <file_selector.XTypeGroup>[
          file_selector.XTypeGroup(
            label: 'ONNX Model',
            extensions: ['onnx'],
          ),
        ],
      );
      if (xFile == null) return;
      if (!xFile.name.toLowerCase().endsWith('.onnx')) {
        if (mounted) context.showMessage(message: "Please select a .onnx file".tl);
        return;
      }
      // 读取字节并物化到应用私有目录（Android 上为 content URI，需落盘）
      final bytes = await xFile.readAsBytes();
      await ColorizationModelManager.importCustomModel(bytes);
      // 让服务立即感知新路径，无需重启
      await ColorizationService.instance.checkModelAvailable();
      await _refreshModelStatus();
      if (mounted) context.showMessage(message: "Custom model selected".tl);
    } catch (e) {
      if (mounted) context.showMessage(message: "Failed to pick file: $e".tl);
    }
  }

  /// 清除自选模型，回退到内置下载模型
  Future<void> _clearCustomModel() async {
    await ColorizationModelManager.setCustomModelPath(null);
    await ColorizationService.instance.checkModelAvailable();
    await _refreshModelStatus();
    if (mounted) context.showMessage(message: "Reverted to built-in model".tl);
  }

  /// 添加一个自定义镜像 URL
  Future<void> _addMirrorUrl() async {
    await showInputDialog(
      context: context,
      title: "Add Mirror URL".tl,
      hintText: "https://.../deoldify.onnx",
      confirmText: "Add".tl,
      onConfirm: (url) async {
        await ColorizationModelManager.addModelUrl(url);
        await _refreshModelStatus();
        return null as Object?;
      },
    );
  }

  /// 删除指定下标的镜像 URL
  Future<void> _removeMirrorUrl(int index) async {
    await ColorizationModelManager.removeModelUrlAt(index);
    await _refreshModelStatus();
  }

  @override
  Widget build(BuildContext context) {
    return SmoothCustomScrollView(
      slivers: [
        SliverAppbar(title: Text("Colorization".tl)),
        _SwitchSetting(
          title: "Enable Image Colorization".tl,
          subtitle:
              _isModelDownloaded
                  ? "Model is ready".tl
                  : "Download model below to enable".tl,
          settingKey: "enableColorization",
        ).toSliver(),
        _SliderSetting(
          title: "Colorization Intensity".tl,
          settingsIndex: "colorizationIntensity",
          min: 0.3,
          max: 1.2,
          interval: 0.05,
        ).toSliver(),
        // 模型管理区域
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Text(
              "Model Management".tl,
              style: TextStyle(
                color: context.colorScheme.primary,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),
        SliverToBoxAdapter(
          child: Card(
            margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "DeOldify Artistic ONNX".tl,
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _isModelDownloaded
                        ? "Model downloaded".tl
                        : "Model not downloaded (~243MB)".tl,
                    style: TextStyle(
                      color: context.colorScheme.onSurfaceVariant,
                      fontSize: 12,
                    ),
                  ),
                  if (_status.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Text(
                      _status,
                      style: TextStyle(
                        color: context.colorScheme.primary,
                        fontSize: 12,
                      ),
                    ),
                  ],
                  if (_isDownloading) ...[
                    const SizedBox(height: 8),
                    LinearProgressIndicator(value: _downloadProgress),
                  ],
                  const SizedBox(height: 12),
                  if (!_usingCustom)
                    Row(
                      children: [
                      if (!_isModelDownloaded)
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed:
                                _isDownloading ? null : _downloadModel,
                            icon:
                                _isDownloading
                                    ? const SizedBox(
                                      width: 16,
                                      height: 16,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                      ),
                                    )
                                    : const Icon(Icons.download),
                            label: Text(
                              _isDownloading ? "Downloading...".tl : "Download Model".tl,
                            ),
                          ),
                        ),
                      if (_isModelDownloaded) ...[
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _deleteModel,
                            icon: const Icon(Icons.delete_outline),
                            label: Text("Delete Model".tl),
                          ),
                        ),
                      ],
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
        // 自选本地模型文件
        SliverToBoxAdapter(
          child: Card(
            margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Custom Model File".tl,
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _usingCustom
                        ? "Using: ${path.basename(_customModelPath ?? '')}".tl
                        : "Select a local .onnx model to override the built-in one"
                            .tl,
                    style: TextStyle(
                      color: context.colorScheme.onSurfaceVariant,
                      fontSize: 12,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _pickLocalModel,
                          icon: const Icon(Icons.folder_open),
                          label: Text("Select Model File".tl),
                        ),
                      ),
                      if (_usingCustom) ...[
                        const SizedBox(width: 8),
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _clearCustomModel,
                            icon: const Icon(Icons.restore),
                            label: Text("Use Built-in".tl),
                          ),
                        ),
                      ],
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
        // 镜像 URL 管理
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Text(
              "Download Mirrors".tl,
              style: TextStyle(
                color: context.colorScheme.primary,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),
        ..._modelUrls.asMap().entries.map(
              (e) => _MirrorUrlTile(
                index: e.key,
                url: e.value,
                onDelete: _removeMirrorUrl,
              ),
            ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _addMirrorUrl,
                    icon: const Icon(Icons.add),
                    label: Text("Add Mirror URL".tl),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () async {
                      await ColorizationModelManager.resetModelUrls();
                      await _refreshModelStatus();
                    },
                    icon: const Icon(Icons.restart_alt),
                    label: Text("Reset".tl),
                  ),
                ),
              ],
            ),
          ),
        ),
        ListTile(
          title: Text("Clear Colorization Cache".tl),
          trailing: const Icon(Icons.delete_sweep),
          onTap: () async {
            await ColorizationService.instance.clearCache();
            if (mounted) {
              context.showMessage(message: "Colorization cache cleared".tl);
            }
          },
        ).toSliver(),
      ],
    );
  }
}

/// 镜像 URL 列表项
class _MirrorUrlTile extends StatelessWidget {
  final int index;
  final String url;
  final void Function(int) onDelete;

  const _MirrorUrlTile({
    required this.index,
    required this.url,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    return SliverToBoxAdapter(
      child: ListTile(
        dense: true,
        leading: Text('${index + 1}'),
        title: Text(
          url,
          style: const TextStyle(fontSize: 12),
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
        ),
        trailing: IconButton(
          icon: const Icon(Icons.delete_outline, size: 20),
          onPressed: () => onDelete(index),
        ),
      ),
    );
  }
}

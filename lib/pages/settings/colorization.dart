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

  @override
  void initState() {
    super.initState();
    _refreshModelStatus();
  }

  Future<void> _refreshModelStatus() async {
    final downloaded = await ColorizationModelManager.isModelDownloaded;
    if (mounted) {
      setState(() {
        _isModelDownloaded = downloaded;
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
    await _refreshModelStatus();
    if (mounted) {
      context.showMessage(message: "Model deleted".tl);
      setState(() {
        _status = '';
        _downloadProgress = 0.0;
      });
    }
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

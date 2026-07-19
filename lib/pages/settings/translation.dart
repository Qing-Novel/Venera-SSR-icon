part of 'settings_page.dart';

/// 漫画翻译设置页
///
/// 提供「下载后自动翻译」（批量后台）与「阅读时实时翻译」开关、目标语言、
/// 强制 OCR、**远程 LLM 配置**以及缓存管理。
///
/// 前置条件（与上游 jedzqer 一致）：翻译依赖用户在此配置的远程 LLM，
/// 以及端上检测 / OCR 模型（仓库内不含模型）。未配置时翻译会静默跳过。
class TranslationSettings extends StatefulWidget {
  const TranslationSettings({super.key});

  @override
  State<TranslationSettings> createState() => _TranslationSettingsState();
}

class _TranslationSettingsState extends State<TranslationSettings> {
  int _cacheSize = 0;
  bool _loadingCacheSize = true;

  // 远程 LLM 配置
  final _apiUrlController = TextEditingController();
  final _apiKeyController = TextEditingController();
  final _modelNameController = TextEditingController();
  String _apiFormat = 'openai_compatible';
  bool _llmConfigured = false;
  bool _loadingConfig = true;
  bool _saving = false;
  bool _useLocalTranslation = false;
  String _localModelDir = '';
  bool _localModelAvailable = false;

  /// API 格式选项：界面显示 -> pref 值（传给原生 ApiFormat.fromPref）
  static const Map<String, String> _apiFormatOptions = {
    "OpenAI Compatible": "openai_compatible",
    "OpenAI Responses": "openai_responses",
    "Gemini": "gemini",
    "谷歌公共翻译(免Key)": "google_public",
  };

  /// 语言选项：label（界面显示） -> pref 值（传给原生 TranslationLanguage.fromPref）
  static const Map<String, String> _languageOptions = {
    "日语 → 中文": "ja_to_zh",
    "英语 → 中文": "en_to_zh",
    "韩语 → 中文": "ko_to_zh",
    "简体中文 → 目标语": "zh_hans_to_target",
    "繁体中文 → 目标语": "zh_hant_to_target",
    "中英混合 → 中文": "chn_eng_to_zh",
    "法语 → 中文": "fr_to_zh",
    "西班牙语 → 中文": "es_to_zh",
    "葡萄牙语 → 中文": "pt_to_zh",
    "德语 → 中文": "de_to_zh",
    "意大利语 → 中文": "it_to_zh",
    "俄语 → 中文": "ru_to_zh",
  };

  @override
  void initState() {
    super.initState();
    _refreshCacheSize();
    _loadLlmConfig();
    _loadLocalConfig();
  }

  @override
  void dispose() {
    _apiUrlController.dispose();
    _apiKeyController.dispose();
    _modelNameController.dispose();
    super.dispose();
  }

  Future<void> _refreshCacheSize() async {
    final size = await TranslationService.instance.getCacheSize();
    if (mounted) {
      setState(() {
        _cacheSize = size;
        _loadingCacheSize = false;
      });
    }
  }

  Future<void> _loadLlmConfig() async {
    final cfg = await TranslationService.instance.getLlmConfig();
    if (mounted) {
      setState(() {
        if (cfg != null) {
          _apiUrlController.text = (cfg['apiUrl'] as String?) ?? '';
          _apiKeyController.text = (cfg['apiKey'] as String?) ?? '';
          _modelNameController.text = (cfg['modelName'] as String?) ?? '';
          _apiFormat = (cfg['apiFormat'] as String?) ?? 'openai_compatible';
          _llmConfigured = cfg['configured'] == true;
        }
        _loadingConfig = false;
      });
    }
  }

  Future<void> _loadLocalConfig() async {
    final localCfg = await TranslationService.instance.getLocalTranslationConfig();
    if (mounted) {
      setState(() {
        _useLocalTranslation = localCfg['useLocalTranslation'] as bool? ?? false;
        _localModelDir = localCfg['modelDir'] as String? ?? '';
        _localModelAvailable = localCfg['modelAvailable'] as bool? ?? false;
      });
    }
  }

  Future<void> _saveLlmConfig() async {
    setState(() => _saving = true);
    final ok = await TranslationService.instance.setLlmConfig(
      apiUrl: _apiUrlController.text.trim(),
      apiKey: _apiKeyController.text.trim(),
      modelName: _modelNameController.text.trim(),
      apiFormat: _apiFormat,
    );
    if (mounted) {
      if (ok) {
        await _loadLlmConfig();
        context.showMessage(message: "LLM configuration saved".tl);
      } else {
        context.showMessage(message: "Failed to save LLM configuration".tl);
      }
      setState(() => _saving = false);
    }
  }

  String _formatSize(int bytes) {
    if (bytes < 1024) return "$bytes B";
    if (bytes < 1024 * 1024) return "${(bytes / 1024).toStringAsFixed(1)} KB";
    return "${(bytes / 1024 / 1024).toStringAsFixed(1)} MB";
  }

  @override
  Widget build(BuildContext context) {
    final currentLang =
        appdata.settings['translationLanguage'] as String? ?? 'ja_to_zh';
    return SmoothCustomScrollView(
      slivers: [
        SliverAppbar(title: Text("Translation".tl)),
        // ===== 远程 LLM 配置（翻译的前置条件，置顶） =====
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Row(
              children: [
                Text(
                  "LLM Configuration".tl,
                  style: TextStyle(
                    color: context.colorScheme.primary,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 8),
                if (!_loadingConfig)
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: (_llmConfigured ? Colors.green : Colors.orange)
                          .withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(
                        color: (_llmConfigured ? Colors.green : Colors.orange)
                            .withValues(alpha: 0.4),
                      ),
                    ),
                    child: Text(
                      _llmConfigured ? "Configured".tl : "Not configured".tl,
                      style: TextStyle(
                        fontSize: 10,
                        color: _llmConfigured
                            ? Colors.green.shade700
                            : Colors.orange.shade700,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
        SliverToBoxAdapter(
          child: Card(
            margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: _loadingConfig
                  ? const Center(
                      child: SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    )
                  : Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _buildTextField(
                          controller: _apiUrlController,
                          label: "API URL".tl,
                          hint: _apiFormat == 'gemini'
                              ? "https://generativelanguage.googleapis.com"
                              : "https://api.openai.com/v1",
                          enabled: _apiFormat != 'google_public',
                        ),
                        const SizedBox(height: 12),
                        _buildTextField(
                          controller: _apiKeyController,
                          label: "API Key".tl,
                          hint: "sk-...",
                          obscure: true,
                          enabled: _apiFormat != 'google_public',
                        ),
                        const SizedBox(height: 12),
                        _buildTextField(
                          controller: _modelNameController,
                          label: "Model".tl,
                          hint: _apiFormat == 'gemini'
                              ? "gemini-2.0-flash"
                              : "gpt-4o-mini",
                          enabled: _apiFormat != 'google_public',
                        ),
                        const SizedBox(height: 12),
                        Text("API Format".tl,
                            style:
                                const TextStyle(fontWeight: FontWeight.bold)),
                        const SizedBox(height: 4),
                        Wrap(
                          spacing: 8,
                          children: _apiFormatOptions.entries.map((e) {
                            final selected = _apiFormat == e.value;
                            return ChoiceChip(
                              label: Text(e.key.tl),
                              selected: selected,
                              onSelected: (_) {
                                setState(() {
                                  _apiFormat = e.value;
                                  // 切到谷歌公共翻译(免Key)时清空其他格式残留的地址/密钥/模型，
                                  // 避免保存后这些字段继续生效导致翻译失败（与原生 setLlmConfig 兜底一致）
                                  if (_apiFormat == 'google_public') {
                                    _apiUrlController.clear();
                                    _apiKeyController.clear();
                                    _modelNameController.clear();
                                  }
                                });
                              },
                            );
                          }).toList(),
                        ),
                        if (_apiFormat == 'google_public') ...[
                          const SizedBox(height: 12),
                          Text(
                            "Google public translate needs no API config".tl,
                            style: TextStyle(
                              fontSize: 12,
                              color: context.colorScheme.primary,
                            ),
                          ),
                        ],
                        const SizedBox(height: 16),
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton.icon(
                            onPressed: _saving ? null : _saveLlmConfig,
                            icon: _saving
                                ? const SizedBox(
                                    width: 16,
                                    height: 16,
                                    child: CircularProgressIndicator(
                                        strokeWidth: 2),
                                  )
                                : const Icon(Icons.save),
                            label: Text(
                                _saving ? "Saving...".tl : "Save".tl),
                          ),
                        ),
                      ],
                    ),
            ),
          ),
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            child: Text(
              "Translation runs entirely on-device for detection/OCR, but text "
              "translation calls this remote LLM. On-device detection/OCR models "
              "are not bundled with the app."
                  .tl,
              style: TextStyle(
                fontSize: 12,
                color: context.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        ),
                // ===== 本地翻译配置 (MarianMT) =====
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Text(
              "Local Translation (MarianMT)".tl,
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
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    title: Text("Use Local Translation".tl),
                    subtitle: Text("Use on-device ONNX model instead of remote LLM".tl),
                    value: _useLocalTranslation,
                    onChanged: (v) async {
                      await TranslationService.instance.setLocalTranslationConfig(
                        useLocalTranslation: v,
                        modelDir: _localModelDir,
                      );
                      setState(() => _useLocalTranslation = v);
                    },
                  ),
                  const SizedBox(height: 8),
                  TextField(
                    decoration: InputDecoration(
                      labelText: "Model Directory".tl,
                      hintText: "/sdcard/Android/data/.../translation/marian",
                      border: const OutlineInputBorder(),
                      isDense: true,
                      suffixIcon: _localModelAvailable
                          ? const Icon(Icons.check_circle, color: Colors.green)
                          : const Icon(Icons.warning, color: Colors.orange),
                    ),
                    controller: TextEditingController(text: _localModelDir),
                    onSubmitted: (v) async {
                      await TranslationService.instance.setLocalTranslationConfig(
                        useLocalTranslation: _useLocalTranslation,
                        modelDir: v,
                      );
                      final cfg = await TranslationService.instance.getLocalTranslationConfig();
                      setState(() {
                        _localModelDir = v;
                        _localModelAvailable = cfg['modelAvailable'] as bool? ?? false;
                      });
                    },
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _localModelAvailable
                        ? "Model files found. Ready to use.".tl
                        : "Place model files (model.onnx/encoder_model.onnx + vocab.json) in the directory above.".tl,
                    style: TextStyle(
                      fontSize: 12,
                      color: _localModelAvailable
                          ? Colors.green
                          : context.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
        // ===== 模型管理（按需下载，参考上色功能）=====
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Text(
              "Translation Models".tl,
              style: TextStyle(
                color: context.colorScheme.primary,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            child: Text(
              "Download translation models on demand to reduce APK size.".tl,
              style: TextStyle(
                fontSize: 12,
                color: context.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        ),
        const SliverToBoxAdapter(
          child: TranslationModelManagement(),
        ),
        // ===== 开关 =====
        _SwitchSetting(
          title: "Translate after download".tl,
          subtitle: "Batch-translate pages in background once download finishes"
              .tl,
          settingKey: "translateAfterDownload",
        ).toSliver(),
        // 阅读时翻译：实验性功能，单页翻译约 30 秒，体验较差，需显式提示
        SliverToBoxAdapter(
          child: ListTile(
            title: Row(
              children: [
                Text("Translate in reader".tl),
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: Colors.orange.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(4),
                    border: Border.all(
                      color: Colors.orange.withValues(alpha: 0.4),
                    ),
                  ),
                  child: Text(
                    "实验性".tl,
                    style: TextStyle(
                      fontSize: 10,
                      color: Colors.orange.shade700,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
            subtitle: Text(
              "Translate visible pages in real time while reading (Experimental: ~30s per page)"
                  .tl,
            ),
            trailing: Switch(
              value: appdata.settings['enableTranslation'] as bool? ?? false,
              onChanged: (v) {
                appdata.settings['enableTranslation'] = v;
                appdata.saveData();
                setState(() {});
              },
            ),
          ),
        ),
        // 目标语言选择
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Text(
              "Target Language".tl,
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
              child: Wrap(
                spacing: 8,
                runSpacing: 8,
                children: _languageOptions.entries.map((e) {
                  final selected = currentLang == e.value;
                  return ChoiceChip(
                    label: Text(e.key.tl),
                    selected: selected,
                    onSelected: (_) {
                      appdata.settings['translationLanguage'] = e.value;
                      setState(() {});
                    },
                  );
                }).toList(),
              ),
            ),
          ),
        ),
        _SwitchSetting(
          title: "Force OCR".tl,
          subtitle: "Re-run OCR even if a cached result exists".tl,
          settingKey: "translationForceOcr",
        ).toSliver(),
        ListTile(
          title: Text("Clear Translation Cache".tl),
          subtitle:
              _loadingCacheSize ? null : Text(_formatSize(_cacheSize).tl),
          trailing: const Icon(Icons.delete_sweep),
          onTap: () async {
            await TranslationService.instance.clearCache();
            await _refreshCacheSize();
            if (mounted) {
              context.showMessage(message: "Translation cache cleared".tl);
            }
          },
        ).toSliver(),
      ],
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    String? hint,
    bool obscure = false,
    bool enabled = true,
  }) {
    return TextField(
      controller: controller,
      obscureText: obscure,
      enabled: enabled,
      decoration: InputDecoration(
        labelText: label,
        hintText: enabled ? hint : null,
        border: const OutlineInputBorder(),
        isDense: true,
      ),
    );
  }
}

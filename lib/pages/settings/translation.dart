part of 'settings_page.dart';

/// 漫画翻译设置页
///
/// 提供「下载后自动翻译」（批量后台）与「阅读时实时翻译」开关、目标语言、
/// 强制 OCR 以及缓存管理。
///
/// 前置条件（与上游 jedzqer 一致）：翻译依赖用户在设置中配置的远程 LLM，
/// 以及端上检测 / OCR 模型（仓库内不含模型）。未配置时翻译会静默跳过。
class TranslationSettings extends StatefulWidget {
  const TranslationSettings({super.key});

  @override
  State<TranslationSettings> createState() => _TranslationSettingsState();
}

class _TranslationSettingsState extends State<TranslationSettings> {
  int _cacheSize = 0;
  bool _loadingCacheSize = true;

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

  String _formatSize(int bytes) {
    if (bytes < 1024) return "$bytes B";
    if (bytes < 1024 * 1024) return "${(bytes / 1024).toStringAsFixed(1)} KB";
    return "${(bytes / 1024 / 1024).toStringAsFixed(1)} MB";
  }

  String get _currentLanguageLabel {
    final value = appdata.settings['translationLanguage'] as String? ?? 'ja_to_zh';
    final entry = _languageOptions.entries.firstWhere(
      (e) => e.value == value,
      orElse: () => const MapEntry("日语 → 中文", "ja_to_zh"),
    );
    return entry.key;
  }

  @override
  Widget build(BuildContext context) {
    final currentLang =
        appdata.settings['translationLanguage'] as String? ?? 'ja_to_zh';
    return SmoothCustomScrollView(
      slivers: [
        SliverAppbar(title: Text("Translation".tl)),
        _SwitchSetting(
          title: "Translate after download".tl,
          subtitle: "Batch-translate pages in background once download finishes"
              .tl,
          settingKey: "translateAfterDownload",
        ).toSliver(),
        _SwitchSetting(
          title: "Translate in reader".tl,
          subtitle: "Translate visible pages in real time while reading".tl,
          settingKey: "enableTranslation",
        ).toSliver(),
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
        // 前置条件说明
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: context.colorScheme.surfaceContainerHighest
                    .withValues(alpha: 0.5),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: context.colorScheme.outlineVariant,
                ),
              ),
              child: Text(
                "Requires a configured remote LLM (in app settings) and on-device "
                "detection/OCR models. Translation silently skips pages when the "
                "LLM is not configured."
                    .tl,
                style: TextStyle(
                  fontSize: 12,
                  color: context.colorScheme.onSurfaceVariant,
                ),
              ),
            ),
          ),
        ),
        ListTile(
          title: Text("Clear Translation Cache".tl),
          subtitle: _loadingCacheSize
              ? null
              : Text(_formatSize(_cacheSize).tl),
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
}

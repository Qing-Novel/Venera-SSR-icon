import 'package:flutter/material.dart';
import 'package:venera/utils/translation/translation_model_manager.dart';

/// 翻译模型管理 UI：列出可下载的翻译模型，支持下载 / 删除 / 进度显示 / 镜像地址管理。
/// 复刻上色功能（ColorizationSettings）的下载体验。原生加载暂仍读取 assets，
/// 此处先把模型下载到应用目录并做校验，便于后续切换到优先读取下载目录。
class TranslationModelManagement extends StatefulWidget {
  const TranslationModelManagement({super.key});

  @override
  State<TranslationModelManagement> createState() =>
      _TranslationModelManagementState();
}

class _TranslationModelManagementState
    extends State<TranslationModelManagement> {
  final Map<String, bool> _downloaded = {};
  final Map<String, int> _sizes = {};
  bool _loading = true;
  String? _downloadingId;
  double _progress = 0;
  String? _status;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  String _formatSize(int bytes) {
    if (bytes < 1024) return "$bytes B";
    if (bytes < 1024 * 1024) return "${(bytes / 1024).toStringAsFixed(1)} KB";
    return "${(bytes / 1024 / 1024).toStringAsFixed(1)} MB";
  }

  Future<void> _refresh() async {
    setState(() => _loading = true);
    final d = <String, bool>{};
    final s = <String, int>{};
    for (final m in TranslationModelManager.models) {
      d[m.id] = await TranslationModelManager.isDownloaded(m.id);
      s[m.id] = await TranslationModelManager.getDownloadedSize(m.id);
    }
    if (mounted) {
      setState(() {
        _downloaded
          ..clear()
          ..addAll(d);
        _sizes
          ..clear()
          ..addAll(s);
        _loading = false;
      });
    }
  }

  Future<void> _download(TranslationModelEntry m) async {
    setState(() {
      _downloadingId = m.id;
      _progress = 0;
      _status = "Starting...".tl;
    });
    try {
      await TranslationModelManager.download(
        id: m.id,
        onProgress: (p) => setState(() => _progress = p),
        onStatus: (s) => setState(() => _status = s),
      );
      if (mounted) context.showMessage(message: "Download complete".tl);
    } catch (e) {
      if (mounted) context.showMessage(message: "Download failed".tl);
    } finally {
      _downloadingId = null;
      await _refresh();
    }
  }

  Future<void> _delete(TranslationModelEntry m) async {
    await TranslationModelManager.deleteModel(m.id);
    await _refresh();
    if (mounted) {
      context.showMessage(message: "Model deleted".tl);
    }
  }

  Future<void> _manageMirrors(TranslationModelEntry m) async {
    final urls = List<String>.from(await TranslationModelManager.getUrls(m.id));
    if (!mounted) return;
    await showDialog(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialog) => AlertDialog(
          title: Text("Mirror URLs".tl),
          content: SizedBox(
            width: double.maxFinite,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                ...urls.asMap().entries.map((entry) {
                  final i = entry.key;
                  final url = entry.value;
                  return ListTile(
                    dense: true,
                    title: Text(url, style: const TextStyle(fontSize: 13)),
                    trailing: IconButton(
                      icon: const Icon(Icons.delete_outline, size: 18),
                      onPressed: () {
                        urls.removeAt(i);
                        setDialog(() {});
                        TranslationModelManager.removeUrlAt(m.id, i);
                      },
                    ),
                  );
                }),
                const SizedBox(height: 8),
                FilledButton.tonal(
                  onPressed: () async {
                    final controller = TextEditingController();
                    final added = await showDialog<String>(
                      context: ctx,
                      builder: (c) => AlertDialog(
                        title: Text("Add Mirror URL".tl),
                        content: TextField(
                          controller: controller,
                          decoration: InputDecoration(
                            hintText: "https://...",
                            border: const OutlineInputBorder(),
                            isDense: true,
                          ),
                          autofocus: true,
                        ),
                        actions: [
                          TextButton(
                            onPressed: () => Navigator.pop(c),
                            child: Text("Cancel".tl),
                          ),
                          FilledButton(
                            onPressed: () => Navigator.pop(c, controller.text.trim()),
                            child: Text("Add".tl),
                          ),
                        ],
                      ),
                    );
                    if (added != null && added.isNotEmpty) {
                      await TranslationModelManager.addUrl(m.id, added);
                      urls
                        ..clear()
                        ..addAll(await TranslationModelManager.getUrls(m.id));
                      setDialog(() {});
                    }
                  },
                  child: Text("Add Mirror URL".tl),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () async {
                await TranslationModelManager.resetUrls(m.id);
                urls
                  ..clear()
                  ..addAll(await TranslationModelManager.getUrls(m.id));
                setDialog(() {});
              },
              child: Text("Reset".tl),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(ctx),
              child: Text("Done".tl),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Padding(
        padding: EdgeInsets.all(16),
        child: Center(
          child: SizedBox(
            width: 24,
            height: 24,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
        ),
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: TranslationModelManager.models.map((m) => _buildCard(m)).toList(),
    );
  }

  Widget _buildCard(TranslationModelEntry m) {
    final isThisDownloading = _downloadingId == m.id;
    final downloaded = _downloaded[m.id] == true;
    final size = _sizes[m.id] ?? 0;
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    m.label,
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                ),
                if (downloaded)
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.green.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      "Model downloaded".tl,
                      style: TextStyle(
                        fontSize: 10,
                        color: Colors.green.shade700,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  )
                else
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.orange.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      "Model not downloaded (lightweight)".tl,
                      style: TextStyle(
                        fontSize: 10,
                        color: Colors.orange.shade700,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
              ],
            ),
            if (downloaded)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  "Model size: @size".tlParams({'size': _formatSize(size)}),
                  style: TextStyle(
                    fontSize: 12,
                    color: context.colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
            if (isThisDownloading) ...[
              const SizedBox(height: 8),
              LinearProgressIndicator(value: _progress),
              const SizedBox(height: 4),
              Text(
                _status ?? "",
                style: const TextStyle(fontSize: 12),
              ),
            ],
            const SizedBox(height: 8),
            Row(
              children: [
                if (!isThisDownloading)
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: downloaded
                          ? null
                          : () => _download(m),
                      icon: const Icon(Icons.download, size: 18),
                      label: Text(
                        downloaded
                            ? "Downloaded".tl
                            : (_downloadingId == null
                                ? "Download Model".tl
                                : "Downloading...".tl),
                      ),
                    ),
                  )
                else
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: null,
                      icon: const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                      label: Text("Downloading...".tl),
                    ),
                  ),
                const SizedBox(width: 8),
                IconButton(
                  onPressed: downloaded ? () => _delete(m) : null,
                  icon: const Icon(Icons.delete_outline),
                  tooltip: "Delete Model".tl,
                ),
                IconButton(
                  onPressed: () => _manageMirrors(m),
                  icon: const Icon(Icons.link),
                  tooltip: "Tap to manage mirrors".tl,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

part of 'settings_page.dart';

import 'package:venera/utils/colorization/colorization_service.dart';

/// 图像上色设置页
///
/// 参考 Anime4K 设置页的模式。提供开关和强度调节。
class ColorizationSettings extends StatefulWidget {
  const ColorizationSettings({super.key});

  @override
  State<ColorizationSettings> createState() => _ColorizationSettingsState();
}

class _ColorizationSettingsState extends State<ColorizationSettings> {
  @override
  Widget build(BuildContext context) {
    return SmoothCustomScrollView(
      slivers: [
        SliverAppbar(title: Text("Colorization".tl)),
        _SwitchSetting(
          title: "Enable Image Colorization".tl,
          settingKey: "enableColorization",
        ).toSliver(),
        _SliderSetting(
          title: "Colorization Intensity".tl,
          settingsIndex: "colorizationIntensity",
          min: 0.3,
          max: 1.2,
          interval: 0.05,
        ).toSliver(),
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

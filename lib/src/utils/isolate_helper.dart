import 'dart:typed_data';

import 'package:image_background_remover/image_background_remover.dart';

/// Isolate-compatible background removal configuration
class RemoveBgConfig {
  final Uint8List imageBytes;
  final double threshold;
  final bool smoothMask;
  final bool enhanceEdges;

  const RemoveBgConfig({
    required this.imageBytes,
    this.threshold = 0.5,
    this.smoothMask = true,
    this.enhanceEdges = true,
  });
}

/// Top-level function for isolate execution
/// Must be top-level or static to work with Isolate.run()
Future<Uint8List> removeBgInIsolate(RemoveBgConfig config) async {
  // Initialize ONNX session in this isolate
  await BackgroundRemover.instance.initializeOrt();

  // Process the image
  final result = await BackgroundRemover.instance.removeBgBytes(
    config.imageBytes,
    threshold: config.threshold,
    smoothMask: config.smoothMask,
    enhanceEdges: config.enhanceEdges,
  );

  // Clean up
  await BackgroundRemover.instance.dispose();

  return result;
}

import 'package:example/image_picker.dart';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_background_remover/image_background_remover.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final ValueNotifier<Uint8List?> outImg = ValueNotifier<Uint8List?>(null);

  @override
  void initState() {
    BackgroundRemover.instance.initializeOrt();
    super.initState();
  }

  @override
  void dispose() {
    // Note: Since dispose is synchronous, we can't await here.
    // The session will be cleaned up by the garbage collector if not explicitly closed.
    // For proper cleanup, consider calling BackgroundRemover.instance.dispose()
    // in a place where async is supported, such as before app termination.
    BackgroundRemover.instance.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.blue,
      appBar: AppBar(
        title: const Text('Background Remover'),
      ),
      body: SafeArea(
        child: ValueListenableBuilder(
          valueListenable: ImagePickerService.pickedFile,
          builder: (context, image, _) {
            return GestureDetector(
              onTap: () async {
                await ImagePickerService.pickImage();
              },
              child: Container(
                alignment: Alignment.center,
                child: image == null
                    ? const Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.image,
                            size: 100,
                          ),
                          Text('No image selected.'),
                        ],
                      )
                    : SingleChildScrollView(
                        child: Column(
                          children: [
                            Image.file(image),
                            const SizedBox(
                              height: 20,
                            ),
                            TextButton(
                              onPressed: () async {
                                Uint8List data = await BackgroundRemover.instance
                                    .removeBg(image.readAsBytesSync());

                                outImg.value = data;
                              },
                              child: const Text('Remove Background'),
                            ),
                            TextButton(
                              onPressed: () async {
                                //Uint8List data = await BackgroundRemover.instance
                                    //.removeBGAddStroke(image.readAsBytesSync(), stokeWidth: 50, stokeColor: Colors.blue, secondaryStrokeWidth: 5);

                                Uint8List result = await BackgroundRemover.instance.removeBGScaleAndAddStroke(
                                  image.readAsBytesSync(),
                                  secondaryStrokeColor: Colors.black,
                                  strokeWidthMM: 2.0, // 2mm stroke
                                  secondaryStrokeWidthMM: 0.0, // optional outer stroke
                                  targetWidthMM: 50, // Resize to 50mm wide
                                  targetHeightMM: 70,
                                  strokeColor: Colors.white, // Resize to 70mm tall
                                );


                                outImg.value = result;
                              },
                              child: const Text('Remove Background With Stroke'),
                            ),
                            ValueListenableBuilder(
                              valueListenable: outImg,
                              builder: (context, img, _) {
                                return img == null
                                    ? const SizedBox()
                                    : Image.memory(img);
                              },
                            ),
                            ValueListenableBuilder(
                              valueListenable: outImg,
                              builder: (context, img, _) {
                                return img == null
                                    ? const SizedBox()
                                    : TextButton(
                                  onPressed: () async {
                                    File file = await saveImageToInternalStorage(img);
                                    if(file.existsSync()) {
                                      ScaffoldMessenger.of(context).showSnackBar(
                                        SnackBar(
                                          content: Text('Image saved to ${file.path}'),
                                        ),
                                      );
                                    } else {
                                      ScaffoldMessenger.of(context).showSnackBar(
                                        const SnackBar(
                                          content: Text('Failed to save image'),
                                        ),
                                      );
                                    }
                                  },
                                  child: const Text('Download Image'),
                                );
                              },
                            ),
                          ],
                        ),
                      ),
              ),
            );
          },
        ),
      ),
    );
  }

  /// Converts millimeters to pixels using DPI
  int mmToPixels(double mm, int dpi) {
    return ((mm / 25.4) * dpi).round(); // 1 inch = 25.4 mm
  }

  /// Resize with or without cropping, preserving aspect ratio
  /// Resize and add border/stroke
  Future<Uint8List> resizeImageWithStroke({
    required Uint8List inputBytes,
    required double targetWidthMm,
    required double targetHeightMm,
    double strokeWidthMm = 2.0,
    int dpi = 300,
    bool cropToFit = false,
    img.Color? strokeColor,
  }) async {
    final targetWidthPx = mmToPixels(targetWidthMm, dpi);
    final targetHeightPx = mmToPixels(targetHeightMm, dpi);
    final strokePx = mmToPixels(strokeWidthMm, dpi);
    strokeColor ??= img.ColorRgb8(0, 0, 0); // default black

    final original = img.decodeImage(inputBytes);
    if (original == null) throw Exception("Invalid image");

    late img.Image resized;
    if (cropToFit) {
      final targetRatio = targetWidthPx / targetHeightPx;
      final originalRatio = original.width / original.height;

      int cropWidth = original.width;
      int cropHeight = original.height;
      if (originalRatio > targetRatio) {
        cropWidth = (original.height * targetRatio).round();
      } else {
        cropHeight = (original.width / targetRatio).round();
      }

      final cropX = ((original.width - cropWidth) / 2).round();
      final cropY = ((original.height - cropHeight) / 2).round();

      final cropped = img.copyCrop(
        original,
        x: cropX,
        y: cropY,
        width: cropWidth,
        height: cropHeight,
      );

      resized = img.copyResize(
        cropped,
        width: targetWidthPx,
        height: targetHeightPx,
      );
    } else {
      final originalRatio = original.width / original.height;
      final targetRatio = targetWidthPx / targetHeightPx;

      int newWidth, newHeight;
      if (originalRatio > targetRatio) {
        newWidth = targetWidthPx;
        newHeight = (targetWidthPx / originalRatio).round();
      } else {
        newHeight = targetHeightPx;
        newWidth = (targetHeightPx * originalRatio).round();
      }

      resized = img.copyResize(original, width: newWidth, height: newHeight);
    }

    final canvasWidth = resized.width + 2 * strokePx;
    final canvasHeight = resized.height + 2 * strokePx;

    final canvas = img.Image(width: canvasWidth, height: canvasHeight);
    img.fill(canvas,  color: img.ColorRgb8(255, 255, 255)); // white background

    img.drawRect(
      canvas,
      x1: strokePx ~/ 2,
      y1: strokePx ~/ 2,
      x2: canvasWidth - strokePx ~/ 2 - 1,
      y2: canvasHeight - strokePx ~/ 2 - 1,
      color: strokeColor,
      thickness: strokePx,
    );

    img.compositeImage(
      canvas,
      resized,
      dstX: strokePx,
      dstY: strokePx,
    );

    return Uint8List.fromList(img.encodePng(canvas));
  }

  /// Saves [imageBytes] as a PNG file in the app's internal directory.
  /// Returns the [File] pointing to the saved image.
  Future<File> saveImageToInternalStorage(Uint8List imageBytes, {String filename = 'image'}) async {
    // Get internal storage directory
    final Directory dir = await getApplicationDocumentsDirectory();
    final String path = '${dir.path}/${filename}_${DateTime.now().millisecondsSinceEpoch}.png';

    // Write to file
    final File file = File(path);
    await file.writeAsBytes(imageBytes);

    return file;
  }
}

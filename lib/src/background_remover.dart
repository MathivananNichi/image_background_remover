import 'dart:async';
import 'dart:developer';
import 'dart:ui' as ui;

import 'package:image/image.dart' as img;
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image_background_remover/src/services/onnx_session_manager.dart';
import 'package:image_background_remover/src/utils/background_composer.dart';
import 'package:image_background_remover/src/utils/image_processor.dart';
import 'package:image_background_remover/src/utils/mask_processor.dart';

/// Main class for background removal functionality
class BackgroundRemover {
  BackgroundRemover._internal();

  static final BackgroundRemover _instance = BackgroundRemover._internal();

  static BackgroundRemover get instance => _instance;

  /// Session manager for ONNX Runtime
  final OnnxSessionManager _sessionManager = OnnxSessionManager();

  /// Model input/output size
  int modelSize = 320;

  /// Initializes the ONNX environment and creates a session.
  ///
  /// This method should be called once before using the [removeBg] method.
  Future<void> initializeOrt() async {
    try {
      await _sessionManager.initialize();
    } catch (e) {
      log('Failed to initialize ONNX session: $e', name: "BackgroundRemover");
      rethrow;
    }
  }

  /// Removes the background from an image.
  ///
  /// This function processes the input image and removes its background,
  /// returning a new image with the background removed.
  ///
  /// - [imageBytes]: The input image as a byte array.
  /// - [threshold]: The threshold value for foreground/background separation (default: 0.5).
  /// - [smoothMask]: Whether to apply smoothing to the output mask (default: true).
  /// - [enhanceEdges]: Whether to enhance mask edges using image gradients (default: true).
  /// - Returns: A [ui.Image] with the background removed.
  ///
  /// Example usage:
  /// ```dart
  /// final imageBytes = await File('path_to_image').readAsBytes();
  /// final ui.Image imageWithoutBackground = await BackgroundRemover.instance.removeBg(imageBytes);
  /// ```
  ///
  /// Note: This function may take some time to process depending on the size
  /// and complexity of the input image.
  Future<ui.Image> removeBg(
      Uint8List imageBytes, {
        double threshold = 0.5,
        bool smoothMask = true,
        bool enhanceEdges = true,
      }) async {
    if (!_sessionManager.isInitialized) {
      throw Exception(
          "ONNX session not initialized. Call initializeOrt() first.");
    }

    /// Decode the input image
    final originalImage = await _compressImageIfNeeded(imageBytes);
    if (originalImage==null) {
      throw Exception(
          "Image size is to large");
    }
    log('Original image size: ${originalImage.width}x${originalImage.height}',
        name: "BackgroundRemover");

    final resizedImage =
    await ImageProcessor.resizeImage(originalImage, modelSize, modelSize);

    /// Convert the resized image into a tensor format required by the ONNX model
    final rgbFloats = await ImageProcessor.imageToFloatTensor(resizedImage);

    /// Migration: Changed from OrtValueTensor.createTensorWithDataList to OrtValue.fromList
    final inputTensor = await OrtValue.fromList(
      Float32List.fromList(rgbFloats),
      [1, 3, modelSize, modelSize],
    );

    /// Prepare the inputs and run inference on the ONNX model
    final inputs = {'input.1': inputTensor};

    /// Migration: Simplified to use run() instead of runAsync() with OrtRunOptions
    final outputs = await _sessionManager.session!.run(inputs);

    /// Migration: Proper tensor disposal for memory management
    await inputTensor.dispose();

    /// Process the output tensor and generate the final image with the background removed
    /// Migration: Access outputs using named output instead of indexed access
    final outputName = _sessionManager.session!.outputNames.first;
    final outputTensor = outputs[outputName];

    if (outputTensor == null) {
      throw Exception('Unexpected output format from ONNX model.');
    }

    /// Migration: Use asList() to get data with proper shape preservation
    final outputData = await outputTensor.asList();
    final mask = outputData[0][0];

    /// Generate and refine the mask
    final resizedMask = smoothMask
        ? MaskProcessor.resizeMaskBilinear(
        mask, originalImage.width, originalImage.height)
        : MaskProcessor.resizeMaskNearest(
        mask, originalImage.width, originalImage.height,
        maskSize: modelSize);

    /// Apply edge enhancement if requested
    final finalMask = enhanceEdges
        ? await MaskProcessor.enhanceMaskEdges(originalImage, resizedMask)
        : resizedMask;

    /// Apply the mask to the original image
    final result = await ImageProcessor.applyMaskToImage(
      originalImage,
      finalMask,
      threshold: threshold,
      smooth: smoothMask,
    );

    /// Migration: Dispose output tensor to free native resources
    await outputTensor.dispose();

    /// Clean up intermediate images
    originalImage.dispose();
    resizedImage.dispose();

    return result;
  }

  /// Removes the background from an image and returns PNG bytes (isolate-compatible).
  ///
  /// This function is designed to work with Dart isolates. Unlike [removeBg],
  /// it returns PNG-encoded bytes instead of a ui.Image, making it safe to use
  /// with Isolate.run() or compute().
  ///
  /// - [imageBytes]: The input image as a byte array.
  /// - [threshold]: The threshold value for foreground/background separation (default: 0.5).
  /// - [smoothMask]: Whether to apply smoothing to the output mask (default: true).
  /// - [enhanceEdges]: Whether to enhance mask edges using image gradients (default: true).
  /// - Returns: PNG-encoded bytes with the background removed.
  ///
  /// Example usage with isolate:
  /// ```dart
  /// final imageBytes = await File('path_to_image').readAsBytes();
  /// final resultBytes = await Isolate.run(() async {
  ///   await BackgroundRemover.instance.initializeOrt();
  ///   return await BackgroundRemover.instance.removeBgBytes(imageBytes);
  /// });
  /// // Convert to ui.Image if needed
  /// final image = await decodeImageFromList(resultBytes);
  /// ```
  ///
  /// Note: When using in an isolate, you must call [initializeOrt] within the isolate.
  Future<Uint8List> removeBgBytes(
      Uint8List imageBytes, {
        double threshold = 0.5,
        bool smoothMask = true,
        bool enhanceEdges = true,
      }) async {
    // Process the image
    final resultImage = await removeBg(
      imageBytes,
      threshold: threshold,
      smoothMask: smoothMask,
      enhanceEdges: enhanceEdges,
    );

    // Convert ui.Image to PNG bytes
    final byteData =
    await resultImage.toByteData(format: ui.ImageByteFormat.png);
    resultImage.dispose();

    if (byteData == null) {
      throw Exception('Failed to convert image to bytes');
    }

    return byteData.buffer.asUint8List();
  }
  Future<Uint8List> removeBGAddStroke(Uint8List image,
      {required Color stokeColor, Color secondaryStokeColor = Colors.black, required double stokeWidth, double secondaryStrokeWidth = 6.0}) async {
    ui.Image bgRemoved = await removeBg(image);
    ui.Image uiImage = bgRemoved;

    // Pad the image so the stroke has space to render
    final paddedImage = await padImageWithTransparentBorder(uiImage, 20);

    ui.Image withDualStroke = await addDualStrokeToTransparentImage(
      image: paddedImage,
      innerBorderColor: stokeColor,
      innerBorderWidth: stokeWidth,
      outerBorderColor: secondaryStokeColor.withValues(alpha: 0.2),
      outerBorderWidth: secondaryStrokeWidth,
    );

    Uint8List strokeAdded = await convertUiImageToPngBytes(withDualStroke);
    return strokeAdded;
  }


  Future<Uint8List> convertUiImageToPngBytes(ui.Image image) async {
    final ByteData? byteData =
    await image.toByteData(format: ui.ImageByteFormat.png);
    return byteData!.buffer.asUint8List();
  }

  Future<ui.Image> padImageWithTransparentBorder(ui.Image image, int padding) async {
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);

    final newWidth = image.width + 2 * padding;
    final newHeight = image.height + 2 * padding;

    final paint = Paint();
    canvas.drawImage(image, Offset(padding.toDouble(), padding.toDouble()), paint);

    final picture = recorder.endRecording();
    return await picture.toImage(newWidth, newHeight);
  }


  Future<ui.Image> addDualStrokeToTransparentImage({
    required ui.Image image,
    required Color innerBorderColor,
    required double innerBorderWidth,
    required Color outerBorderColor,
    required double outerBorderWidth,
  }) async {
    final width = image.width;
    final height = image.height;

    final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) return image;
    final pixels = byteData.buffer.asUint8List();

    // Copy original image data to result buffer
    final resultPixels = Uint8List.fromList(pixels);
    final edgePoints = <Offset>[];

    // Step 1: Detect edge pixels (pixels that are opaque but have transparent neighbors)
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final i = (y * width + x) * 4;
        final alpha = pixels[i + 3];

        // Skip transparent pixels
        if (alpha < 128) continue;

        bool isEdge = false;

        // Check if this opaque pixel has any transparent neighbors
        for (int dy = -1; dy <= 1 && !isEdge; dy++) {
          for (int dx = -1; dx <= 1 && !isEdge; dx++) {
            if (dx == 0 && dy == 0) continue;

            final nx = x + dx;
            final ny = y + dy;

            // Consider boundary pixels as edges
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
              isEdge = true;
            } else {
              final ni = (ny * width + nx) * 4;
              // If neighbor is transparent, this is an edge pixel
              if (pixels[ni + 3] < 128) {
                isEdge = true;
              }
            }
          }
        }

        if (isEdge) {
          edgePoints.add(Offset(x.toDouble(), y.toDouble()));
        }
      }
    }

    // Step 2: Apply outer stroke (only to transparent areas near edges)
    final outerRadius = (innerBorderWidth + outerBorderWidth).round();
    final outerR = outerBorderColor.red;
    final outerG = outerBorderColor.green;
    final outerB = outerBorderColor.blue;
    final outerA = (outerBorderColor.opacity * 255).round();

    for (final point in edgePoints) {
      for (int dy = -outerRadius; dy <= outerRadius; dy++) {
        for (int dx = -outerRadius; dx <= outerRadius; dx++) {
          final x = point.dx.toInt() + dx;
          final y = point.dy.toInt() + dy;
          if (x < 0 || x >= width || y < 0 || y >= height) continue;

          final distance = dx * dx + dy * dy;
          if (distance > outerRadius * outerRadius) continue;

          final i = (y * width + x) * 4;

          // Only apply outer stroke to transparent areas
          if (pixels[i + 3] < 128 && resultPixels[i + 3] < outerA) {
            resultPixels[i] = outerR;
            resultPixels[i + 1] = outerG;
            resultPixels[i + 2] = outerB;
            resultPixels[i + 3] = outerA;
          }
        }
      }
    }

    // Step 3: Apply inner stroke (only to transparent areas near edges, closer to the object)
    final innerRadius = innerBorderWidth.round();
    final innerR = (innerBorderColor.r * 255).round().clamp(0, 255);
    final innerG = (innerBorderColor.g * 255).round().clamp(0, 255);
    final innerB = (innerBorderColor.b * 255).round().clamp(0, 255);

    for (final point in edgePoints) {
      for (int dy = -innerRadius; dy <= innerRadius; dy++) {
        for (int dx = -innerRadius; dx <= innerRadius; dx++) {
          final x = point.dx.toInt() + dx;
          final y = point.dy.toInt() + dy;
          if (x < 0 || x >= width || y < 0 || y >= height) continue;

          final distance = dx * dx + dy * dy;
          if (distance > innerRadius * innerRadius) continue;

          final i = (y * width + x) * 4;

          // Only apply inner stroke to transparent areas
          if (pixels[i + 3] < 128) {
            resultPixels[i] = innerR;
            resultPixels[i + 1] = innerG;
            resultPixels[i + 2] = innerB;
            resultPixels[i + 3] = 255; // solid inner stroke
          }
        }
      }
    }

    // Step 4: Create ui.Image from modified pixels
    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      resultPixels,
      width,
      height,
      ui.PixelFormat.rgba8888,
      completer.complete,
    );
    return completer.future;
  }

  Future<Uint8List> removeBGScaleAndAddStroke(
      Uint8List inputImageBytes, {
        required double targetWidthMM,
        required double targetHeightMM,
        required Color strokeColor,
        Color secondaryStrokeColor = Colors.black,
        double strokeWidthMM = 2.0,
        double secondaryStrokeWidthMM = 0.0,
        double dpi = 300,
      }) async {

    ui.Image? uiImage;
    ui.Image? padded;
    ui.Image? finalImage;

    try {
      // 1. Convert mm to px
      final targetWidthPx = ((targetWidthMM / 25.4) * dpi).round();
      final targetHeightPx = ((targetHeightMM / 25.4) * dpi).round();
      final strokePx = ((strokeWidthMM / 25.4) * dpi).round();
      final secondaryStrokePx = ((secondaryStrokeWidthMM / 25.4) * dpi).round();

      // 2. Remove background with compression
      ui.Image bgRemoved = await removeBg(inputImageBytes);
      final byteData = await bgRemoved.toByteData(format: ui.ImageByteFormat.png);
      var rr= byteData?.buffer.asUint8List();
      final image = img.decodeImage(rr!)!;

      // 3. Maintain aspect ratio and resize
      final srcWidth = image.width;
      final srcHeight = image.height;
      final srcAspectRatio = srcWidth / srcHeight;
      final targetAspectRatio = targetWidthPx / targetHeightPx;

      int resizedWidth, resizedHeight;
      if (srcAspectRatio > targetAspectRatio) {
        resizedWidth = targetWidthPx;
        resizedHeight = (targetWidthPx / srcAspectRatio).round();
      } else {
        resizedHeight = targetHeightPx;
        resizedWidth = (targetHeightPx * srcAspectRatio).round();
      }

      final resized = img.copyResize(
        image,
        width: resizedWidth,
        height: resizedHeight,
        interpolation: img.Interpolation.linear,
      );

      // 4. Convert to ui.Image
      uiImage = await convertImageToUiImage(resized);

      // 5. Pad to ensure space for stroke
      padded = await padImageWithTransparentBorder(
        uiImage,
        strokePx + secondaryStrokePx + 10,
      );

      // 6. Add stroke
      finalImage = await addDualStrokeToTransparentImage(
        image: padded,
        innerBorderColor: strokeColor,
        innerBorderWidth: strokePx.toDouble(),
        outerBorderColor: secondaryStrokeColor.withOpacity(0.2),
        outerBorderWidth: secondaryStrokePx.toDouble(),
      );

      // 7. Encode final image
      return await convertUiImageToPngBytes(finalImage);
    } finally {
      // Clean up UI images
      uiImage?.dispose();
      padded?.dispose();
      finalImage?.dispose();
    }
  }


  Future<ui.Image> convertImageToUiImage(img.Image image) async {
    final completer = Completer<ui.Image>();
    final bytes = Uint8List.fromList(img.encodePng(image));
    ui.decodeImageFromList(bytes, (ui.Image img) => completer.complete(img));
    return completer.future;
  }
  /// Adds a background color to the given image.
  ///
  /// This method takes an image in the form of a [Uint8List] and a background
  /// color as a [Color]. It decodes the image, creates a new image with the
  /// same dimensions, fills it with the specified background color, and then
  /// composites the original image onto the new image with the background color.
  ///
  /// Returns a [Future] that completes with the modified image as a [Uint8List].
  ///
  /// - Parameters:
  ///   - image: The original image as a [Uint8List].
  ///   - bgColor: The background color as a [Color].
  ///
  /// - Returns: A [Future] that completes with the modified image as a [Uint8List].
  Future<Uint8List> addBackground({
    required Uint8List image,
    required Color bgColor,
  }) async {
    return BackgroundComposer.addBackground(image: image, bgColor: bgColor);
  }

  // Compresses image if it's too large while maintaining quality
  Future<ui.Image?> _compressImageIfNeeded(Uint8List imageBytes) async {
    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) return null;

    final estimatedMemoryUsage =
    _getImageMemoryUsage(originalImage.width, originalImage.height);
    final availableMemory = _getAvailableMemory();

    if (estimatedMemoryUsage <= availableMemory) {
      return await decodeImageFromList(imageBytes);
    }

    final memoryRatio = availableMemory / estimatedMemoryUsage;
    final scaleFactor = math.sqrt(memoryRatio * 0.8);

    final newWidth = (originalImage.width * scaleFactor).round();
    final newHeight = (originalImage.height * scaleFactor).round();

    log('Compressing image from ${originalImage.width}x${originalImage.height} to ${newWidth}x${newHeight}');

    img.Image compressed = img.copyResize(
      originalImage,
      width: newWidth,
      height: newHeight,
      interpolation: img.Interpolation.linear,
    );

    // ✅ Convert img.Image → Uint8List
    final compressedBytes = Uint8List.fromList(
      img.encodeJpg(compressed),
    );

    // ✅ Convert Uint8List → ui.`Image
    return await decodeImageFromList(compressedBytes);
  }

  int _getImageMemoryUsage(int width, int height) {
    // RGBA format: 4 bytes per pixel
    // Plus overhead for processing (multiple copies during processing)
    return width * height * 4 * 3; // 3x for processing overhead
  }

  /// Gets available memory for image processing (conservative estimate)
  int _getAvailableMemory() {
    if (kIsWeb) {
      // Web has more limited memory
      return 100 * 1024 * 1024; // 100MB
    } else {
      // Mobile devices - conservative estimate
      return 200 * 1024 * 1024; // 200MB
    }
  }

  /// Release resources
  /// Migration: Changed to async and use close() instead of release()
  Future<void> dispose() async {
    await _sessionManager.dispose();
  }
}
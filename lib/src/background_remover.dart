import 'dart:async';
import 'dart:developer';
import 'dart:typed_data';
import 'dart:ui' as ui;

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
    final originalImage = await decodeImageFromList(imageBytes);
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

  /// Release resources
  /// Migration: Changed to async and use close() instead of release()
  Future<void> dispose() async {
    await _sessionManager.dispose();
  }
}

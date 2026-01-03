# Using Background Remover with Isolates

## The Problem

When trying to use `Isolate.run()` with Flutter's `ui.Image` objects, you'll encounter this error:

```
Invalid argument(s): Illegal argument in isolate message: object is unsendable
```

**Root Cause:** Dart isolates can only communicate using simple, serializable types. Flutter framework objects like `ui.Image`, `Widget`, etc., contain references to the Flutter bindings and cannot cross isolate boundaries.

### What CAN be sent across isolates:
✅ Primitive types: `int`, `double`, `String`, `bool`, `null`  
✅ Lists and Maps of sendable types  
✅ `TypedData`: `Uint8List`, `Int32List`, `Float64List`, etc.  
✅ Custom classes with only sendable fields  

### What CANNOT be sent:
❌ `ui.Image`  
❌ `Widget` objects  
❌ `BuildContext`  
❌ Platform channels  
❌ Any object with Flutter framework bindings  

---

## The Solution

Work with **bytes** (`Uint8List`) instead of `ui.Image` when using isolates.

### Using the Built-in Isolate 

```dart
import 'dart:isolate';
import 'package:image_background_remover/image_background_remover.dart';

// In your widget
Future<void> removeBackground() async {
  final imageBytes = await File('path/to/image.jpg').readAsBytes();
  
  // Process in isolate (runs on separate thread)
  final resultBytes = await Isolate.run(() {
    return removeBgInIsolate(
      RemoveBgConfig(
        imageBytes: imageBytes,
        threshold: 0.5,
        smoothMask: true,
        enhanceEdges: true,
      ),
    );
  });
  
  // Convert back to ui.Image in main isolate
  final ui.Image resultImage = await decodeImageFromList(resultBytes);
  
  // Use the image
  setState(() {
    _processedImage = resultImage;
  });
}
```

## Performance Considerations

### When to use Isolates:
✅ Processing large images  
✅ When you want to avoid UI freezing  
✅ Batch processing multiple images  

### When NOT to use Isolates:
❌ Small, quick operations (isolate overhead > processing time)  
❌ When immediate UI update is needed  
❌ When processing is already fast enough  

### Important Notes:

1. **Isolate Overhead**: Creating an isolate and initializing ONNX session adds overhead. For small images, direct processing might be faster.

2. **Memory**: Each isolate has its own memory space. The ONNX model will be loaded separately in each isolate.

3. **Initialization**: The ONNX session must be initialized **inside** the isolate, not shared from the main isolate.

4. **Cleanup**: Always dispose resources in both main and isolate contexts.

---

## Troubleshooting

### Error: "Illegal argument in isolate message"
**Cause:** Trying to send non-serializable objects (like `ui.Image`)  
**Solution:** Use `Uint8List` (bytes) instead

### Error: "ONNX session not initialized"
**Cause:** Forgot to call `initializeOrt()` inside the isolate  
**Solution:** Ensure initialization happens in the isolate function

### Performance is worse with isolates
**Cause:** Overhead of isolate creation + model loading  
**Solution:** Use direct method for small images or implement isolate pooling

---


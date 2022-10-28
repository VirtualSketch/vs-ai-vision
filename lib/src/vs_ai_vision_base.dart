import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

typedef CTrain = ffi.Void Function();
typedef DTrain = void Function();
typedef CPredict = ffi.Pointer<Utf8> Function(ffi.Pointer<ffi.Uint8>, ffi.Int, ffi.Int);
typedef DPredict = ffi.Pointer<Utf8> Function(ffi.Pointer<ffi.Uint8>, int, int);
typedef CPredictSample = ffi.Pointer<Utf8> Function();
typedef DPredictSample = ffi.Pointer<Utf8> Function();

class SVMFunctions {
  final svmUtilsLib = ffi.DynamicLibrary.open("libsvmutils.so");

  void train() {
    final _train = svmUtilsLib.lookupFunction<CTrain, DTrain>('train');
    _train();
  }

  ffi.Pointer<Utf8> predictSample() {
    final _predictSample = svmUtilsLib.lookupFunction<CPredictSample, DPredictSample>('predictSample');
    return _predictSample();
  }

  ffi.Pointer<Utf8> predict(ffi.Pointer<ffi.Uint8> imageList, int rows, int cols) {
    final _predict = svmUtilsLib.lookupFunction<CPredict, DPredict>('predict');
    return _predict(imageList, rows, cols);
  }
}

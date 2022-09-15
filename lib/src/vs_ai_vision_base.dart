import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

typedef CTrain = ffi.Void Function();
typedef DTrain = void Function();
typedef CPredict = ffi.Pointer<Utf8> Function();
typedef DPredict = ffi.Pointer<Utf8> Function();

class SVMFunctions {
  final svmUtilsLib = ffi.DynamicLibrary.open("libsvmutils.so");

  void train() {
    final _train = svmUtilsLib.lookupFunction<CTrain, DTrain>('train');
    _train();
  }

  ffi.Pointer<Utf8> predict() {
    final _predict = svmUtilsLib.lookupFunction<CPredict, DPredict>('predictSample');
    return _predict();
  }
}
import 'dart:ffi' as ffi;

typedef CTrain = ffi.Void Function();
typedef DTrain = void Function();
typedef CPredict = ffi.Void Function();
typedef DPredict = void Function();

class SVMFunctions {
  final trainLib = ffi.DynamicLibrary.open("src/svm/build/libtrain.so");
  final predictLib = ffi.DynamicLibrary.open("src/svm/build/libpredict.so");

  void train() {
    final _train = trainLib.lookupFunction<CTrain, DTrain>('train');
    _train();
  }

  void predict() {
    final _predict = predictLib.lookupFunction<CPredict, DPredict>('predict');
    _predict();
  }
}

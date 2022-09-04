import 'dart:ffi' as ffi;
import 'dart:io';

typedef CTrain = ffi.Void Function();
typedef DTrain = void Function();
typedef CPredict = ffi.Void Function();
typedef DPredict = void Function();

class SVMFunctions {
  void test() {
    print(Platform.script.toFilePath());
  }

//  final trainLib = ffi.DynamicLibrary.open("svm/build/libtrain.so");
//  final predictLib = ffi.DynamicLibrary.open("svm/build/libpredict.so");
//
//  void train() {
//    final _train = trainLib.lookupFunction<CTrain, DTrain>('train');
//    _train();
//  }
//
//  void predict() {
//    final _predict = predictLib.lookupFunction<CPredict, DPredict>('predict');
//    _predict();
//  }
}

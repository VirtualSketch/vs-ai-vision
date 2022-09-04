# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android

# Include any dependencies generated for this target.
include CMakeFiles/predict.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/predict.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/predict.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/predict.dir/flags.make

CMakeFiles/predict.dir/predict.cpp.o: CMakeFiles/predict.dir/flags.make
CMakeFiles/predict.dir/predict.cpp.o: /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/predict.cpp
CMakeFiles/predict.dir/predict.cpp.o: CMakeFiles/predict.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/predict.dir/predict.cpp.o"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/predict.dir/predict.cpp.o -MF CMakeFiles/predict.dir/predict.cpp.o.d -o CMakeFiles/predict.dir/predict.cpp.o -c /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/predict.cpp

CMakeFiles/predict.dir/predict.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/predict.dir/predict.cpp.i"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/predict.cpp > CMakeFiles/predict.dir/predict.cpp.i

CMakeFiles/predict.dir/predict.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/predict.dir/predict.cpp.s"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/predict.cpp -o CMakeFiles/predict.dir/predict.cpp.s

# Object files for target predict
predict_OBJECTS = \
"CMakeFiles/predict.dir/predict.cpp.o"

# External object files for target predict
predict_EXTERNAL_OBJECTS =

libpredict.so: CMakeFiles/predict.dir/predict.cpp.o
libpredict.so: CMakeFiles/predict.dir/build.make
libpredict.so: libsvmutils.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_gapi.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_highgui.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_ml.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_objdetect.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_photo.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_stitching.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_video.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_calib3d.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_dnn.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_features2d.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_flann.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_videoio.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_imgcodecs.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_imgproc.so
libpredict.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_core.so
libpredict.so: CMakeFiles/predict.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libpredict.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/predict.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/predict.dir/build: libpredict.so
.PHONY : CMakeFiles/predict.dir/build

CMakeFiles/predict.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/predict.dir/cmake_clean.cmake
.PHONY : CMakeFiles/predict.dir/clean

CMakeFiles/predict.dir/depend:
	cd /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android/CMakeFiles/predict.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/predict.dir/depend

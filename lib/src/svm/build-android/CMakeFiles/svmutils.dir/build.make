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
include CMakeFiles/svmutils.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/svmutils.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/svmutils.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/svmutils.dir/flags.make

CMakeFiles/svmutils.dir/utils/svm.cpp.o: CMakeFiles/svmutils.dir/flags.make
CMakeFiles/svmutils.dir/utils/svm.cpp.o: /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/svm.cpp
CMakeFiles/svmutils.dir/utils/svm.cpp.o: CMakeFiles/svmutils.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/svmutils.dir/utils/svm.cpp.o"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/svmutils.dir/utils/svm.cpp.o -MF CMakeFiles/svmutils.dir/utils/svm.cpp.o.d -o CMakeFiles/svmutils.dir/utils/svm.cpp.o -c /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/svm.cpp

CMakeFiles/svmutils.dir/utils/svm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svmutils.dir/utils/svm.cpp.i"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/svm.cpp > CMakeFiles/svmutils.dir/utils/svm.cpp.i

CMakeFiles/svmutils.dir/utils/svm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svmutils.dir/utils/svm.cpp.s"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/svm.cpp -o CMakeFiles/svmutils.dir/utils/svm.cpp.s

CMakeFiles/svmutils.dir/utils/labelassigner.cpp.o: CMakeFiles/svmutils.dir/flags.make
CMakeFiles/svmutils.dir/utils/labelassigner.cpp.o: /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/labelassigner.cpp
CMakeFiles/svmutils.dir/utils/labelassigner.cpp.o: CMakeFiles/svmutils.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/svmutils.dir/utils/labelassigner.cpp.o"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/svmutils.dir/utils/labelassigner.cpp.o -MF CMakeFiles/svmutils.dir/utils/labelassigner.cpp.o.d -o CMakeFiles/svmutils.dir/utils/labelassigner.cpp.o -c /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/labelassigner.cpp

CMakeFiles/svmutils.dir/utils/labelassigner.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svmutils.dir/utils/labelassigner.cpp.i"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/labelassigner.cpp > CMakeFiles/svmutils.dir/utils/labelassigner.cpp.i

CMakeFiles/svmutils.dir/utils/labelassigner.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svmutils.dir/utils/labelassigner.cpp.s"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/labelassigner.cpp -o CMakeFiles/svmutils.dir/utils/labelassigner.cpp.s

CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.o: CMakeFiles/svmutils.dir/flags.make
CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.o: /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/descriptordeterminer.cpp
CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.o: CMakeFiles/svmutils.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.o"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.o -MF CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.o.d -o CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.o -c /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/descriptordeterminer.cpp

CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.i"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/descriptordeterminer.cpp > CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.i

CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.s"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/descriptordeterminer.cpp -o CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.s

CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.o: CMakeFiles/svmutils.dir/flags.make
CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.o: /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/rectangledeterminer.cpp
CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.o: CMakeFiles/svmutils.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.o"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.o -MF CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.o.d -o CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.o -c /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/rectangledeterminer.cpp

CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.i"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/rectangledeterminer.cpp > CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.i

CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.s"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/rectangledeterminer.cpp -o CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.s

CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.o: CMakeFiles/svmutils.dir/flags.make
CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.o: /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/imagepreprocessing.cpp
CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.o: CMakeFiles/svmutils.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.o"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.o -MF CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.o.d -o CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.o -c /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/imagepreprocessing.cpp

CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.i"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/imagepreprocessing.cpp > CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.i

CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.s"
	/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/atikin/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/utils/imagepreprocessing.cpp -o CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.s

# Object files for target svmutils
svmutils_OBJECTS = \
"CMakeFiles/svmutils.dir/utils/svm.cpp.o" \
"CMakeFiles/svmutils.dir/utils/labelassigner.cpp.o" \
"CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.o" \
"CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.o" \
"CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.o"

# External object files for target svmutils
svmutils_EXTERNAL_OBJECTS =

libsvmutils.so: CMakeFiles/svmutils.dir/utils/svm.cpp.o
libsvmutils.so: CMakeFiles/svmutils.dir/utils/labelassigner.cpp.o
libsvmutils.so: CMakeFiles/svmutils.dir/utils/descriptordeterminer.cpp.o
libsvmutils.so: CMakeFiles/svmutils.dir/utils/rectangledeterminer.cpp.o
libsvmutils.so: CMakeFiles/svmutils.dir/utils/imagepreprocessing.cpp.o
libsvmutils.so: CMakeFiles/svmutils.dir/build.make
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_gapi.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_highgui.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_ml.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_objdetect.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_photo.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_stitching.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_video.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_videoio.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_imgcodecs.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_dnn.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_calib3d.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_features2d.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_flann.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_imgproc.so
libsvmutils.so: /home/atikin/Downloads/opencv-android/build-android/lib/arm64-v8a/libopencv_core.so
libsvmutils.so: CMakeFiles/svmutils.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX shared library libsvmutils.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/svmutils.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/svmutils.dir/build: libsvmutils.so
.PHONY : CMakeFiles/svmutils.dir/build

CMakeFiles/svmutils.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/svmutils.dir/cmake_clean.cmake
.PHONY : CMakeFiles/svmutils.dir/clean

CMakeFiles/svmutils.dir/depend:
	cd /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android /home/atikin/Documents/Programming/VirtualSketch/vs-ai-vision/lib/src/svm/build-android/CMakeFiles/svmutils.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/svmutils.dir/depend


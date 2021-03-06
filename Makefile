# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cpslab/yolov3cplusplus

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cpslab/yolov3cplusplus

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/cpslab/yolov3cplusplus/CMakeFiles /home/cpslab/yolov3cplusplus/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/cpslab/yolov3cplusplus/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named darknet

# Build rule for target.
darknet: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 darknet
.PHONY : darknet

# fast build rule for target.
darknet/fast:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/build
.PHONY : darknet/fast

src/convLayer.o: src/convLayer.cpp.o

.PHONY : src/convLayer.o

# target to build an object file
src/convLayer.cpp.o:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/convLayer.cpp.o
.PHONY : src/convLayer.cpp.o

src/convLayer.i: src/convLayer.cpp.i

.PHONY : src/convLayer.i

# target to preprocess a source file
src/convLayer.cpp.i:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/convLayer.cpp.i
.PHONY : src/convLayer.cpp.i

src/convLayer.s: src/convLayer.cpp.s

.PHONY : src/convLayer.s

# target to generate assembly for a file
src/convLayer.cpp.s:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/convLayer.cpp.s
.PHONY : src/convLayer.cpp.s

src/convOutputLayer.o: src/convOutputLayer.cpp.o

.PHONY : src/convOutputLayer.o

# target to build an object file
src/convOutputLayer.cpp.o:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/convOutputLayer.cpp.o
.PHONY : src/convOutputLayer.cpp.o

src/convOutputLayer.i: src/convOutputLayer.cpp.i

.PHONY : src/convOutputLayer.i

# target to preprocess a source file
src/convOutputLayer.cpp.i:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/convOutputLayer.cpp.i
.PHONY : src/convOutputLayer.cpp.i

src/convOutputLayer.s: src/convOutputLayer.cpp.s

.PHONY : src/convOutputLayer.s

# target to generate assembly for a file
src/convOutputLayer.cpp.s:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/convOutputLayer.cpp.s
.PHONY : src/convOutputLayer.cpp.s

src/createModel.o: src/createModel.cpp.o

.PHONY : src/createModel.o

# target to build an object file
src/createModel.cpp.o:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/createModel.cpp.o
.PHONY : src/createModel.cpp.o

src/createModel.i: src/createModel.cpp.i

.PHONY : src/createModel.i

# target to preprocess a source file
src/createModel.cpp.i:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/createModel.cpp.i
.PHONY : src/createModel.cpp.i

src/createModel.s: src/createModel.cpp.s

.PHONY : src/createModel.s

# target to generate assembly for a file
src/createModel.cpp.s:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/createModel.cpp.s
.PHONY : src/createModel.cpp.s

src/main.o: src/main.cpp.o

.PHONY : src/main.o

# target to build an object file
src/main.cpp.o:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/main.cpp.o
.PHONY : src/main.cpp.o

src/main.i: src/main.cpp.i

.PHONY : src/main.i

# target to preprocess a source file
src/main.cpp.i:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/main.cpp.i
.PHONY : src/main.cpp.i

src/main.s: src/main.cpp.s

.PHONY : src/main.s

# target to generate assembly for a file
src/main.cpp.s:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/main.cpp.s
.PHONY : src/main.cpp.s

src/routeLayer.o: src/routeLayer.cpp.o

.PHONY : src/routeLayer.o

# target to build an object file
src/routeLayer.cpp.o:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/routeLayer.cpp.o
.PHONY : src/routeLayer.cpp.o

src/routeLayer.i: src/routeLayer.cpp.i

.PHONY : src/routeLayer.i

# target to preprocess a source file
src/routeLayer.cpp.i:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/routeLayer.cpp.i
.PHONY : src/routeLayer.cpp.i

src/routeLayer.s: src/routeLayer.cpp.s

.PHONY : src/routeLayer.s

# target to generate assembly for a file
src/routeLayer.cpp.s:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/routeLayer.cpp.s
.PHONY : src/routeLayer.cpp.s

src/routeLayerConcat.o: src/routeLayerConcat.cpp.o

.PHONY : src/routeLayerConcat.o

# target to build an object file
src/routeLayerConcat.cpp.o:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/routeLayerConcat.cpp.o
.PHONY : src/routeLayerConcat.cpp.o

src/routeLayerConcat.i: src/routeLayerConcat.cpp.i

.PHONY : src/routeLayerConcat.i

# target to preprocess a source file
src/routeLayerConcat.cpp.i:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/routeLayerConcat.cpp.i
.PHONY : src/routeLayerConcat.cpp.i

src/routeLayerConcat.s: src/routeLayerConcat.cpp.s

.PHONY : src/routeLayerConcat.s

# target to generate assembly for a file
src/routeLayerConcat.cpp.s:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/routeLayerConcat.cpp.s
.PHONY : src/routeLayerConcat.cpp.s

src/shortCutLayer.o: src/shortCutLayer.cpp.o

.PHONY : src/shortCutLayer.o

# target to build an object file
src/shortCutLayer.cpp.o:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/shortCutLayer.cpp.o
.PHONY : src/shortCutLayer.cpp.o

src/shortCutLayer.i: src/shortCutLayer.cpp.i

.PHONY : src/shortCutLayer.i

# target to preprocess a source file
src/shortCutLayer.cpp.i:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/shortCutLayer.cpp.i
.PHONY : src/shortCutLayer.cpp.i

src/shortCutLayer.s: src/shortCutLayer.cpp.s

.PHONY : src/shortCutLayer.s

# target to generate assembly for a file
src/shortCutLayer.cpp.s:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/shortCutLayer.cpp.s
.PHONY : src/shortCutLayer.cpp.s

src/upsampleLayer.o: src/upsampleLayer.cpp.o

.PHONY : src/upsampleLayer.o

# target to build an object file
src/upsampleLayer.cpp.o:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/upsampleLayer.cpp.o
.PHONY : src/upsampleLayer.cpp.o

src/upsampleLayer.i: src/upsampleLayer.cpp.i

.PHONY : src/upsampleLayer.i

# target to preprocess a source file
src/upsampleLayer.cpp.i:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/upsampleLayer.cpp.i
.PHONY : src/upsampleLayer.cpp.i

src/upsampleLayer.s: src/upsampleLayer.cpp.s

.PHONY : src/upsampleLayer.s

# target to generate assembly for a file
src/upsampleLayer.cpp.s:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/upsampleLayer.cpp.s
.PHONY : src/upsampleLayer.cpp.s

src/utils.o: src/utils.cpp.o

.PHONY : src/utils.o

# target to build an object file
src/utils.cpp.o:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/utils.cpp.o
.PHONY : src/utils.cpp.o

src/utils.i: src/utils.cpp.i

.PHONY : src/utils.i

# target to preprocess a source file
src/utils.cpp.i:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/utils.cpp.i
.PHONY : src/utils.cpp.i

src/utils.s: src/utils.cpp.s

.PHONY : src/utils.s

# target to generate assembly for a file
src/utils.cpp.s:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/utils.cpp.s
.PHONY : src/utils.cpp.s

src/yoloLayer.o: src/yoloLayer.cpp.o

.PHONY : src/yoloLayer.o

# target to build an object file
src/yoloLayer.cpp.o:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/yoloLayer.cpp.o
.PHONY : src/yoloLayer.cpp.o

src/yoloLayer.i: src/yoloLayer.cpp.i

.PHONY : src/yoloLayer.i

# target to preprocess a source file
src/yoloLayer.cpp.i:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/yoloLayer.cpp.i
.PHONY : src/yoloLayer.cpp.i

src/yoloLayer.s: src/yoloLayer.cpp.s

.PHONY : src/yoloLayer.s

# target to generate assembly for a file
src/yoloLayer.cpp.s:
	$(MAKE) -f CMakeFiles/darknet.dir/build.make CMakeFiles/darknet.dir/src/yoloLayer.cpp.s
.PHONY : src/yoloLayer.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... darknet"
	@echo "... src/convLayer.o"
	@echo "... src/convLayer.i"
	@echo "... src/convLayer.s"
	@echo "... src/convOutputLayer.o"
	@echo "... src/convOutputLayer.i"
	@echo "... src/convOutputLayer.s"
	@echo "... src/createModel.o"
	@echo "... src/createModel.i"
	@echo "... src/createModel.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
	@echo "... src/routeLayer.o"
	@echo "... src/routeLayer.i"
	@echo "... src/routeLayer.s"
	@echo "... src/routeLayerConcat.o"
	@echo "... src/routeLayerConcat.i"
	@echo "... src/routeLayerConcat.s"
	@echo "... src/shortCutLayer.o"
	@echo "... src/shortCutLayer.i"
	@echo "... src/shortCutLayer.s"
	@echo "... src/upsampleLayer.o"
	@echo "... src/upsampleLayer.i"
	@echo "... src/upsampleLayer.s"
	@echo "... src/utils.o"
	@echo "... src/utils.i"
	@echo "... src/utils.s"
	@echo "... src/yoloLayer.o"
	@echo "... src/yoloLayer.i"
	@echo "... src/yoloLayer.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system


@echo off
rem Build a shared transcribe.dll (+ CPU-only ggml) from a transcribe.cpp 0.4.1 source tree.
rem Flags: the official wheel-windows-cpu-vulkan lane (pyproject cpu-vulkan + SKBUILD
rem posture) without Vulkan. Only transcribe.dll is used; the official ggml DLLs stay.
rem usage: build.cmd <source-dir> <build-dir>
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cmake -S "%~1" -B "%~2" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DTRANSCRIBE_BUILD_SHARED=ON ^
  -DTRANSCRIBE_BUILD_TESTS=OFF ^
  -DTRANSCRIBE_BUILD_EXAMPLES=OFF ^
  -DTRANSCRIBE_BUILD_TOOLS=OFF ^
  -DTRANSCRIBE_USE_OPENMP=OFF ^
  -DTRANSCRIBE_USE_SYSTEM_BLAS=OFF ^
  -DTRANSCRIBE_GGML_BACKEND_DL=ON ^
  -DTRANSCRIBE_VULKAN=OFF ^
  -DGGML_CPU_ALL_VARIANTS=ON ^
  -DTRANSCRIBE_X86_CONSERVATIVE=ON ^
  -DGGML_NATIVE=OFF || exit /b 1
cmake --build "%~2" --target transcribe || exit /b 1

# CMake generated Testfile for 
# Source directory: C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt
# Build directory: C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(pbrt_unit_test "pbrt_test")
set_tests_properties(pbrt_unit_test PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/CMakeLists.txt;559;ADD_TEST;C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/CMakeLists.txt;0;")
subdirs("src/ext/zlib")
subdirs("src/ext/openexr")
subdirs("src/ext/glog")
subdirs("src/ext/ptex")

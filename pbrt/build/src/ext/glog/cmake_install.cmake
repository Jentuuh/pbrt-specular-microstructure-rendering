# Install script for directory: C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/src/ext/glog

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/PBRT-V3")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/Debug/glog.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/Release/glog.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/MinSizeRel/glog.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/RelWithDebInfo/glog.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/glog" TYPE FILE FILES
    "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/config.h"
    "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/glog/logging.h"
    "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/glog/raw_logging.h"
    "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/glog/stl_logging.h"
    "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/glog/vlog_is_on.h"
    "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/src/ext/glog/src/glog/log_severity.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glog" TYPE FILE FILES
    "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/CMakeFiles/glog-config.cmake"
    "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/glog-config-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glog/glog-targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glog/glog-targets.cmake"
         "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/CMakeFiles/Export/3b58dfe2d8c90c3a489d71c8991c4dd3/glog-targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glog/glog-targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glog/glog-targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glog" TYPE FILE FILES "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/CMakeFiles/Export/3b58dfe2d8c90c3a489d71c8991c4dd3/glog-targets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glog" TYPE FILE FILES "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/CMakeFiles/Export/3b58dfe2d8c90c3a489d71c8991c4dd3/glog-targets-debug.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glog" TYPE FILE FILES "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/CMakeFiles/Export/3b58dfe2d8c90c3a489d71c8991c4dd3/glog-targets-minsizerel.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glog" TYPE FILE FILES "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/CMakeFiles/Export/3b58dfe2d8c90c3a489d71c8991c4dd3/glog-targets-relwithdebinfo.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glog" TYPE FILE FILES "C:/Users/jente/Desktop/University/pbrt-project/pbrt-specular-microstructure-rendering/pbrt/build/src/ext/glog/CMakeFiles/Export/3b58dfe2d8c90c3a489d71c8991c4dd3/glog-targets-release.cmake")
  endif()
endif()


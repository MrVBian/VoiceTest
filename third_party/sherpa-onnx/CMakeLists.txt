# third_party/CMakeLists.txt

# 根据目标平台选择库路径
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_GENERATOR_PLATFORM STREQUAL "x64")
    set(SHERPA_ONNX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/win-x64/v1.12.0")
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# 创建导入库目标
add_library(sherpa-onnx-c-api SHARED IMPORTED GLOBAL)
add_library(sherpa-onnx-cxx-api SHARED IMPORTED GLOBAL)
add_library(cargs STATIC IMPORTED GLOBAL)

# 设置目标属性
set_target_properties(sherpa-onnx-c-api PROPERTIES
    IMPORTED_IMPLIB "${SHERPA_ONNX_ROOT}/lib/sherpa-onnx-c-api.lib"
    IMPORTED_LOCATION "${SHERPA_ONNX_ROOT}/lib/sherpa-onnx-c-api.dll"
    INTERFACE_INCLUDE_DIRECTORIES
        "${SHERPA_ONNX_ROOT}/include;${SHERPA_ONNX_ROOT}/include/sherpa-onnx/c-api"
)

set_target_properties(sherpa-onnx-cxx-api PROPERTIES
    IMPORTED_IMPLIB "${SHERPA_ONNX_ROOT}/lib/sherpa-onnx-cxx-api.lib"
    IMPORTED_LOCATION "${SHERPA_ONNX_ROOT}/lib/sherpa-onnx-cxx-api.dll"
)

set_target_properties(cargs PROPERTIES
    IMPORTED_LOCATION "${SHERPA_ONNX_ROOT}/lib/cargs.lib"
)

# 创建组合目标
# 使用一个别名目标来统一引用所有需要的库
add_library(sherpa-onnx ALIAS sherpa-onnx-c-api)

# 设置库之间的依赖关系
target_link_libraries(sherpa-onnx-c-api INTERFACE
    sherpa-onnx-cxx-api
    cargs
)

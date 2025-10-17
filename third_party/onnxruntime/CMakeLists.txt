# third_party/onnxruntime/CMakeLists.txt

# 根据目标平台选择库路径
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64" OR CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/win-arm64/v1.22.1")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_GENERATOR_PLATFORM STREQUAL "x64")
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/win-x64/v1.22.1")
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# 创建导入库目标
add_library(onnxruntime SHARED IMPORTED GLOBAL)

# 设置目标属性
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)
set(SOURCES ir.cc kernels/matmul.cc)

add_library(peachy_ir_agent ${SOURCES})
target_include_directories(peachy_ir_agent PUBLIC ${PROJECT_SOURCE_DIR}/include)

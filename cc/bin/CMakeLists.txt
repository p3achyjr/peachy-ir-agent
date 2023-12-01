function(create_executable TARGET_NAME)
  set(INCLUDE_PATHS ${PROJECT_SOURCE_DIR}/include)
  add_executable(${TARGET_NAME} ${TARGET_NAME}_main.cc)
  target_link_libraries(${TARGET_NAME} peachy_ir_agent)
  target_include_directories(${TARGET_NAME} PRIVATE ${INCLUDE_PATHS})
endfunction()

create_executable(random_search)

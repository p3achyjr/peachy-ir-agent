function(create_test TARGET_NAME)
  set(TEST_INCLUDE_PATHS ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/deps/doctest)
  add_executable(${TARGET_NAME} ${TARGET_NAME}.cc)
  target_link_libraries(${TARGET_NAME} peachy_ir_agent)
  target_include_directories(${TARGET_NAME} PRIVATE ${TEST_INCLUDE_PATHS})
  add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endfunction()

create_test(ir_test)
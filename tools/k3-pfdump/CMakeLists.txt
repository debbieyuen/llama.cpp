set(TARGET llama-k3-pfdump)
add_executable(${TARGET} k3-pfdump.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

cmake_minimum_required(VERSION 3.8)
project(f1tenth_common)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Header-only library exposing the C++ mirror of the observation layout.
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip the copyright linter (it requires a full recognized license block, which
  # we do not enforce repo-wide). cpplint/uncrustify/cppcheck still run.
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # Parity test: asserts the C++ observation mirror matches the Python contract
  # fixtures. Keeps the two layouts honest at deploy time.
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_obs_parity test/test_obs_parity.cpp)
  if(TARGET test_obs_parity)
    target_link_libraries(test_obs_parity ${PROJECT_NAME})
  endif()
endif()

ament_package()

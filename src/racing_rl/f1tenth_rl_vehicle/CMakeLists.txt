cmake_minimum_required(VERSION 3.8)
project(f1tenth_rl_vehicle)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# --- core observation library -------------------------------------------------
add_library(rl_obs_core src/rl_obs_core.cpp)
target_include_directories(rl_obs_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# --- nodes --------------------------------------------------------------------
add_executable(vehicle_obs src/vehicle_obs_node.cpp)
target_link_libraries(vehicle_obs rl_obs_core)
ament_target_dependencies(vehicle_obs rclcpp nav_msgs std_msgs sensor_msgs)

add_executable(drive src/drive_node.cpp)
target_link_libraries(drive rl_obs_core)
ament_target_dependencies(drive rclcpp ackermann_msgs std_msgs)

add_executable(opponent_detector src/opponent_detector_node.cpp)
target_link_libraries(opponent_detector rl_obs_core)
ament_target_dependencies(opponent_detector rclcpp nav_msgs sensor_msgs visualization_msgs)

install(TARGETS vehicle_obs drive opponent_detector
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)

# --- tests --------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_rl_obs_core test/test_rl_obs_core.cpp)
  target_link_libraries(test_rl_obs_core rl_obs_core)
  target_compile_definitions(test_rl_obs_core PRIVATE
    OBS_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test/obs_fixture.txt"
    GYM_SCAN_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test/gym_scan_fixture.json"
    IV2026_TRACK_CSV_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../f1tenth_rl_agent/assets/IV_2026_SIM_centerline.csv")
endif()

ament_package()

cmake_minimum_required(VERSION 3.8)
project(f1tenth_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Observation.msg"
  "msg/Action.msg"
  "msg/ActuatorCommand.msg"
  DEPENDENCIES builtin_interfaces std_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  find_package(rclcpp REQUIRED)
  ament_add_gtest(test_actuator_command_serialization test/test_actuator_command.cpp)
  if(TARGET test_actuator_command_serialization)
    target_include_directories(test_actuator_command_serialization PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp>)
    ament_target_dependencies(test_actuator_command_serialization rclcpp)
    rosidl_get_typesupport_target(
      cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
    target_link_libraries(test_actuator_command_serialization "${cpp_typesupport_target}")
  endif()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_actuator_command_py test/test_actuator_command.py)
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()

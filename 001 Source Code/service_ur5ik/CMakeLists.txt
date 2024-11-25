cmake_minimum_required(VERSION 3.8)
project(service_ur5ik)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interfaces_ur5ik REQUIRED)

find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

add_executable(target_client src/target_client.cpp)
ament_target_dependencies(
  target_client
  "rclcpp"
  "std_msgs"
  "interfaces_ur5ik"
)
target_include_directories(target_client PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(target_client PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

add_executable(calc_server src/calc_server.cpp)
ament_target_dependencies(
  calc_server
  "rclcpp"
  "std_msgs"
  "interfaces_ur5ik"
)
target_include_directories(calc_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(calc_server PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS
  target_client
  calc_server
  DESTINATION lib/${PROJECT_NAME})

#python module
ament_python_install_package(${PROJECT_NAME})
install (PROGRAMS
  scripts/target_client_py.py
  scripts/calc_server_py.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

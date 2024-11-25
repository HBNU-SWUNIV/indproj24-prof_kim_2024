#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <stdio.h>
#include <iostream>
#include <typeinfo>

#include <memory>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/bool.hpp"

#include "rclcpp/rclcpp.hpp"
/* 추가한 부분 시작 */
#include "interfaces_ur5ik/srv/six_theta.hpp"
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <vector>
/* 추가한 부분 끝*/
using std::placeholders::_1;

using namespace std::chrono_literals;

using moveit::planning_interface::MoveGroupInterface;
/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

#define camera 1
double theta[6] = { 0 };
double pi = 3.141592;

class RobotMasterController : public rclcpp::Node
{
  public:
    RobotMasterController(std::shared_ptr<rclcpp::Node> move_group_node)
    : Node("main_node"), is_horizontally_centered(false),
    is_vertically_centered(false), is_moving(false), prev_x(0), prev_y(0),
    is_depth_reached(false), was_centered_message_shown(false), depth(0.0),
    is_task_completed(false)
    {
      RCLCPP_INFO(this->get_logger(), "Node started. Awaiting commands...");
      RCLCPP_INFO(this->get_logger(), "=======================================================");
      RCLCPP_INFO(this->get_logger(), "SETUP LOGS");
      RCLCPP_INFO(this->get_logger(), "=======================================================");

/* 추가한 부분 시작 */
      subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "selected_coordinates", 1, std::bind(&RobotMasterController::topic_callback, this, _1));
      RCLCPP_INFO(this->get_logger(), "Subscribed to /custom_camera topic");
      publisher_ = this->create_publisher<std_msgs::msg::Bool>("task_completed", 1);
      RCLCPP_INFO(this->get_logger(), "Publisher for task completion created"); 
      service_client_ = this->create_client<interfaces_ur5ik::srv::SixTheta>("service_ik");
      RCLCPP_INFO(this->get_logger(), "Service client ur5 inverse kinematics");
/* 추가한 부분 끝*/
      move_group_ = new moveit::planning_interface::MoveGroupInterface(move_group_node, "ur_manipulator");
      move_planning_scene_ = new moveit::planning_interface::PlanningSceneInterface();
      make_Collision();
/* 추가한 부분 시작 */
      send_target(0.0, -0.6, 0.3, pi, 0.0, 0.0);

/* 추가한 부분 끝*/

#if 0
      auto const robot_pos = move_group_->getCurrentPose("wrist_3_link");
      RCLCPP_INFO(this->get_logger(), "Robot position: %f, %f, %f", robot_pos.pose.position.x, robot_pos.pose.position.y, robot_pos.pose.position.z);
      RCLCPP_INFO(this->get_logger(), "Robot rotation: %f, %f, %f, %f", 
      robot_pos.pose.orientation.x, robot_pos.pose.orientation.y, robot_pos.pose.orientation.z, robot_pos.pose.orientation.w
      );
#endif
    }

  private:
    void send_target(double x, double y, double z,
      double roll, double pitch, double yaw)
    {
      auto request = std::make_shared<interfaces_ur5ik::srv::SixTheta::Request>();
      request -> srv_target[0] = x; //0.0
      request -> srv_target[1] = y; //0.6
      request -> srv_target[2] = z; //0.3
      request -> srv_target[3] = roll; //pi
      request -> srv_target[4] = pitch; //0.0
      request -> srv_target[5] = yaw; //0.0
      RCLCPP_INFO(this -> get_logger(), "set target : %f, %f, %f, %f, %f, %f",
           request -> srv_target[0], request -> srv_target[1], request -> srv_target[2],
           request -> srv_target[3], request -> srv_target[4], request -> srv_target[5]);

      while (!service_client_ -> wait_for_service(1s))
      {
        if(!rclcpp::ok())
        {
          RCLCPP_ERROR(this -> get_logger(), "Interrupted while waiting for the service. Exitiong.");
          return;
        }
        RCLCPP_INFO(this -> get_logger(),"service not available, waiting again...");
      }
    

      auto result = service_client_->async_send_request(request, std::bind(&RobotMasterController::handle_response, this, std::placeholders::_1));
    }

    void handle_response(rclcpp::Client<interfaces_ur5ik::srv::SixTheta>::SharedFuture future) {
      auto response = future.get();
        // 처리할 로직
        theta[0] = response -> srv_theta[0];
        theta[1] = response -> srv_theta[1];
        theta[2] = response -> srv_theta[2];
        theta[3] = response -> srv_theta[3];
        theta[4] = response -> srv_theta[4];
        theta[5] = response -> srv_theta[5];
        RCLCPP_INFO(this->get_logger(), "response_callback theta : %f, %f, %f, %f, %f, %f",
            theta[0], theta[1], theta[2],
            theta[3], theta[4], theta[5]);

      this -> move_this_position(
        theta[0], theta[1], theta[2], theta[3], theta[4], theta[5],
        "initial_position");
    }

    void move_this_position(double theta1, double theta2, double theta3,
      double theta4, double theta5, double theta6, const std::string& target)
    {
      RCLCPP_INFO(this->get_logger(), "=======================================================");
      auto current_state = move_group_ -> getCurrentState();
      const moveit::core::JointModelGroup *joint_model_group_arm =
        current_state->getJointModelGroup("ur_manipulator");

      move_group_ -> setEndEffectorLink("wrist_3_link");

      for (auto name : move_group_ -> getLinkNames())
      {
        RCLCPP_INFO(get_logger(), "Link: %s", name.c_str());
      }

      moveit::core::RobotStatePtr current_state_arm =
        move_group_->getCurrentState(10);

      std::vector<double> joint_group_positions_arm;
      current_state_arm->copyJointGroupPositions(joint_model_group_arm,
                                                   joint_group_positions_arm);

      move_group_->setStartStateToCurrentState();

      joint_group_positions_arm[0] = theta1;      // Base
      joint_group_positions_arm[1] = theta2; // Shoulder
      joint_group_positions_arm[2] = theta3;      // Elbow
      joint_group_positions_arm[3] = theta4; // Wrist 1
      joint_group_positions_arm[4] = theta5;      // Wrist 2
      joint_group_positions_arm[5] = theta6;      // Wrist 3

      move_group_->setJointValueTarget(joint_group_positions_arm);

      moveit::planning_interface::MoveGroupInterface::Plan my_plan;
      move_group_ -> setMaxVelocityScalingFactor(0.5);
      move_group_ -> setMaxAccelerationScalingFactor(0.5);
      move_group_->setNamedTarget(target);
      bool success = (move_group_->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
      if (success) {
        move_group_->execute(my_plan);
        move_group_->setStartStateToCurrentState();
        RCLCPP_INFO(this->get_logger(), "Movement to target completed.");
        send_task_completion_status(true);
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to move to the target");
      }

    }

    void topic_callback(const std_msgs::msg::Float64MultiArray msg)
    {
#if camera
      double x = msg.data[0];
      double y = msg.data[1];

      if (fabs(x - prev_x) > 0.001 && fabs(y - prev_y) > 0.01) {
        send_target(x, y, 0.3, pi, 0.0, 0.0);
      } else if (fabs(x - prev_x) > 0.01) {
        move_along_axis("x", x);
      } else if (fabs(y - prev_y) > 0.01) {
        move_along_axis("y", y);
      }
      prev_x = x;
      prev_y = y;

#endif
    }

    void move_along_axis(const std::string& axis, double value)
    {
      moveit::planning_interface::MoveGroupInterface::Plan my_plan;
      move_group_ -> setMaxVelocityScalingFactor(0.5);
      move_group_ -> setMaxAccelerationScalingFactor(0.5);
      move_group_->setStartStateToCurrentState();

      if (axis == "x") {
        target_pose.position.x = value;
      } else if (axis == "y") {
        target_pose.position.z = value;
      }

      move_group_->setPoseTarget(target_pose);

      bool success = (move_group_->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
      if (success) {
        move_group_->execute(my_plan);
        RCLCPP_INFO(this->get_logger(), "Moved along %s axis to value: %f", axis.c_str(), value);
        send_task_completion_status(true);
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to move along %s axis", axis.c_str());
      }
    }

    void send_task_completion_status(bool status)
    {
      auto msg = std_msgs::msg::Bool();
      msg.data = status;
      publisher_->publish(msg);
      RCLCPP_INFO(this->get_logger(), "Task completion status published: %s", status ? "true" : "false");
    }

    void make_Collision()
    {
        // Make collosion to appropriate path planning start
        std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
        collision_objects.resize(1);
#if 0
        // Add the first table where the cube will originally be kept.
        collision_objects[0].id = "back";
        collision_objects[0].header.frame_id = "base_link";

        //Define the primitive and its dimensions.
        collision_objects[0].primitives.resize(1);
        collision_objects[0].primitives[0].type = collision_objects[0].primitives[0].BOX;
        collision_objects[0].primitives[0].dimensions.resize(3);
        collision_objects[0].primitives[0].dimensions[0] = 1.0;
        collision_objects[0].primitives[0].dimensions[1] = 0.001;
        collision_objects[0].primitives[0].dimensions[2] = 0.75;

        //Define the pose of the table.
        collision_objects[0].primitive_poses.resize(1);
        collision_objects[0].primitive_poses[0].position.x = 0.0;
        collision_objects[0].primitive_poses[0].position.y = -0.3;
        collision_objects[0].primitive_poses[0].position.z = 0.375;
        collision_objects[0].primitive_poses[0].orientation.w = 1.0;
        // END_SUB_TUTORIAL

        collision_objects[0].operation = collision_objects[0].ADD;

        // BEGIN_SUB_TUTORIAL table2
        // Add the second table where we will be placing the cube.
        collision_objects[1].id = "right";
        collision_objects[1].header.frame_id = "base_link";

        //Define the primitive and its dimensions.
        collision_objects[1].primitives.resize(1);
        collision_objects[1].primitives[0].type = collision_objects[1].primitives[0].BOX;
        collision_objects[1].primitives[0].dimensions.resize(3);
        collision_objects[1].primitives[0].dimensions[0] = 0.001;
        collision_objects[1].primitives[0].dimensions[1] = 1.0;
        collision_objects[1].primitives[0].dimensions[2] = 0.75;

        //Define the pose of the table.
        collision_objects[1].primitive_poses.resize(1);
        collision_objects[1].primitive_poses[0].position.x = 0.6;
        collision_objects[1].primitive_poses[0].position.y = 0.2;
        collision_objects[1].primitive_poses[0].position.z = 0.375;
        collision_objects[1].primitive_poses[0].orientation.w = 1.0;
        // END_SUB_TUTORIAL

        collision_objects[1].operation = collision_objects[1].ADD;

        // BEGIN_SUB_TUTORIAL object
        // Define the object that we will be manipulating
        collision_objects[2].header.frame_id = "base_link";
        collision_objects[2].id = "left";

        //Define the primitive and its dimensions.
        collision_objects[2].primitives.resize(1);
        collision_objects[2].primitives[0].type = collision_objects[1].primitives[0].BOX;
        collision_objects[2].primitives[0].dimensions.resize(3);
        collision_objects[2].primitives[0].dimensions[0] = 0.001;
        collision_objects[2].primitives[0].dimensions[1] = 1.0;
        collision_objects[2].primitives[0].dimensions[2] = 0.75;

        //Define the pose of the object.
        collision_objects[2].primitive_poses.resize(1);
        collision_objects[2].primitive_poses[0].position.x = -0.6;
        collision_objects[2].primitive_poses[0].position.y = 0.2;
        collision_objects[2].primitive_poses[0].position.z = 0.375;
        collision_objects[2].primitive_poses[0].orientation.w = 1.0;
        // END_SUB_TUTORIAL

        collision_objects[2].operation = collision_objects[2].ADD;
#endif
        collision_objects[0].header.frame_id = "base_link";
        collision_objects[0].id = "ur_floor";
          
        /* Define the primitive and its dimensions. */
        collision_objects[0].primitives.resize(1);
        collision_objects[0].primitives[0].type = collision_objects[1].primitives[0].BOX;
        collision_objects[0].primitives[0].dimensions.resize(3);
        collision_objects[0].primitives[0].dimensions[0] = 0.9;
        collision_objects[0].primitives[0].dimensions[1] = 0.9;
        collision_objects[0].primitives[0].dimensions[2] = 0.9;

        /* Define the pose of the object. */
        collision_objects[0].primitive_poses.resize(1);
        collision_objects[0].primitive_poses[0].position.x = 0.0;
        collision_objects[0].primitive_poses[0].position.y = 0.25;
        collision_objects[0].primitive_poses[0].position.z = -0.452;
        collision_objects[0].primitive_poses[0].orientation.w = 0.0;
        
        // END_SUB_TUTORIAL

        collision_objects[0].operation = collision_objects[0].ADD;   

        move_planning_scene_ -> applyCollisionObjects(collision_objects);
        // Make collosion to appropriate path planning finish
      }

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr publisher_;
    bool is_horizontally_centered;
    bool is_vertically_centered;
    bool is_moving;
    bool is_depth_reached;
    int prev_x;
    int prev_y;
    bool is_task_completed;
    bool was_centered_message_shown;
    float depth;
    std::vector<float> depths;
    rclcpp::Time end_timer;
    rclcpp::Time timer;
    moveit::planning_interface::MoveGroupInterface* move_group_;
    geometry_msgs::msg::Pose target_pose;
    std::vector<geometry_msgs::msg::Pose> waypoints;
    /* 추가한 부분 시작 */
    rclcpp::Client<interfaces_ur5ik::srv::SixTheta>::SharedPtr service_client_;
    moveit::planning_interface::PlanningSceneInterface* move_planning_scene_;
    /* 추가한 부분 끝*/
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);

  auto move_robot_node = rclcpp::Node::make_shared("move_robot", node_options);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(move_robot_node);
  std::thread spinner = std::thread([&executor]() { executor.spin(); });

  rclcpp::spin(std::make_shared<RobotMasterController>(move_robot_node));
  rclcpp::shutdown();
  return 0;
}
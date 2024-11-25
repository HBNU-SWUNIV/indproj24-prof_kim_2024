#include <chrono>
#include <memory>
#include <cstdlib>
#include <math.h>

#include "rclcpp/rclcpp.hpp"
#include "interfaces_ur5ik/srv/six_theta.hpp"

using namespace std::chrono_literals;

double pi =3.141592;

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("target_client");

    rclcpp::Client<interfaces_ur5ik::srv::SixTheta>::SharedPtr client =
        node -> create_client<interfaces_ur5ik::srv::SixTheta>("service_ik");

        auto request = std::make_shared<interfaces_ur5ik::srv::SixTheta::Request>();
        request -> srv_target[0] = atof(argv[1]); //0.0
        request -> srv_target[1] = atof(argv[2]); //0.6
        request -> srv_target[2] = atof(argv[3]); //0.3
        request -> srv_target[3] = atof(argv[4]); //pi
        request -> srv_target[4] = atof(argv[5]); //0.0
        request -> srv_target[5] = atof(argv[6]); //0.0


    while (!client -> wait_for_service(1s))
    {
        if(!rclcpp::ok())
        {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exitiong.");
            return 0;
        }
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"),"service not available, waiting again...");
    }

    auto result = client -> async_send_request(request);

    // Check if the future is valid
    if (!result.valid()) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Future object is invalid.");
        rclcpp::shutdown();
        return 1;
    }

    if(rclcpp::spin_until_future_complete(node, result) ==
        rclcpp::FutureReturnCode::SUCCESS)
        {
            #if 0
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta1 : %f", result.get() -> theta1);
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta2 : %f", result.get() -> theta2);
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta3 : %f", result.get() -> theta3);
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta4 : %f", result.get() -> theta4);
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta5 : %f", result.get() -> theta5);
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta6 : %f", result.get() -> theta6);
            #endif

            #if 1
            auto response = result.get();
            double theta1 = response -> srv_theta[0];
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta1 : %f", theta1);
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta2 : %f", response -> srv_theta[1]);
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta3 : %f", response -> srv_theta[2]);
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta4 : %f", response -> srv_theta[3]);
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta5 : %f", response -> srv_theta[4]);
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "theta6 : %f", response -> srv_theta[5]);
            #endif
        }
    else
    {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service");
    }

    rclcpp::shutdown();
    return 0;
}
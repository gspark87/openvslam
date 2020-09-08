#ifdef USE_PANGOLIN_VIEWER
#include <pangolin_viewer/viewer.h>
#elif USE_SOCKET_PUBLISHER
#include <socket_publisher/publisher.h>
#endif

#include <openvslam/system.h>
#include <openvslam/config.h>

#include <iostream>
#include <chrono>
#include <numeric>

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
// #include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif


#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>


void openvslam_tracking(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path,
                   const std::string& mask_img_path, const bool eval_log, const std::string& map_db_path) {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup();

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    std::vector<double> track_times;
    const auto tp_0 = std::chrono::steady_clock::now();

    // initialize this node
    auto node = std::make_shared<rclcpp::Node>("run_slam");
    rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
    custom_qos.depth = 1;


    // Topic publisher
    //auto node_pub = rclcpp::Node::make_shared("run_slam_pub");
    auto pub = node->create_publisher<geometry_msgs::msg::Pose>("navsol", 10);
    auto dat_nav = std::make_shared<geometry_msgs::msg::Pose>();

    // run the SLAM as subscriber
    //// working: approx. sync
    message_filters::Subscriber<sensor_msgs::msg::Image> rgbd_sub(node.get(), "camera/color/image_raw", custom_qos);
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub(node.get(), "camera/aligned_depth_to_color/image_raw", custom_qos);
    using rgbd_sync_policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    message_filters::Synchronizer<rgbd_sync_policy> sync(rgbd_sync_policy(10), rgbd_sub, depth_sub); // rgbd_sync_policy(queue_size)
    sync.registerCallback(std::bind([&](const sensor_msgs::msg::Image::ConstSharedPtr& rgb_img_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth_img_msg) {
        const auto tp_1 = std::chrono::steady_clock::now();
        const auto timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(tp_1 - tp_0).count();

        // input the current frame and estimate the camera pose
        // std::cout<<"hello0" << std::endl;
        // SLAM.feed_monocular_frame(cv_bridge::toCvShare(rgb_img_msg, "bgr8")->image, timestamp, mask);
        const auto cam_pos_cw = SLAM.feed_RGBD_frame(cv_bridge::toCvShare(rgb_img_msg, "bgr8")->image, cv_bridge::toCvShare(depth_img_msg, "16UC1")->image, timestamp, mask);

        const Eigen::Matrix4d& cam_pos_wc = cam_pos_cw.inverse();
        const Eigen::Matrix3d& rot_wc = cam_pos_wc.block<3, 3>(0, 0);
        const Eigen::Vector3d& trans_wc = cam_pos_wc.block<3, 1>(0, 3);
        const Eigen::Quaterniond quat_wc(rot_wc);

        dat_nav->position.x = trans_wc(0);
        dat_nav->position.y = trans_wc(1);
        dat_nav->position.z = trans_wc(2);
        dat_nav->orientation.x = quat_wc.x();
        dat_nav->orientation.y = quat_wc.y();
        dat_nav->orientation.z = quat_wc.z();
        dat_nav->orientation.w = quat_wc.w();
        pub->publish(*dat_nav);
            
        //std::cout << "[Trajectory]\t" << trans_wc << std::endl;

        const auto tp_2 = std::chrono::steady_clock::now();

        const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
        track_times.push_back(track_time);
    }, std::placeholders::_1, std::placeholders::_2));

    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);


    // Pangolin needs to run in the main thread on OSX
    std::thread thread([&]() {
        exec.spin();
    });

    // run the viewer in this thread
#ifdef USE_PANGOLIN_VIEWER
    viewer.run();
    if (SLAM.terminate_is_requested()) {
        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
        rclcpp::shutdown();
    }
#elif USE_SOCKET_PUBLISHER
    publisher.run();
    if (SLAM.terminate_is_requested()) {
        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
        rclcpp::shutdown();
    }
#else
    std::cout << "\nPress Q to exit\n" << std::endl;
    char xxx;
    std::cin >> xxx;
    if (xxx == 'q') {
        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
        rclcpp::shutdown();
     }
#endif

    // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
    viewer.request_terminate();
    thread.join();
#elif USE_SOCKET_PUBLISHER
    publisher.request_terminate();
    thread.join();
#endif

    // shutdown the SLAM process
    SLAM.shutdown();

    if (eval_log) {
        // output the trajectories for evaluation
        SLAM.save_frame_trajectory("frame_trajectory.txt", "TUM");
        SLAM.save_keyframe_trajectory("keyframe_trajectory.txt", "TUM");
        // output the tracking times for evaluation
        std::ofstream ofs("track_times.txt", std::ios::out);
        if (ofs.is_open()) {
            for (const auto track_time : track_times) {
                ofs << track_time << std::endl;
            }
            ofs.close();
        }
    }

    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path);
    }

    if (track_times.size()) {
        std::sort(track_times.begin(), track_times.end());
        const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
        std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
        std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
    }
}

int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif
    rclcpp::init(argc, argv);
    rclcpp::uninstall_signal_handlers();

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto setting_file_path = op.add<popl::Value<std::string>>("c", "config", "setting file path");
    auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
    auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
    auto eval_log = op.add<popl::Switch>("", "eval-log", "store trajectory and tracking times for evaluation");
    auto map_db_path = op.add<popl::Value<std::string>>("", "map-db", "store a map database at this path after SLAM", "");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !setting_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode->is_set()) {
        spdlog::set_level(spdlog::level::debug);
    }
    else {
        spdlog::set_level(spdlog::level::info);
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(setting_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    // run tracking
    if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular ||
        cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::RGBD ||
        cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Stereo) {
        openvslam_tracking(cfg, vocab_file_path->value(), mask_img_path->value(), eval_log->is_set(), map_db_path->value());
    }
    else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
    }

#include "ppf_recognizer_ros_msgs/recognize.h"
#include "ppf_recognizer_ros_msgs/retrain_recognizer.h"
#include "ppf_recognizer_ros_msgs/set_camera.h"
#include "object_detector_msgs/estimate_poses.h"

#include <image_transport/image_transport.h>
#include <PPFRecognizer.h>
#include <v4r/recognition/object_hypothesis.h>

namespace v4r
{
template<typename PointT>
class RecognizerROS {
public:
  RecognizerROS() {}

  bool initialize(int argc, char** argv);
  bool setupRecognizerOptions(std::vector<std::string>& arguments, const bf::path& config_dir);

  bool setCamera(ppf_recognizer_ros_msgs::set_camera::Request& req,
                 ppf_recognizer_ros_msgs::set_camera::Response& response);

  bool recognizeROS(ppf_recognizer_ros_msgs::recognize::Request& req,
                    ppf_recognizer_ros_msgs::recognize::Response& response);

  bool estimate(object_detector_msgs::estimate_poses::Request& req,
                object_detector_msgs::estimate_poses::Response& response);

  void pointcloudFromRGBD(sensor_msgs::Image& rgb, sensor_msgs::Image& depth,
                          object_detector_msgs::Detection& det,
                          typename pcl::PointCloud<PointT>::Ptr& cloud);

private:
  boost::shared_ptr<ros::NodeHandle> n_;

  ros::Publisher vis_pc_pub_;

  ros::Publisher conv_cloud_pub_;     // publisher for pointcloud converted from rgbd image pair
  ros::Publisher pose_estimate_cloud_pub_;  // publishes model object aligned with its estimated pose

  image_transport::Publisher image_pub_;
  boost::shared_ptr<image_transport::ImageTransport> it_;

  ros::ServiceServer recognize_;
  ros::ServiceServer recognizer_set_camera_;
  ros::ServiceServer estimate_poses_;

  typename boost::shared_ptr<v4r::apps::PPFRecognizer<PointT>> mrec_; // recognizer
  typename pcl::PointCloud<PointT>::Ptr scene_; // input cloud
  v4r::Intrinsics::Ptr camera_; // camera (if cloud is not organized)

  // @todo object_hypotheses_ doesn't really need to be a member variable, could be
  //       eliminated by passing it as a additional parameter to respondSrvCall()
  std::vector<ObjectHypothesesGroup> object_hypotheses_;  // recognized objects

  bool respondSrvCall(ppf_recognizer_ros_msgs::recognize::Request &req,
                      ppf_recognizer_ros_msgs::recognize::Response &response) const;
};

} // end namespace v4r

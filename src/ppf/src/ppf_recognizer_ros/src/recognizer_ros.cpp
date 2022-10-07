#include "recognizer_ros.h"

#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>

#include <iostream>
#include <sstream>

#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <sensor_msgs/image_encodings.h>

#include "object_detector_msgs/PoseWithConfidence.h"

namespace po = boost::program_options;

namespace v4r
{

/**
 * @todo sanitize this function
 *       possibly split into subfunctions
 */
template<typename PointT>
bool RecognizerROS<PointT>::respondSrvCall(ppf_recognizer_ros_msgs::recognize::Request &req,
                                           ppf_recognizer_ros_msgs::recognize::Response &response) const
{
  typename pcl::PointCloud<PointT>::Ptr pRecognizedModels{new pcl::PointCloud<PointT>{}};

  // convert point cloud
  v4r::PCLOpenCVConverter<PointT> img_conv;
  img_conv.setInputCloud(scene_);
  img_conv.setCameraIntrinsics(camera_);
  cv::Mat annotated_img = img_conv.getRGBImage();

  float intrinsic[9] = {camera_->fx, 0, camera_->cx, 0, camera_->fy, camera_->cy, 0.f, 0.f, 1.f};

  for (size_t ohg_id=0; ohg_id < object_hypotheses_.size(); ++ohg_id) {
    for (const v4r::ObjectHypothesis::Ptr &oh : object_hypotheses_[ohg_id].ohs_) {
      if ( ! oh->is_verified_ )
        continue;

      std_msgs::String ss_tmp;
      ss_tmp.data = oh->model_id_;
      response.ids.push_back(ss_tmp);

      Eigen::Matrix4f trans = oh->pose_refinement_ * oh->transform_;
      geometry_msgs::Transform tt;
      tt.translation.x = trans(0,3);
      tt.translation.y = trans(1,3);
      tt.translation.z = trans(2,3);

      Eigen::Matrix3f rotation = trans.block<3,3>(0,0);
      Eigen::Quaternionf q(rotation);
      tt.rotation.x = q.x();
      tt.rotation.y = q.y();
      tt.rotation.z = q.z();
      tt.rotation.w = q.w();
      response.transforms.push_back(tt);

      // typename pcl::PointCloud<PointT>::ConstPtr model_cloud = mrec_->getModel( oh->model_id_, 5 );
      auto model_cloud_with_normals = mrec_->getModel(oh->model_id_, 5);
      typename pcl::PointCloud<PointT>::Ptr model_cloud (new pcl::PointCloud<PointT>);
      pcl::copyPointCloud(*model_cloud_with_normals, *model_cloud);
      typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
      pcl::transformPointCloud (*model_cloud, *model_aligned, trans);
      *pRecognizedModels += *model_aligned;
      sensor_msgs::PointCloud2 rec_model;
      pcl::toROSMsg(*model_aligned, rec_model);
      rec_model.header = req.cloud.header;
      response.models_cloud.push_back(rec_model);

      //        pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud = model->getNormalsAssembled ( resolution_ );

      //        pcl::PointCloud<pcl::Normal>::Ptr normal_aligned (new pcl::PointCloud<pcl::Normal>);
      //        transformNormals(*normal_cloud, *normal_aligned, oh->transform_);
      //      VisibilityReasoning<pcl::PointXYZRGB> vr (focal_length, img_width, img_height);
      //      vr.setThresholdTSS (0.01f);
      //      /*float fsv_ratio =*/ vr.computeFSVWithNormals (scene_, model_aligned, normal_aligned);
      //      confidence = 1.f - vr.getFSVUsedPoints() / static_cast<float>(model_aligned->points.size());
      //      response.confidence.push_back(confidence);

      // draw bounding box
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*model_aligned, centroid);
      geometry_msgs::Point32 centroid_msg;
      centroid_msg.x = centroid[0];
      centroid_msg.y = centroid[1];
      centroid_msg.z = centroid[2];
      response.centroid.push_back(centroid_msg);

      Eigen::Vector4f min;
      Eigen::Vector4f max;
      pcl::getMinMax3D (*model_aligned, min, max);

      geometry_msgs::Polygon bbox;
      bbox.points.resize(8);
      geometry_msgs::Point32 pt;
      pt.x = min[0]; pt.y = min[1]; pt.z = min[2]; bbox.points[0] = pt;
      pt.x = min[0]; pt.y = min[1]; pt.z = max[2]; bbox.points[1] = pt;
      pt.x = min[0]; pt.y = max[1]; pt.z = min[2]; bbox.points[2] = pt;
      pt.x = min[0]; pt.y = max[1]; pt.z = max[2]; bbox.points[3] = pt;
      pt.x = max[0]; pt.y = min[1]; pt.z = min[2]; bbox.points[4] = pt;
      pt.x = max[0]; pt.y = min[1]; pt.z = max[2]; bbox.points[5] = pt;
      pt.x = max[0]; pt.y = max[1]; pt.z = min[2]; bbox.points[6] = pt;
      pt.x = max[0]; pt.y = max[1]; pt.z = max[2]; bbox.points[7] = pt;
      response.bbox.push_back(bbox);

      int min_u, min_v, max_u, max_v;
      min_u = annotated_img.cols;
      min_v = annotated_img.rows;
      max_u = max_v = 0;

      for (size_t m_pt_id=0; m_pt_id < model_aligned->points.size(); m_pt_id++) {
        float x = model_aligned->points[m_pt_id].x;
        float y = model_aligned->points[m_pt_id].y;
        float z = model_aligned->points[m_pt_id].z;
        int u = static_cast<int> ( camera_->fx * x / z + camera_->cx);
        int v = static_cast<int> ( camera_->fy * y / z + camera_->cy);

        if (u >= annotated_img.cols || v >= annotated_img.rows || u < 0 || v < 0)
          continue;

        if (u < min_u)
          min_u = u;

        if (v < min_v)
          min_v = v;

        if (u > max_u)
          max_u = u;

        if (v > max_v)
          max_v = v;
      }

      cv::rectangle(annotated_img, cv::Point(min_u, min_v), cv::Point(max_u, max_v), cv::Scalar(0, 255, 255), 2);

      // draw coordinate system
      float size=0.1;
      float thickness = 4;
      const Eigen::Matrix3f &R = trans.topLeftCorner<3,3>();
      const Eigen::Vector3f &t = trans.block<3, 1>(0,3);

      Eigen::Vector3f pt0  = R * Eigen::Vector3f(0,0,0) + t;
      Eigen::Vector3f pt_x = R * Eigen::Vector3f(size,0,0) + t;
      Eigen::Vector3f pt_y = R * Eigen::Vector3f(0,size,0) + t;
      Eigen::Vector3f pt_z = R * Eigen::Vector3f(0,0,size) + t ;

      cv::Point2f im_pt0, im_pt_x, im_pt_y, im_pt_z;

      // @todo what happened here ???
//            if (!dist_coeffs.empty())
//            {
//                v4r::projectPointToImage(&pt0 [0], &intrinsic(0), &dist_coeffs(0), &im_pt0.x );
//                v4r::projectPointToImage(&pt_x[0], &intrinsic(0), &dist_coeffs(0), &im_pt_x.x);
//                v4r::projectPointToImage(&pt_y[0], &intrinsic(0), &dist_coeffs(0), &im_pt_y.x);
//                v4r::projectPointToImage(&pt_z[0], &intrinsic(0), &dist_coeffs(0), &im_pt_z.x);
//            }
//            else
      {
          v4r::projectPointToImage(&pt0 [0], &intrinsic[0], &im_pt0.x );
          v4r::projectPointToImage(&pt_x[0], &intrinsic[0], &im_pt_x.x);
          v4r::projectPointToImage(&pt_y[0], &intrinsic[0], &im_pt_y.x);
          v4r::projectPointToImage(&pt_z[0], &intrinsic[0], &im_pt_z.x);
      }

      cv::line(annotated_img, im_pt0, im_pt_x, CV_RGB(255,0,0), thickness);
      cv::line(annotated_img, im_pt0, im_pt_y, CV_RGB(0,255,0), thickness);
      cv::line(annotated_img, im_pt0, im_pt_z, CV_RGB(0,0,255), thickness);

      cv::Point text_start;
      text_start.x = min_u;
      text_start.y = std::max(0, min_v - 10);
      cv::putText(annotated_img, oh->model_id_, text_start, cv::FONT_HERSHEY_COMPLEX_SMALL,
                  0.8, cv::Scalar(255,0,255), 1, CV_AA);
    }
  }

  sensor_msgs::PointCloud2 recognizedModelsRos;
  pcl::toROSMsg (*pRecognizedModels, recognizedModelsRos);
  recognizedModelsRos.header = req.cloud.header;
  vis_pc_pub_.publish(recognizedModelsRos);

  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", annotated_img).toImageMsg();
  msg->header = req.cloud.header;
  image_pub_.publish(msg);

  return true;
}

template<typename PointT>
bool RecognizerROS<PointT>::setCamera(ppf_recognizer_ros_msgs::set_camera::Request& req,
                                      ppf_recognizer_ros_msgs::set_camera::Response& response)
{
  v4r::Intrinsics::Ptr cam;
  if (req.cam.K[0] < std::numeric_limits<float>::epsilon()) {

    ROS_WARN("Given camera calibration matrix has focal length 0. Using default(Primesense) settings");
    cam.reset(new v4r::Intrinsics{v4r::Intrinsics::PrimeSense()});

  } else {
    cam.reset(new v4r::Intrinsics({
      static_cast<float>(req.cam.K[0]), // fx
      static_cast<float>(req.cam.K[4]), // fy
      static_cast<float>(req.cam.K[2]), // cx
      static_cast<float>(req.cam.K[5]), // cy
      req.cam.width, req.cam.height     // w, h
    }));
  }

  camera_ = cam;
  mrec_->setCameraIntrinsics(*cam);

  // (void)response;    // ros service callbacks should return bool to indicate success or failure
  return true;
}

template<typename PointT>
bool RecognizerROS<PointT>::recognizeROS(ppf_recognizer_ros_msgs::recognize::Request &req,
                                         ppf_recognizer_ros_msgs::recognize::Response &response)
{
  scene_.reset(new pcl::PointCloud<PointT>());
  pcl::fromROSMsg (req.cloud, *scene_);

  scene_->sensor_orientation_ = Eigen::Quaternionf( req.transform.rotation.w, req.transform.rotation.x, req.transform.rotation.y, req.transform.rotation.z );
  scene_->sensor_origin_ =
    Eigen::Vector4f(req.transform.translation.x, req.transform.translation.y, req.transform.translation.z, 0.f);
    // @note In PCL the last component always seems to be set to 0.
    // Not sure what it does though. Behaves differently if set to 1.

  // get object ids to look for
  std::vector<std::string> objects_to_look_for;
  std::transform(req.objects.begin(), req.objects.end(), std::back_inserter(objects_to_look_for),
                 [](std_msgs::String& s) { return s.data; });

  object_hypotheses_ = mrec_->recognize( scene_, objects_to_look_for);

  for (size_t ohg_id=0; ohg_id < object_hypotheses_.size(); ++ohg_id) {
    for (const v4r::ObjectHypothesis::Ptr &oh : object_hypotheses_[ohg_id].ohs_) {
      const std::string &model_id = oh->model_id_;
      const Eigen::Matrix4f &tf = oh->pose_refinement_ * oh->transform_;

      if (oh->is_verified_)
        ROS_INFO_STREAM("********************" << model_id << std::endl << tf << std::endl);
    }
  }

  return respondSrvCall(req, response);
}

std::tuple<bool, int, int, int, int> colorOffsetsFromEncoding(std::string encoding) {

  bool valid{true};
  int red_offset, green_offset, blue_offset, color_step;

  if (encoding == sensor_msgs::image_encodings::RGB8) {
    red_offset   = 0;
    green_offset = 1;
    blue_offset  = 2;
    color_step   = 3;
  } else if (encoding == sensor_msgs::image_encodings::BGR8) {
    red_offset   = 2;
    green_offset = 1;
    blue_offset  = 0;
    color_step   = 3;
  } else if (encoding == sensor_msgs::image_encodings::MONO8) {
    red_offset   = 0;
    green_offset = 0;
    blue_offset  = 0;
    color_step   = 1;
  } else {
    valid = false;
  }

  return std::make_tuple(valid, red_offset, green_offset, blue_offset, color_step);
}

template<typename T>
void convert(v4r::Intrinsics::Ptr& camera, sensor_msgs::Image& rgb, sensor_msgs::Image& depth,
             object_detector_msgs::Detection& det, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float scale)
{
  // no c++ 17 support in catkin yet so std::tie has to be used instead of structured bindings
  int valid, red_offset, green_offset, blue_offset, color_step;
  std::tie(valid, red_offset, green_offset, blue_offset, color_step) = colorOffsetsFromEncoding(rgb.encoding);
  
  if (!valid) {
    ROS_ERROR("unsupported color image encoding : %s\n", rgb.encoding.c_str());
    return;
  }


  const T* depth_start = reinterpret_cast<const T*>(&depth.data[0]);
  int row_step = depth.step / sizeof(T); // sensor_msgs::Image::step = length of row in bytes
  const uint8_t* color_start = &rgb.data[0];

  int valid_mask_points = 0;

  // det.mask stores indices
  float min_depth = std::numeric_limits<float>::max();
  float max_depth = std::numeric_limits<float>::min();

  for (auto ind : det.mask) {
    // convert linear index ind to i, j coordinates in bounding box
    int v = ind / depth.width; // row
    int u = ind % depth.width; // col

    valid_mask_points++;

    float depth = float(depth_start[v*row_step + u]) / scale;
    const uint8_t* color = color_start + v*rgb.step + u*color_step;

    pcl::PointXYZRGB p;

    // @todo check that depth is valid

    // compute and fill xyz
    p.x = float(u - camera->cx) * depth / camera->fx;
    p.y = float(v - camera->cy) * depth / camera->fy;
    p.z = depth;

    if(p.z < min_depth)
      min_depth = p.z;
    if(p.z > max_depth)
      max_depth = p.z;

    // fill rgb
    p.r = color[red_offset];
    p.g = color[green_offset];
    p.b = color[blue_offset];

    // cloud->push_back(std::move(p)); // push_back sets height to 1!
    // cloud->at(u, v) = p;
    cloud->points[ind] = p;
  }

  cloud->is_dense = false;
}

template<>
void RecognizerROS<pcl::PointXYZRGB>::pointcloudFromRGBD(sensor_msgs::Image& rgb, sensor_msgs::Image& depth,
                                                         object_detector_msgs::Detection& det,
                                                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
{
  if (depth.encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    float scale = 1000.0;   // 16UC1 depth values are in millimeters
    convert<uint16_t>(camera_, rgb, depth, det, cloud, scale);
  } else if (depth.encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    float scale = 1.0;   // 32FC1 depth values are in meters
    convert<float>(camera_, rgb, depth, det, cloud, scale);
  }
  
  sensor_msgs::PointCloud2 converted_cloud;
  pcl::toROSMsg (*cloud, converted_cloud);
  converted_cloud.header = depth.header;
  conv_cloud_pub_.publish(converted_cloud);
}

template<>
bool RecognizerROS<pcl::PointXYZRGB>::estimate(object_detector_msgs::estimate_poses::Request& request,
                                               object_detector_msgs::estimate_poses::Response& response)
{
  // create pointcloud from message rgb and depth data
  // pointcloud needs to be organized
  auto bad_point = pcl::PointXYZRGB();
  bad_point.x = bad_point.y = bad_point.z = std::numeric_limits<float>::quiet_NaN();
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud{
    new pcl::PointCloud<pcl::PointXYZRGB>(request.depth.width, request.depth.height, bad_point)
  };
  
  // convert rgb and depth image to pointcloud
  pointcloudFromRGBD(request.rgb, request.depth, request.det, cloud);

  // set pointcloud sensor origin / orientation ?

  std::vector<std::string> objects_to_look_for{request.det.name};

  auto hypothesis_groups = mrec_->recognize(cloud, objects_to_look_for);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_estimates{new pcl::PointCloud<pcl::PointXYZRGB>{}};

  // fill service response with generated hypotheses
  for (auto& group : hypothesis_groups) {
    for (auto& hypothesis : group.ohs_) {
      
      object_detector_msgs::PoseWithConfidence p;

      // p.name = hypothesis->model_id_;
      p.name = request.det.name;

      Eigen::Matrix4f trans = hypothesis->transform_;
      p.pose.position.x = trans(0,3);
      p.pose.position.y = trans(1,3);
      p.pose.position.z = trans(2,3);

      Eigen::Matrix3f rotation = trans.block<3,3>(0,0);
      Eigen::Quaternionf q(rotation);
      p.pose.orientation.x = q.x();
      p.pose.orientation.y = q.y();
      p.pose.orientation.z = q.z();
      p.pose.orientation.w = q.w();

      p.confidence = hypothesis->confidence_wo_hv_;
      ROS_INFO("id : %s, confidence %f\n", hypothesis->model_id_.c_str(), hypothesis->confidence_wo_hv_);

      response.poses.push_back(std::move(p));

    }

    // find hypothesis with highest confidence
    auto best_hypothesis = std::max_element(group.ohs_.begin(), group.ohs_.end(),
      [](auto a, auto b)
      {
        return a->confidence_wo_hv_ < b->confidence_wo_hv_;
      }
    );

    // add the model of the best hypothesis for this object, transformed into the estimated frame to the
    // pointcloud with estimated models (for vis. purposes)
    if (best_hypothesis != group.ohs_.end()) {

      auto hypothesis = *best_hypothesis;

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud{new pcl::PointCloud<pcl::PointXYZRGB>{}};
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_aligned{new pcl::PointCloud<pcl::PointXYZRGB>{}};

      auto model_cloud_with_normals = mrec_->getModel(hypothesis->model_id_, 5);
      pcl::copyPointCloud(*model_cloud_with_normals, *model_cloud);
      pcl::transformPointCloud(*model_cloud, *model_aligned, hypothesis->transform_);
      *model_estimates += *model_aligned;
    }
  }

  // publish pose estimate visualization pointcloud

  // convert pointcloud to ros msg
  sensor_msgs::PointCloud2 model_estimate_msg;
  pcl::toROSMsg(*model_estimates, model_estimate_msg);
  model_estimate_msg.header = request.depth.header;

  // publish 
  pose_estimate_cloud_pub_.publish(model_estimate_msg);

  return true;
}

template<typename PointT>
bool RecognizerROS<PointT>::initialize(int argc, char** argv)
{
  n_.reset(new ros::NodeHandle{"~"});

  // @note if I uncomment this line glog doesn't complain about loggin before init,
  //       but also doesn't log anything (to stdout/stderr), need to set verbosity ?
  // google::InitGoogleLogging(argv[0]);

  // This should get all console arguments except the ROS ones
  std::vector<std::string> arguments{argv + 1, argv + argc};

  std::string models_dir = "", cfg_dir = "";
  if (!n_->getParam("models_dir", models_dir)) {
    ROS_ERROR("Models directory is not set. Must be set with ROS parameter \"models_dir\"!");
    return false;
  } else {
    arguments.push_back("-m");
    arguments.push_back(models_dir);
  }

  if (!n_->getParam("cfg_dir", cfg_dir)) {
    ROS_ERROR("The directory containing the XML config folders for object recognition is not set. "
              "Must be set with ROS parameter \"cfg_dir\"!");
    return false;
  }

  if (arguments.empty()) {
    std::string additional_arguments;
    if (n_->getParam("arg", additional_arguments)) {
      std::vector<std::string> strs;
      boost::split(strs, additional_arguments, boost::is_any_of("\t "));
      arguments.insert(arguments.end(), strs.begin(), strs.end());
    }
  }

  std::cout << "Initializing recognizer with: " << std::endl;
  for (auto& arg : arguments)
      std::cout << arg << " ";
  std::cout << std::endl;

  if (!setupRecognizerOptions(arguments, cfg_dir))
    return false;

  vis_pc_pub_ = n_->advertise<sensor_msgs::PointCloud2>("recognized_object_instances", 1);

  conv_cloud_pub_ = n_->advertise<sensor_msgs::PointCloud2>("converted_pointcloud", 1);
  pose_estimate_cloud_pub_ = n_->advertise<sensor_msgs::PointCloud2>("estimated_model_pose", 1);

  recognize_  = n_->advertiseService("recognize", &RecognizerROS::recognizeROS, this);
  recognizer_set_camera_  = n_->advertiseService("set_camera", &RecognizerROS::setCamera, this);
  estimate_poses_ = n_->advertiseService("estimate_poses", &RecognizerROS::estimate, this);

  it_.reset(new image_transport::ImageTransport{*n_});
  image_pub_ = it_->advertise("recognized_object_instances_img", 1, true);

  ROS_INFO("Ready to get service calls.");
  ROS_INFO("Don't forget to set camera intrinsics using the \"set_camera\" service !");

  return true;
}

template<typename PointT>
bool RecognizerROS<PointT>::setupRecognizerOptions(std::vector<std::string>& arguments, const bf::path& config_dir)
{
  bf::path models_dir;
  bf::path config_file = config_dir/"ppf_pose_estimation_config.ini";

  int verbosity = -1;
  bool visualize = false;
  bool ignore_ROI_from_file = false;
  bool ask_for_ROI = false;
  bool force_retrain = false;  ///< if true, will retrain object models even if trained data already exists
  v4r::apps::PPFRecognizerParameter ppf_params;

  po::options_description desc("PPF Object Instance Recognizer\n"
                               "==============================\n"
                               "     **Allowed options**\n");

  // get config file or use default
  desc.add_options()
    ("help,h", "produce help message")
    ("cfg,c", po::value<bf::path>(&config_file)->default_value(config_file),
     "File path of V4R config (.ini) file containing parameters for the recognition pipeline");

  po::variables_map vm;
  po::parsed_options parsed_tmp = po::command_line_parser(arguments)
    .options(desc).allow_unregistered().run();
  std::vector<std::string> to_pass_further =
    po::collect_unrecognized(parsed_tmp.options, po::include_positional);
  po::store(parsed_tmp, vm);

  try {
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
  }

  desc.add_options()
    ("model_dir,m", po::value<bf::path>(&models_dir)->required(), "Directory with object models.")
    ("verbosity", po::value<int>(&verbosity)->default_value(verbosity),
     "set verbosity level for output (<0 minimal output)")
    ("visualize,v", po::bool_switch(&visualize), "visualize recognition results")
    ("ignore_ROI_from_file", po::bool_switch(&ignore_ROI_from_file),
     "if set, does not try to read ROI from file")
    ("ask_for_ROI", po::bool_switch(&ask_for_ROI), "if true, asks the user to provide ROI")
    ("retrain", po::bool_switch(&force_retrain),
     "If set, retrains the object models no matter if they already exists.");

  ppf_params.init(desc);
  po::parsed_options parsed = po::command_line_parser(to_pass_further).options(desc).run();
  po::store(parsed, vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return false;
  }

  if (v4r::io::existsFile(config_file)) {
    std::ifstream f(config_file.string());
    po::parsed_options config_parsed = po::parse_config_file(f, desc);
    po::store(config_parsed, vm);
    f.close();
  } else {
    LOG(ERROR) << config_file.string() << " does not exist! Usage: " << desc;
  }

  try {
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
    return false;
  }

  if (verbosity >= 0) {
    FLAGS_v = verbosity;
    std::cout << "Enabling verbose logging." << std::endl;
  } else {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  }

  mrec_.reset(new v4r::apps::PPFRecognizer<PointT>(ppf_params));
  mrec_->setModelsDir(models_dir);
  mrec_->setup(force_retrain);
  camera_.reset(new v4r::Intrinsics(mrec_->getCameraIntrinsics()));

  return true;
}

} // end namespace v4r

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ppf_object_recognizer");
  v4r::RecognizerROS<pcl::PointXYZRGB> r;

  if(!r.initialize(argc, argv))
    return 0;
  else
    ros::spin();

  return 0;
}

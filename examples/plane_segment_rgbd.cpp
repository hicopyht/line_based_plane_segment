#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <boost/bind.hpp>

#include "plane_segment.h"

using namespace std;
using namespace line_based_plane_segment;

class KinectListener
{
    typedef message_filters::Subscriber<sensor_msgs::Image> ImageSubType;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>  RgbdSyncPolicy;

public:
    KinectListener(const std::string &file_setting)
        : nh_()
        , private_nh_("~")
    {
        //
        cv::FileStorage fSettings(file_setting, cv::FileStorage::READ);
        queue_size_ = (int)fSettings["ROS.queue_size"];
        topic_rgb_ = (string)fSettings["ROS.topic_rgb"];
        topic_depth_ = (string)fSettings["ROS.topic_depth"];
        if( queue_size_ == 0 )
            private_nh_.param<int>("queue_size", queue_size_, 4);
        if( topic_rgb_.empty() )
            private_nh_.param<string>("topic_rgb", topic_rgb_, "/camera/rgb/image_color");
        if( topic_depth_.empty() )
            private_nh_.param<string>("topic_depth", topic_depth_, "/camera/depth/image");

        plane_segment_ = new PlaneSegment(file_setting);
        //
        rgb_sub_ = new ImageSubType(nh_, topic_rgb_, 1);
        depth_sub_ = new ImageSubType(nh_, topic_depth_, 1);
        rgbd_sync_ = new message_filters::Synchronizer<RgbdSyncPolicy>(RgbdSyncPolicy(queue_size_), *rgb_sub_, *depth_sub_);
        rgbd_sync_->registerCallback(boost::bind(&KinectListener::rgbdCallback, this, _1, _2));
        ROS_INFO_STREAM("Subscribe to topics '" << topic_rgb_ << "' and '" << topic_depth_ << "'." );
    }



    KinectListener()
        : nh_()
        , private_nh_("~")
        , queue_size_(4)
        , topic_rgb_("/camera/rgb/image_color")
        , topic_depth_("/camera/depth/image")
    {
        //
        CAMERA_INFO camera_image(640, 480, 319.5, 239.5, 525.0, 525.0, 1.0);
        plane_segment_ = new PlaneSegment(&camera_image);

        //
        private_nh_.param<int>("queue_size", queue_size_, queue_size_);
        private_nh_.param<string>("topic_rgb", topic_rgb_, topic_rgb_);
        private_nh_.param<string>("topic_depth", topic_depth_, topic_depth_);
        //
        rgb_sub_ = new ImageSubType(nh_, topic_rgb_, 1);
        depth_sub_ = new ImageSubType(nh_, topic_depth_, 1);
        rgbd_sync_ = new message_filters::Synchronizer<RgbdSyncPolicy>(RgbdSyncPolicy(queue_size_), *rgb_sub_, *depth_sub_);
        rgbd_sync_->registerCallback(boost::bind(&KinectListener::rgbdCallback, this, _1, _2));
        ROS_INFO_STREAM("Subscribe to topics '" << topic_rgb_ << "' and '" << topic_depth_ << "'." );
    }

    void rgbdCallback(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
    {
        // Copy the ros image message to cv::Mat.
        cv_bridge::CvImagePtr cv_ptrRGB;
        try
        {
            cv_ptrRGB = cv_bridge::toCvCopy(msgRGB);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv_bridge::CvImagePtr cv_ptrD;
        try
        {
            cv_ptrD = cv_bridge::toCvCopy(msgD);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        ROS_INFO_STREAM("RGBD message: " << msgD->header.seq );

        plane_segment_->segment(cv_ptrRGB->image, cv_ptrD->image);
    }

private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    //
    int queue_size_;
    std::string topic_rgb_;
    std::string topic_depth_;
    //
    ImageSubType* rgb_sub_;
    ImageSubType* depth_sub_;
    message_filters::Synchronizer<RgbdSyncPolicy>* rgbd_sync_;

    //
    PlaneSegment *plane_segment_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "plane_segment_rgbd");
    ros::NodeHandle nh;

    std::string file_setting;
    if( argc >= 2)
        file_setting = argv[1];

    KinectListener *kl;
    if( file_setting.empty())
        kl = new KinectListener();
    else
        kl = new KinectListener(file_setting);

    ros::spin();

}

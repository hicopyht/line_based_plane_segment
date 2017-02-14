#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include "plane_segment.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "omps_segment_images");
    ros::NodeHandle nh;
    if( argc < 3 )
    {
        cout << "Usage: ./plane_segment_images imageRGB imageDepth <fileSetting.yaml>" << endl;
        exit(-1);
    }
    std::string frgb = argv[1];
    std::string fdepth = argv[2];
    std::string file_setting;
    if( argc >= 4 )
        file_setting = argv[3];
    //
    cv::Mat image_rgb = cv::imread(frgb);
    cv::Mat image_depth = cv::imread(fdepth, -1);
    cout << "Image rgb: " << image_rgb.type() << " " << image_rgb.cols << " " << image_rgb.rows << " " << image_rgb.channels() << endl;
    cout << "Image depth: " << image_depth.type() << " " << image_depth.cols << " " << image_depth.rows << " " << image_depth.channels() << endl;
    if( image_depth.channels() == 3 )
    {
        cv::cvtColor(image_depth, image_depth, CV_BGR2GRAY);
    }
    cout << "Image depth: " << image_depth.type() << " " << image_depth.cols << " " << image_depth.rows << " " << image_depth.channels() << endl;
    if( image_rgb.empty() || image_depth.empty()
            || image_rgb.channels() != 3 || image_depth.channels() != 1
            || image_depth.cols != 640 || image_depth.rows != 480
            || image_rgb.cols != 640 || image_rgb.rows != 480)
    {
        cout << "[ERROR]: Loading images error." << endl;
        exit(-1);
    }
    //
//    CAMERA_INFO camera_image(640, 480, 319.5, 239.5, 525.0, 525.0, 1.0);
    PlaneSegment* plane_segmentor;
//    if(file_setting.empty())
//        plane_segmentor = new PlaneSegment(&camera_image, true);
//    else
//        plane_segmentor = new PlaneSegment(file_setting, true);
    plane_segmentor = new PlaneSegment(file_setting, true);

    while(ros::ok())
    {
        plane_segmentor->segment(image_rgb, image_depth);
        usleep(100000);
    }
}



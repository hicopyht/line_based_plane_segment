#ifndef PLANE_SEGMENT_H
#define PLANE_SEGMENT_H

#include <ros/ros.h>
#include "plane_viewer.h"
#include "line_based_plane_segmentation.h"
#include "organized_multi_plane_segmentor.h"

using namespace std;
using namespace line_based_plane_segment;

class PlaneSegment
{
public:
    PlaneSegment(const std::string &file_setting);
    PlaneSegment(CAMERA_INFO *camera_image = new CAMERA_INFO(640, 480, 319.5, 239.5, 525.0, 525.0, 1.0));
    PlaneSegment(const std::string &file_setting, bool use_omps);
    ~PlaneSegment();
//    void quit(){ viewer_->finish();}
    void segment(const cv::Mat &image_rgb, const cv::Mat &image_depth);
    void segment(const cv::Mat &image_rgb, const cv::Mat &image_depth,
                 const CAMERA_INFO &camera_image, const CAMERA_INFO &camera_cloud);

    //
    PointCloudTypePtr getPointCloud(const cv::Mat &image_rgb, const cv::Mat &image_depth,
                                    const CAMERA_INFO &camera_image, const CAMERA_INFO &camera_cloud);
    //
    void setStop() {stopped_ = true;}
    void setResume() {stopped_ = false;}
    bool &isStopped() {return stopped_;}

    //
    LineBasedPlaneSegmentation* getPlaneSegmentor(){ return plane_segmentor_; }
    void updatePlaneSegmentParameters();
    void resetPlaneSegmentParameters();
    //
    void setSkipPixel(int skip);
    int getSkipPixel() const{return skip_pixel_;}

    /// For omps
    void ompsSegment(const cv::Mat &image_rgb, const cv::Mat &image_depth,
                     const CAMERA_INFO &camera_image, const CAMERA_INFO &camera_cloud);

    /// For with normals
    pcl::IntegralImageNormalEstimation<PointType, pcl::Normal> &getNormalEstimation() { return ne_; }

private:
    inline void pushRuntime(std::vector<std::string> &procedures,
                            std::vector<float> &runtimes,
                            ros::Time &start_dura,
                            const std::string &step)
    {
        runtimes.push_back((ros::Time::now()-start_dura).toSec()*1000.0f);
        procedures.push_back(step);
        start_dura = ros::Time::now();
    }

private:
    bool stopped_;
    PlaneViewer* viewer_;
    LineBasedPlaneSegmentation* plane_segmentor_;
    CAMERA_INFO* camera_image_;
    CAMERA_INFO* camera_cloud_;
    //
    bool use_omps_;
    OrganizedPlaneSegmentor* omps_segmentor_;
    //
    pcl::IntegralImageNormalEstimation<PointType, pcl::Normal> ne_;

public:
    int skip_pixel_;
    bool use_horizontal_line_;
    bool use_verticle_line_;
    int y_interval_;
    int x_interval_;

    /** \brief Line extraction */
    float line_point_min_distance_;
    int slide_window_size_;
    int line_min_inliers_;
    float line_fitting_threshold_;

    /** \brief Normals per line */
    int normals_per_line_;
    int normal_smoothing_size_;
    float normal_min_inliers_percentage_;
    float normal_maximum_curvature_;

    /** \brief Remove duplicate candidate if True */
    bool remove_reduplicate_candidate_;
    float reduplicate_candidate_normal_thresh_;
    float reduplicate_candidate_distance_thresh_;

    /** \brief Plane extraction */
    int min_inliers_;
    float max_curvature_;
    float distance_threshold_;
    float neighbor_threshold_;

    /** \brief Refine Plane segmentation result. Note: Not Valid. */
    bool solve_over_segment_;
    bool refine_plane_;
    bool optimize_coefficients_;
    bool project_points_;
    bool extract_boundary_;

    ////
    bool use_normal_cloud_;
    //
    int normal_estimate_method_;
    float normal_estimate_depth_change_factor_;
    float normal_estimate_smoothing_size_;
    float angular_threshold_;

public:
    /// OMPS
    bool is_update_omps_parameters_;
    // parameters
    int omps_ne_method_;
    float omps_ne_max_depth_change_factor_;
    float omps_ne_normal_smoothing_size_;
    int omps_min_inliers_;
    float omps_angular_threshold_;
    float omps_distance_threshold_;
    bool omps_project_bounding_points_;
};

#endif // PLANE_SEGMENT_H

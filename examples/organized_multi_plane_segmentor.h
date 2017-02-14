#ifndef ORGANIZED_MULTI_PLANE_SEGMENTOR_H
#define ORGANIZED_MULTI_PLANE_SEGMENTOR_H

#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ros/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <vector>
#include <iostream>

using namespace std;

typedef std::vector<pcl::PlanarRegion<pcl::PointXYZRGBA>, Eigen::aligned_allocator<pcl::PlanarRegion<pcl::PointXYZRGBA> > >  VectorPlanarRegion;
typedef std::vector<pcl::ModelCoefficients>  VectorModelCoefficient;
typedef std::vector<pcl::PointIndices> VectorPointIndices;

struct OrganizedPlaneSegmentResult {
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_rgba;
    VectorPlanarRegion regions;
    VectorModelCoefficient model_coeffs;
    VectorPointIndices inlier_indices;
    pcl::PointCloud<pcl::Label>::Ptr labels;
    VectorPointIndices label_indices;
    VectorPointIndices boundary_indices;

    inline OrganizedPlaneSegmentResult() :
        labels(new pcl::PointCloud<pcl::Label>)
    {
    }
};

class OrganizedPlaneSegmentor{

public:
    OrganizedPlaneSegmentor(const std::string &file_setting);
    OrganizedPlaneSegmentor();
    void segment(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &input, VectorPlanarRegion &regions);
    void segment(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &input, OrganizedPlaneSegmentResult &result);

    // Each single step
    void updateOrganizedSegmentParameters();
    void normalExtract(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &input,
                       pcl::PointCloud<pcl::Normal>::Ptr &normal_cloud);
    void planeSegment(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &input,
                      pcl::PointCloud<pcl::Normal>::Ptr &normal_cloud,
                      OrganizedPlaneSegmentResult &result);
private:
    // Organisized multiple planar segmentation
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne_;
    pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZRGBA, pcl::Normal, pcl::Label> mps_;

public:
    //
    bool is_update_omps_parameters_;
    // parameters
    int ne_method_;
    float ne_max_depth_change_factor_;
    float ne_normal_smoothing_size_;
    int min_inliers_;
    float angular_threshold_;
    float distance_threshold_;
    bool project_bounding_points_;

};

#endif // ORGANIZED_PLANE_SEGMENTOR_H

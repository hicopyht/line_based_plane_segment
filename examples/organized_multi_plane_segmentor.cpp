#include "organized_multi_plane_segmentor.h"
//
#include "line_based_plane_segmentation.h"

OrganizedPlaneSegmentor::OrganizedPlaneSegmentor(const std::string &file_setting)
    : ne_()
    , mps_()
    , is_update_omps_parameters_( true )
    , ne_method_(0)
    , ne_max_depth_change_factor_(0.02)
    , ne_normal_smoothing_size_(20.0)
    , min_inliers_(3600)
    , angular_threshold_(3.0)
    , distance_threshold_(0.02)
    , project_bounding_points_(true)
{
    cv::FileStorage fs(file_setting, cv::FileStorage::READ);
    //
    using namespace line_based_plane_segment;
    loadIntParam(fs, "OMPS.ne_method", ne_method_, ne_method_);
    loadFloatParam(fs, "OMPS.ne_max_depth_change_factor", ne_max_depth_change_factor_, ne_max_depth_change_factor_);
    loadFloatParam(fs, "OMPS.ne_normal_smoothing_size", ne_normal_smoothing_size_, ne_normal_smoothing_size_);
    loadFloatParam(fs, "OMPS.angular_threshold", angular_threshold_, angular_threshold_);
    loadFloatParam(fs, "OMPS.distance_threshold", distance_threshold_, distance_threshold_);
    loadIntParam(fs, "OMPS.min_inliers", min_inliers_, min_inliers_);
    loadBoolParam(fs, "OMPS.project_bounding_points", project_bounding_points_, project_bounding_points_);

    updateOrganizedSegmentParameters();
}

OrganizedPlaneSegmentor::OrganizedPlaneSegmentor()
    : ne_()
    , mps_()
    , is_update_omps_parameters_( true )
    , ne_method_(0)
    , ne_max_depth_change_factor_(0.02)
    , ne_normal_smoothing_size_(20.0)
    , min_inliers_(3600)
    , angular_threshold_(3.0)
    , distance_threshold_(0.02)
    , project_bounding_points_(true)
{
    updateOrganizedSegmentParameters();
}

void OrganizedPlaneSegmentor::updateOrganizedSegmentParameters()
{
    if( !is_update_omps_parameters_ )
        return;

//    cout << "/******** Organized Multi Plane extractor ********/" << endl;
//    cout << " ne method: " << ne_method_ << endl;
//    cout << " ne_max_depth_change_factor: " << ne_max_depth_change_factor_ << endl;
//    cout << " ne_normal_smoothing_size: " << ne_normal_smoothing_size_ << endl;
//    cout << " -------------------------------------------------" << endl;
//    cout << " min_inliers: " << min_inliers_ << endl;
//    cout << " angular_threshold: " << angular_threshold_ << endl;
//    cout << " distance_threshold: " << distance_threshold_ << endl;
//    cout << " project_bounding_points: " << project_bounding_points_ << endl;
//    cout << "/*************************************************/" << endl;

    switch(ne_method_)
    {
        case 0:
            ne_.setNormalEstimationMethod(ne_.COVARIANCE_MATRIX);
            break;
        case 1:
            ne_.setNormalEstimationMethod(ne_.AVERAGE_3D_GRADIENT);
            break;
        case 2:
            ne_.setNormalEstimationMethod(ne_.AVERAGE_DEPTH_CHANGE);
            break;
        case 3:
            ne_.setNormalEstimationMethod(ne_.SIMPLE_3D_GRADIENT);
            break;
        default:
            ne_.setNormalEstimationMethod(ne_.COVARIANCE_MATRIX);
    }
    ne_.setMaxDepthChangeFactor(ne_max_depth_change_factor_);
    ne_.setNormalSmoothingSize(ne_normal_smoothing_size_);
    //
    mps_.setMinInliers (min_inliers_);
    mps_.setAngularThreshold (0.017453 * angular_threshold_);
//    mps_.setAngularThreshold( DEG_TO_RAD * angular_threshold_ );
    mps_.setDistanceThreshold (distance_threshold_);
    mps_.setProjectPoints(project_bounding_points_);

    is_update_omps_parameters_ = false;
}

void OrganizedPlaneSegmentor::segment(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &input, VectorPlanarRegion &regions)
{
    //
    updateOrganizedSegmentParameters();

    // Calculate Normals
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    ne_.setInputCloud(input);
    ne_.compute(*normal_cloud);

    // Segment
    mps_.setInputNormals(normal_cloud);
    mps_.setInputCloud(input);
    mps_.segmentAndRefine(regions);
}

void OrganizedPlaneSegmentor::segment(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &input, OrganizedPlaneSegmentResult &result)
{
    //
    updateOrganizedSegmentParameters();

    // Calculate Normals
    // Calculate Normals
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    ne_.setInputCloud(input);
    ne_.compute(*normal_cloud);

    // Segment
    mps_.setInputNormals(normal_cloud);
    mps_.setInputCloud(input);
    mps_.segmentAndRefine(result.regions, result.model_coeffs, result.inlier_indices, result.labels, result.label_indices, result.boundary_indices);
}

void OrganizedPlaneSegmentor::normalExtract(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &input,
                                            pcl::PointCloud<pcl::Normal>::Ptr &normal_cloud)
{
    // Calculate Normals
    ne_.setInputCloud(input);
    ne_.compute(*normal_cloud);
}

void OrganizedPlaneSegmentor::planeSegment(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &input,
                                           pcl::PointCloud<pcl::Normal>::Ptr &normal_cloud,
                                           OrganizedPlaneSegmentResult &result)
{
    mps_.setInputNormals(normal_cloud);
    mps_.setInputCloud(input);
    mps_.segmentAndRefine(result.regions, result.model_coeffs, result.inlier_indices,
                          result.labels, result.label_indices, result.boundary_indices);
}

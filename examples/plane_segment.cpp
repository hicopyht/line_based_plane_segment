#include "plane_segment.h"
#include <ros/time.h>

PlaneSegment::PlaneSegment(const string &file_setting)
    : stopped_(false)
    , use_omps_(false)
    , use_normal_cloud_(false)
    , ne_()
    , camera_image_(new CAMERA_INFO(640, 480, 319.5, 239.5, 525.0, 525.0, 1.0))
    , skip_pixel_(2)
{
    cv::FileStorage fs(file_setting, cv::FileStorage::READ);
    //
    cout << " Start LineBasedPlaneSegmentation..." << endl;
    plane_segmentor_ = new LineBasedPlaneSegmentation(file_setting);
    resetPlaneSegmentParameters();

    //
    loadBoolParam(fs, "PlaneSegment.use_normal_cloud", use_normal_cloud_, use_normal_cloud_);

    // Set customized camera parameters
    cout << " 'PlaneSegment' load camera parameters..." << endl;
    loadIntParam(fs, "Camera.width", camera_image_->width, camera_image_->width);
    loadIntParam(fs, "Camera.height", camera_image_->height, camera_image_->height);
    loadFloatParam(fs, "Camera.cx", camera_image_->cx, camera_image_->cx);
    loadFloatParam(fs, "Camera.cy", camera_image_->cy, camera_image_->cy);
    loadFloatParam(fs, "Camera.fx", camera_image_->fx, camera_image_->fx);
    loadFloatParam(fs, "Camera.fy", camera_image_->fy, camera_image_->fy);
    loadFloatParam(fs, "Camera.scale", camera_image_->scale, camera_image_->scale);
    //
    int skip = skip_pixel_;
    loadIntParam(fs, "Camera.skip_pixels", skip, skip);
    setSkipPixel(skip);
    //
    camera_cloud_ = new CAMERA_INFO(*camera_image_, skip_pixel_);

    // Normal estimation
    loadIntParam(fs, "NormalEstimate.method", normal_estimate_method_, 0);
    loadFloatParam(fs, "NormalEstimate.depth_change_factor", normal_estimate_depth_change_factor_, 0.02);
    loadFloatParam(fs, "NormalEstimate.smoothing_size", normal_estimate_smoothing_size_, 20);
    switch(normal_estimate_method_)
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
    ne_.setMaxDepthChangeFactor(normal_estimate_depth_change_factor_);
    ne_.setNormalSmoothingSize(normal_estimate_smoothing_size_);

    //
    plane_segmentor_->setCameraInfo(*camera_cloud_);
    plane_segmentor_->initialize();

    cout << " Start PlaneViewer..." << endl;
    viewer_ = new PlaneViewer(file_setting);
    viewer_->setPlaneSegmentor(this);

    cout << " Run Viewer." << endl;
    // run viewer
    viewer_->spin();
}

PlaneSegment::PlaneSegment(CAMERA_INFO *camera_image)
    : stopped_(false)
    , use_omps_(false)
    , camera_image_(new CAMERA_INFO(*camera_image))
    , skip_pixel_(2)
{
    camera_cloud_ = new CAMERA_INFO(*camera_image, skip_pixel_);
    cout << " Start LineBasedPlaneSegmentation..." << endl;
    plane_segmentor_ = new LineBasedPlaneSegmentation(camera_cloud_);
    resetPlaneSegmentParameters();

    cout << " Start PlaneViewer..." << endl;
    viewer_ = new PlaneViewer();
    viewer_->setPlaneSegmentor(this);

    cout << " Run Viewer." << endl;
    // run viewer
    viewer_->spin();
}

PlaneSegment::PlaneSegment(const string &file_setting, bool use_omps)
    : stopped_(false)
    , use_omps_(true)
    , camera_image_(new CAMERA_INFO(640, 480, 319.5, 239.5, 525.0, 525.0, 1.0))
    , skip_pixel_(2)
{
    cv::FileStorage fs(file_setting, cv::FileStorage::READ);
    // Set customized camera parameters
    cout << " 'OmpsSegment' load camera parameters..." << endl;
    loadIntParam(fs, "Camera.width", camera_image_->width, camera_image_->width);
    loadIntParam(fs, "Camera.height", camera_image_->height, camera_image_->height);
    loadFloatParam(fs, "Camera.cx", camera_image_->cx, camera_image_->cx);
    loadFloatParam(fs, "Camera.cy", camera_image_->cy, camera_image_->cy);
    loadFloatParam(fs, "Camera.fx", camera_image_->fx, camera_image_->fx);
    loadFloatParam(fs, "Camera.fy", camera_image_->fy, camera_image_->fy);
    loadFloatParam(fs, "Camera.scale", camera_image_->scale, camera_image_->scale);
    //
    int skip = skip_pixel_;
    loadIntParam(fs, "Camera.skip_pixels", skip, skip);
    setSkipPixel(skip);
    //
    camera_cloud_ = new CAMERA_INFO(*camera_image_, skip_pixel_);
    //
    cout << " Start OrganizedMultiPlaneSegmentation..." << endl;
    omps_segmentor_ = new OrganizedPlaneSegmentor(file_setting);
    cout << " Start PlaneViewer..." << endl;
    viewer_ = new PlaneViewer(file_setting);
    viewer_->setPlaneSegmentor(this);
    viewer_->setPlaneSegmentor(omps_segmentor_);
    viewer_->setUseOmps(true);


    cout << " Run Viewer." << endl;
    // run viewer
    viewer_->spin();
}

PlaneSegment::~PlaneSegment()
{
    viewer_->finish();
}

void PlaneSegment::segment(const cv::Mat &image_rgb, const cv::Mat &image_depth)
{
    static int skip = camera_image_->width/camera_cloud_->width;


    if(!use_omps_)
    {
        int skip_pixel = skip_pixel_;
        if(skip != skip_pixel)
        {
            camera_cloud_ = new CAMERA_INFO(*camera_image_, skip_pixel);
            plane_segmentor_->setCameraInfo(*camera_cloud_);
            plane_segmentor_->initialize();
            skip = skip_pixel;
        }
        segment(image_rgb, image_depth, *camera_image_, *camera_cloud_);
    }
    // OMPS segmentation
    else
    {
        int skip_pixel = skip_pixel_;
        if(skip != skip_pixel)
        {
            camera_cloud_ = new CAMERA_INFO(*camera_image_, skip_pixel);
            skip = skip_pixel;
        }
        ompsSegment(image_rgb, image_depth, *camera_image_, *camera_cloud_);
    }
}

void PlaneSegment::segment(const cv::Mat &image_rgb, const cv::Mat &image_depth, const CAMERA_INFO &camera_image, const CAMERA_INFO &camera_cloud)
{
    if(stopped_)
        return;

    updatePlaneSegmentParameters();

    std::vector<std::string> procedures;
    std::vector<float> runtimes;
    ros::Time start_dura = ros::Time::now();
    // Get point cloud
    viewer_->setImageRGB(image_rgb);
    viewer_->setImageDepth(image_depth);
    PointCloudTypePtr cloud = getPointCloud(image_rgb, image_depth, camera_image, camera_cloud);
    viewer_->setCloud(cloud);
    pushRuntime(procedures, runtimes, start_dura, "Get Point Cloud");

    //
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>);
    bool use_normal_cloud = use_normal_cloud_;
    if(use_normal_cloud)
    {
        ne_.setInputCloud(cloud);
        ne_.compute(*normal_cloud);
        pushRuntime(procedures, runtimes, start_dura, "Normal Estimation");
    }

    //
    VectorLines lines;
    VectorNormals normals;
    VectorPlanes planes;
    plane_segmentor_->setInputCloud(cloud);
    if(use_normal_cloud)
        plane_segmentor_->setNormalCloud(normal_cloud);

    // Segmentation by step
    if(!plane_segmentor_->initCompute())
    {
        PCL_ERROR("Error initCompute().");
        return;
    }
    pushRuntime(procedures, runtimes, start_dura, "Build Indices Mask");

    std::vector<NormalType> candidates;

    // line fitting using progression approach
    plane_segmentor_->lineFitting( lines );
    pushRuntime(procedures, runtimes, start_dura, "Line Segmentation");

    // compute normal of middle point of line
    plane_segmentor_->normalEstimate( lines, candidates);
    pushRuntime(procedures, runtimes, start_dura, "Normal Estimation");

    // delete reduplicate candidates
    plane_segmentor_->removeDuplicateCandidates( candidates, normals );
    pushRuntime(procedures, runtimes, start_dura, "Reduplicated Removal");

    // iteratively extract plane
//    plane_segmentor_->planeExtraction(lines, candidates, normals, planes, plane_segmentor_->solve_over_segment_);
    plane_segmentor_->planeExtraction2(lines, candidates, normals, planes, plane_segmentor_->solve_over_segment_);
    pushRuntime(procedures, runtimes, start_dura, "Plane Extraction");

    // Refine boundary
    plane_segmentor_->refinePlanes(planes);
    pushRuntime(procedures, runtimes, start_dura, "Refine Boundary");

    //
    plane_segmentor_->deinitCompute();
    //
    viewer_->setSegmentResult(lines, normals, planes);
    viewer_->setRuntimes(procedures, runtimes);
}

void PlaneSegment::ompsSegment(const cv::Mat &image_rgb, const cv::Mat &image_depth, const CAMERA_INFO &camera_image, const CAMERA_INFO &camera_cloud)
{
    if(stopped_)
        return;

    std::vector<std::string> procedures;
    std::vector<float> runtimes;
    ros::Time start_dura = ros::Time::now();
    // Get point cloud
    viewer_->setImageRGB(image_rgb);
    viewer_->setImageDepth(image_depth);
    PointCloudTypePtr cloud = getPointCloud(image_rgb, image_depth, camera_image, camera_cloud);
    viewer_->setCloud(cloud);
    pushRuntime(procedures, runtimes, start_dura, "Get Point Cloud");

    // Do segmentation
    OrganizedPlaneSegmentResult segment_result;
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    omps_segmentor_->updateOrganizedSegmentParameters();
    omps_segmentor_->normalExtract(cloud, normal_cloud);
    pushRuntime(procedures, runtimes, start_dura, "Normal Extraction");
    omps_segmentor_->planeSegment(cloud, normal_cloud, segment_result);
    pushRuntime(procedures, runtimes, start_dura, "Plane Segmentation");
//    omps_segmentor_->segment( cloud, segment_result );
//    pushRuntime(procedures, runtimes, start_dura, "OMPS Segmentation");

    VectorLines lines;
    VectorNormals normals;
    VectorPlanes planes;
    // convert format
    for( int i = 0; i < segment_result.regions.size(); i++)
    {
        pcl::ModelCoefficients &coef = segment_result.model_coeffs[i];
        pcl::PointIndices &indices = segment_result.inlier_indices[i];
        pcl::PlanarRegion<PointType> &pr = segment_result.regions[i];
        pcl::PointIndices &boundary = segment_result.boundary_indices[i];
        //
        PlaneType plane;
        Eigen::Vector3f centroid = pr.getCentroid();
        plane.centroid.x = centroid[0];
        plane.centroid.y = centroid[1];
        plane.centroid.z = centroid[2];
        plane.coefficients[0] = coef.values[0];
        plane.coefficients[1] = coef.values[1];
        plane.coefficients[2] = coef.values[2];
        plane.coefficients[3] = coef.values[3];
        plane.indices = indices.indices;
        plane.boundary_indices = boundary.indices;
        plane.hull_indices = boundary.indices;
        planes.push_back( plane );
    }
    //
    pushRuntime(procedures, runtimes, start_dura, "Convert Plane Type");
    //
    viewer_->setSegmentResult(lines, normals, planes);
    viewer_->setRuntimes(procedures, runtimes);
}


void PlaneSegment::setSkipPixel(int skip)
{
    if( skip == 1 || skip == 2 || skip == 4 || skip == 8 || skip == 16)
    {
        skip_pixel_ = skip;
    }
}

void PlaneSegment::resetPlaneSegmentParameters()
{
    //
    use_horizontal_line_ = plane_segmentor_->use_horizontal_line_;
    use_verticle_line_  = plane_segmentor_->use_verticle_line_;
    y_interval_  = plane_segmentor_->y_interval_;
    x_interval_  = plane_segmentor_->x_interval_;


    /** \brief Line extraction */
    line_point_min_distance_  = plane_segmentor_->line_point_min_distance_;
    slide_window_size_  = plane_segmentor_->slide_window_size_;
    line_min_inliers_  = plane_segmentor_->line_min_inliers_;
    line_fitting_threshold_  = plane_segmentor_->line_fitting_threshold_;

    /** \brief Normals per line */
    normals_per_line_  = plane_segmentor_->normals_per_line_;
    normal_smoothing_size_  = plane_segmentor_->normal_smoothing_size_;
    normal_min_inliers_percentage_  = plane_segmentor_->normal_min_inliers_percentage_;
    normal_maximum_curvature_  = plane_segmentor_->normal_maximum_curvature_;

    /** \brief Remove duplicate candidate if True */
    remove_reduplicate_candidate_  = plane_segmentor_->remove_reduplicate_candidate_;
    reduplicate_candidate_normal_thresh_  = plane_segmentor_->reduplicate_candidate_normal_thresh_;
    reduplicate_candidate_distance_thresh_  = plane_segmentor_->reduplicate_candidate_distance_thresh_;

    /** \brief Plane extraction */
    min_inliers_  = plane_segmentor_->min_inliers_;
    max_curvature_  = plane_segmentor_->max_curvature_;
    distance_threshold_  = plane_segmentor_->distance_threshold_;
    neighbor_threshold_  = plane_segmentor_->neighbor_threshold_;

    /** \brief Refine Plane segmentation result. Note: Not Valid. */
    solve_over_segment_ = plane_segmentor_->solve_over_segment_;
    refine_plane_  = plane_segmentor_->refine_plane_;
    optimize_coefficients_  = plane_segmentor_->optimize_coefficients_;
    project_points_  = plane_segmentor_->project_points_;
    extract_boundary_  = plane_segmentor_->extract_boundary_;

    //
    angular_threshold_ = plane_segmentor_->angular_threshold_;
}

void PlaneSegment::updatePlaneSegmentParameters()
{
    bool update_selected = false;
    //
    plane_segmentor_->use_horizontal_line_ = use_horizontal_line_;
    plane_segmentor_->use_verticle_line_ = use_verticle_line_;
    if( plane_segmentor_->y_interval_ != y_interval_ )
    {
        plane_segmentor_->y_interval_ = y_interval_;
        update_selected = true;
    }
    if( plane_segmentor_->x_interval_ != x_interval_ )
    {
        plane_segmentor_->x_interval_ = x_interval_;
        update_selected = true;
    }
    //
    if( update_selected )
    {
        plane_segmentor_->updateSelectedRowsAndCols();
    }

    /** \brief Line extraction */
    plane_segmentor_->line_point_min_distance_ = line_point_min_distance_;
    plane_segmentor_->setLineRegressionParams(slide_window_size_, line_fitting_threshold_, line_min_inliers_);
//    plane_segmentor_->slide_window_size_ = slide_window_size_;
//    plane_segmentor_->line_min_inliers_ = line_min_inliers_;
//    plane_segmentor_->line_fitting_threshold_ = line_fitting_threshold_;

    /** \brief Normals per line */
    plane_segmentor_->normals_per_line_ = normals_per_line_;
    plane_segmentor_->normal_smoothing_size_ = normal_smoothing_size_;
    plane_segmentor_->normal_min_inliers_percentage_ = normal_min_inliers_percentage_;
    plane_segmentor_->normal_maximum_curvature_ = normal_maximum_curvature_;

    /** \brief Remove duplicate candidate if True */
    plane_segmentor_->remove_reduplicate_candidate_ = remove_reduplicate_candidate_;
    plane_segmentor_->reduplicate_candidate_normal_thresh_ = reduplicate_candidate_normal_thresh_;
    plane_segmentor_->reduplicate_candidate_distance_thresh_ = reduplicate_candidate_distance_thresh_;

    /** \brief Plane extraction */
    plane_segmentor_->min_inliers_ = min_inliers_;
    plane_segmentor_->max_curvature_ = max_curvature_;
    plane_segmentor_->distance_threshold_ = distance_threshold_;
    plane_segmentor_->neighbor_threshold_ = neighbor_threshold_;

    /** \brief Refine Plane segmentation result. Note: Not Valid. */
    plane_segmentor_->solve_over_segment_ = solve_over_segment_;
    plane_segmentor_->refine_plane_ = refine_plane_;
    plane_segmentor_->optimize_coefficients_ = optimize_coefficients_;
    plane_segmentor_->project_points_ = project_points_;
    plane_segmentor_->extract_boundary_ = extract_boundary_;

    //
    plane_segmentor_->angular_threshold_ = angular_threshold_;
}


PointCloudTypePtr PlaneSegment::getPointCloud(const cv::Mat &image_rgb, const cv::Mat &image_depth,
                                              const CAMERA_INFO &camera_image, const CAMERA_INFO &camera_cloud)
{
    PointCloudTypePtr cloud( new PointCloudType);
    cloud->is_dense = false;
    cloud->width = camera_cloud.width;
    cloud->height = camera_cloud.height;
    cloud->points.resize(cloud->width * cloud->height);

    const float invfx = 1.0 / camera_image.fx;
    const float invfy = 1.0 / camera_image.fy;
    const float scale = 1.0 / camera_image.scale;
//    const float min_depth = range_min_depth_;
    const float min_depth = 0.1;
    pcl::PointCloud<pcl::PointXYZRGBA>::iterator pt_iter = cloud->begin();
    int skip = camera_image.width / camera_cloud.width;
    int depth_idx = 0;
    int color_idx = 0;
    int color_skip_idx = 3 * skip;
    for (int v = 0; v < image_depth.rows; v+=skip)
    {
        depth_idx = v *image_depth.cols;
        color_idx = depth_idx * 3;
        for (int u = 0; u < image_depth.cols; u+=skip)
        {
            if(pt_iter == cloud->end())
            {
                break;
            }
            pcl::PointXYZRGBA &pt = *pt_iter;
            float Z = image_depth.at<float>(depth_idx) * scale;
            // Check for invalid measurements
            if (Z <= min_depth || isnan(Z)) //Should also be trigger on NaN//std::isnan (Z))
            {
                pt.x = (u - camera_image.cx) * 1.0 * invfx; //FIXME: better solution as to act as at 1meter?
                pt.y = (v - camera_image.cy) * 1.0 * invfy;
                pt.z = std::numeric_limits<float>::quiet_NaN();
            }
            else // Fill in XYZ
            {
                pt.x = (u - camera_image.cx) * Z * invfx;
                pt.y = (v - camera_image.cy) * Z * invfy;
                pt.z = Z;
            }

            RGBValue color;

            color.Blue = image_rgb.at<uint8_t>(color_idx);
            color.Green = image_rgb.at<uint8_t>(color_idx+1);
            color.Red = image_rgb.at<uint8_t>(color_idx+2);
            color.Alpha = 255.0;
            pt.rgb = color.float_value;

            //
            pt_iter ++;
            depth_idx += skip;
            color_idx += color_skip_idx;
        }
    }

    return cloud;
}

#include "line_based_plane_segmentation.h"

//#define DEBUG

#ifdef DEBUG
#include <ros/ros.h>
#endif

/*
 * TODO:
 * - Parameter depend on depth:
 *   distance_threshold
 *   neighbour_threshold
 *   line_neighbour_threshold
 *   (line_fidelity_threshold, depend on point error)
 *   normal_curvature
 *   normal_neighbour_dis
 */


namespace line_based_plane_segment
{

#ifdef DEBUG
float getScopeTime(ros::Time &start)
{
    float dura = (ros::Time::now()-start).toSec()*1000.0f;
    start = ros::Time::now();
    return dura;
}
#endif

const cv::Point2i neighbor8_dir[8] = {cv::Point2i(-1, -1), cv::Point2i(0, -1), cv::Point2i(1, -1),
                                     cv::Point2i(-1, 0),                      cv::Point2i(1, 0),
                                     cv::Point2i(-1, 1), cv::Point2i(0, 1), cv::Point2i(1, 1)};

/*------------------------------------------------------------------------------------------*/
LineBasedPlaneSegmentation::LineBasedPlaneSegmentation(const std::string &setting_file)
    : input_(NULL)
    , normals_(NULL)
    , indices_mask_(cv::Mat())
    , ne_()
    , prttcp_(new pcl::DefaultPointRepresentation<PointT>)
    , initialized_(false)
    , compute_initialized_(false)
    , use_horizontal_line_(true)
    , use_verticle_line_(true)
    , y_interval_(10)
    , x_interval_(10)
    , line_point_min_distance_(0.08f)
    , line_fitting_angular_threshold_(3.0)
    , line_fitting_min_indices_(15)
    , normals_per_line_(1)
    , normal_smoothing_size_(12)
    , normal_min_inliers_percentage_(0.6f)
    , normal_maximum_curvature_(0.005f)
    , remove_reduplicate_candidate_(true)
    , reduplicate_candidate_normal_thresh_(0.08f)
    , reduplicate_candidate_distance_thresh_(0.02f)
    , min_inliers_(600)
    , max_curvature_(0.005f)
    , distance_threshold_(0.02f)
    , neighbor_threshold_(0.2f)
    , angular_threshold_(10.0)
    , solve_over_segment_(true)
    , refine_plane_(false)
    , optimize_coefficients_(true)
    , project_points_(false)
    , extract_boundary_(true)
    , normal_estimate_method_(0)
    , normal_estimate_depth_change_factor_(0.05)
    , normal_estimate_smoothing_size_(11)
{
    cout << WHITE << "  Load plane parameters:" << RESET << endl;
    cv::FileStorage fs(setting_file, cv::FileStorage::READ);
    //
    CAMERA_INFO camera_image(640, 480, 319.5, 239.5, 525.0, 525.0, 1.0);
    loadIntParam(fs, "Camera.width", camera_image.width, camera_image.width);
    loadIntParam(fs, "Camera.height", camera_image.height, camera_image.height);
    loadFloatParam(fs, "Camera.cx", camera_image.cx, camera_image.cx);
    loadFloatParam(fs, "Camera.cy", camera_image.cy, camera_image.cy);
    loadFloatParam(fs, "Camera.fx", camera_image.fx, camera_image.fx);
    loadFloatParam(fs, "Camera.fy", camera_image.fy, camera_image.fy);
    loadFloatParam(fs, "Camera.scale", camera_image.scale, camera_image.scale);
    //
    int skip = 2;
    loadIntParam(fs, "Camera.skip_pixels", skip, skip);
    camera_info_ = new CAMERA_INFO(camera_image, skip);

    //
    cout << MAGENTA << "  - Set cloud parameters:" << RESET << endl;
    cout << WHITE << "    width:  " << GREEN << camera_info_->width << RESET << endl;
    cout << WHITE << "    height: " << GREEN << camera_info_->height << RESET << endl;
    cout << WHITE << "    cx:     " << GREEN << camera_info_->cx << RESET << endl;
    cout << WHITE << "    cy:     " << GREEN << camera_info_->cy << RESET << endl;
    cout << WHITE << "    fx:     " << GREEN << camera_info_->fx << RESET << endl;
    cout << WHITE << "    ft:     " << GREEN << camera_info_->fy << RESET << endl;
    cout << WHITE << "    scale:  " << GREEN << camera_info_->scale << RESET << endl;
    //
    std::string prefix = "PlaneSegment.";
    //
    loadBoolParam(fs, prefix+"use_horizontal_line", use_horizontal_line_, use_horizontal_line_);
    loadBoolParam(fs, prefix+"use_verticle_line", use_verticle_line_, use_verticle_line_);
    loadIntParam(fs, prefix+"y_interval", y_interval_, y_interval_);
    loadIntParam(fs, prefix+"x_interval", x_interval_, x_interval_);
    //
    loadFloatParam(fs, prefix+"line_point_min_distance", line_point_min_distance_, line_point_min_distance_);
    loadFloatParam(fs, prefix+"line_fitting_angular_threshold", line_fitting_angular_threshold_, line_fitting_angular_threshold_);
    loadIntParam(fs, prefix+"line_fitting_min_indices", line_fitting_min_indices_, line_fitting_min_indices_);
    //
    loadIntParam(fs, prefix+"normals_per_line", normals_per_line_, normals_per_line_);
    loadIntParam(fs, prefix+"normal_smoothing_size", normal_smoothing_size_, normal_smoothing_size_);
    loadFloatParam(fs, prefix+"normal_min_inliers_percentage", normal_min_inliers_percentage_, normal_min_inliers_percentage_);
    loadFloatParam(fs, prefix+"normal_maximum_curvature", normal_maximum_curvature_, normal_maximum_curvature_);
    //
    loadBoolParam(fs, prefix+"remove_reduplicate_candidate", remove_reduplicate_candidate_, remove_reduplicate_candidate_);
    loadFloatParam(fs, prefix+"reduplicate_candidate_normal_thresh", reduplicate_candidate_normal_thresh_, reduplicate_candidate_normal_thresh_);
    loadFloatParam(fs, prefix+"reduplicate_candidate_distance_thresh", reduplicate_candidate_distance_thresh_, reduplicate_candidate_distance_thresh_);
    //
    loadIntParam(fs, prefix+"min_inliers", min_inliers_, min_inliers_);
    loadFloatParam(fs, prefix+"max_curvature", max_curvature_, max_curvature_);
    loadFloatParam(fs, prefix+"distance_threshold", distance_threshold_, distance_threshold_);
    loadFloatParam(fs, prefix+"neighbor_threshold", neighbor_threshold_, neighbor_threshold_);
    loadFloatParam(fs, prefix+"angular_threshold", angular_threshold_, angular_threshold_);
    //
    loadBoolParam(fs, prefix+"solve_over_segment", solve_over_segment_, solve_over_segment_);
    loadBoolParam(fs, prefix+"refine_plane", refine_plane_, refine_plane_);
    loadBoolParam(fs, prefix+"optimize_coefficients", optimize_coefficients_, optimize_coefficients_);
    loadBoolParam(fs, prefix+"project_points", project_points_, project_points_);
    loadBoolParam(fs, prefix+"extract_boundary", extract_boundary_, extract_boundary_);
    // Normal estimation
    loadIntParam(fs, "NormalEstimate.method", normal_estimate_method_, normal_estimate_method_);
    loadFloatParam(fs, "NormalEstimate.depth_change_factor", normal_estimate_depth_change_factor_, normal_estimate_depth_change_factor_);
    loadFloatParam(fs, "NormalEstimate.smoothing_size", normal_estimate_smoothing_size_, normal_estimate_smoothing_size_);

    setNormalEstimateParams(normal_estimate_method_, normal_estimate_depth_change_factor_, normal_estimate_smoothing_size_);
    //
    initialize();
}

LineBasedPlaneSegmentation::LineBasedPlaneSegmentation(CAMERA_INFO *camera)
    : input_(NULL)
    , normals_(NULL)
    , indices_mask_(cv::Mat())
    , ne_()
    , prttcp_(new pcl::DefaultPointRepresentation<PointT>)
    , camera_info_(new CAMERA_INFO(*camera))
    , initialized_(false)
    , compute_initialized_(false)
    , use_horizontal_line_(true)
    , use_verticle_line_(true)
    , y_interval_(10)
    , x_interval_(10)
    , line_point_min_distance_(0.08f)
    , line_fitting_angular_threshold_(3.0)
    , line_fitting_min_indices_(15)
    , normals_per_line_(1)
    , normal_smoothing_size_(12)
    , normal_min_inliers_percentage_(0.6f)
    , normal_maximum_curvature_(0.005f)
    , remove_reduplicate_candidate_(true)
    , reduplicate_candidate_normal_thresh_(0.08f)
    , reduplicate_candidate_distance_thresh_(0.02f)
    , min_inliers_(600)
    , max_curvature_(0.005f)
    , distance_threshold_(0.02f)
    , neighbor_threshold_(0.2f)
    , angular_threshold_(10.0)
    , solve_over_segment_(true)
    , refine_plane_(false)
    , optimize_coefficients_(true)
    , project_points_(false)
    , extract_boundary_(true)
    , normal_estimate_method_(0)
    , normal_estimate_depth_change_factor_(0.05)
    , normal_estimate_smoothing_size_(11)
{
    setNormalEstimateParams(normal_estimate_method_, normal_estimate_depth_change_factor_, normal_estimate_smoothing_size_);
    //
    initialize();
}

//
void LineBasedPlaneSegmentation::initialize()
{
    if(!camera_info_)
    {
        cout << "[Error]: LineBasedPlaneSegmentation need to set camera parameters." << endl;
        exit(-1);
    }

    // set cloud width and height
    cloud_width_ = camera_info_->width;
    cloud_height_ = camera_info_->height;

    // get pixel line loc
    z_factor_row_.clear();
    z_factor_row_.resize( cloud_height_ );
    float fy2 = pow(camera_info_->fy, 2);
    for(int i = 0; i < cloud_height_; i++)
    {
        z_factor_row_[i] = sqrt( 1 + ( pow((i - camera_info_->cy), 2) / fy2 ) );
    }
    //
    z_factor_col_.clear();
    z_factor_col_.resize( cloud_width_ );
    float fx2 = pow(camera_info_->fx, 2);
    for(int i = 0; i < cloud_width_; i++)
    {
        z_factor_col_[i] = sqrt( 1 + ( pow((i - camera_info_->cx), 2) / fx2 ) );
    }

    // Update selected rows and cols
    updateSelectedRowsAndCols();

    //
    initialized_ = true;
}

//
void LineBasedPlaneSegmentation::updateSelectedRowsAndCols()
{
    selected_rows_.clear();
    const int r_space = y_interval_;
    const int mid_r = cloud_height_ / 2 - 1;
    selected_rows_.push_back( mid_r );
    // decrease
    for( int r = mid_r - r_space; (r - r_space*0.5) >= 0; r -= r_space)
    {
        selected_rows_.push_back( r );
    }
    // increase
    for( int r = mid_r + r_space; (r+r_space*0.5) < cloud_height_; r += r_space)
    {
        selected_rows_.push_back( r );
    }

    selected_cols_.clear();
    const int c_space = x_interval_;
    const int mid_c = cloud_width_ / 2 - 1;
    selected_cols_.push_back( mid_c );
    // decrease
    for( int c = mid_c - c_space; (c - c_space*0.5) >= 0; c -= c_space)
    {
        selected_cols_.push_back( c );
    }
    // increase
    for( int c = mid_c + c_space; (c + c_space*0.5) < cloud_width_; c += c_space)
    {
        selected_cols_.push_back( c );
    }
}


//
bool LineBasedPlaneSegmentation::initCompute()
{
    // Check if some internal variables are initialized
    if(!initialized_)
    {
        initialize();
    }

    // Check if input was set
    if(!input_)
    {
        PCL_ERROR("Error: input cloud was not set.");
        return (false);
    }
    // Check if size of input cloud is feasible
    if(input_->width != cloud_width_ || input_->height != cloud_height_ )
    {
        PCL_ERROR("Error: input cloud width and height are not matched with those from camera parameters.");
        return (false);
    }

    // Check if has normal cloud
    if(normals_ == NULL || normals_->width != cloud_width_ || normals_->height != cloud_height_)
    {
        normals_ = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
        ne_.setInputCloud(input_);
        ne_.compute(*normals_);
    }

//    // Check if size of normal cloud is feasible
//    if(normals_->width != cloud_width_ || normals_->height != cloud_height_)
//    {
//        PCL_ERROR("Error: Normal cloud width and height are not matched with those from camera parameters.");
//        return (false);
//    }

    // Check if indices mask is valid, otherwise build a mask.
    if( !validMask() )
    {
        if(input_->is_dense)
        {
            indices_mask_ = cv::Mat::ones(cloud_height_, cloud_width_, CV_8UC1);
        }
        else
        {
//            cout << " - LBPS: build indices mask from pointcloud." << endl;
            // check valid point, build a pointcloud mask
            indices_mask_ = cv::Mat::zeros(cloud_height_, cloud_width_, CV_8UC1);
            PointCloud::const_iterator pt_iter, pt_end;
            pt_iter = input_->begin();
            pt_end = input_->end();
            cv::MatIterator_<uchar> mask_iter, mask_end;
            mask_iter = indices_mask_.begin<uchar>();
            mask_end = indices_mask_.end<uchar>();
            //
            for (; (pt_iter != pt_end) && (mask_iter!=mask_end); ++pt_iter, ++mask_iter)
            {
                if( isValidPoint( *pt_iter ) )
                {
                    *mask_iter = 255;
                }
            }
        }
    }

    compute_initialized_ = true;
    return true;
}

//
void LineBasedPlaneSegmentation::deinitCompute()
{
    input_ = NULL;
    normals_ = NULL;
    indices_mask_ = cv::Mat();
    compute_initialized_ = false;
}

//
void LineBasedPlaneSegmentation::segment(std::vector<PlaneType>& planes)
{
    std::vector<LineType> lines;
    std::vector<NormalType> normals;
    segment(lines, normals, planes);
}

//
void LineBasedPlaneSegmentation::segment(std::vector<LineType>& lines,
                                   std::vector<NormalType>& normals,
                                   std::vector<PlaneType>& planes)
{
    //
    if(!initCompute())
    {
        PCL_ERROR("Error initCompute().");
        return;
    }

    std::vector<NormalType> candidates;

    // line fitting using progression approach
    lineFitting(lines);

    // compute normal of middle point of line
    candidateDetection(lines, candidates);

    // delete duplicate candidates
    removeDuplicateCandidates( candidates, normals );

    // iteratively extract plane
    planeExtraction(lines, candidates, normals, planes, solve_over_segment_);

    // Refine boundary
    refinePlanes(planes);

    deinitCompute();
}


/****************************************************************************************/
// Each individual step of segmentation
//
void LineBasedPlaneSegmentation::lineFitting(std::vector<LineType>& lines)
{
    // Roughly get line regions
    std::vector<LineType> regions;
    lineRegions(regions);

    // Segment lines
    lineSegment(regions, lines);
}

//
void LineBasedPlaneSegmentation::candidateDetection( std::vector<LineType>& lines, std::vector<NormalType>& normals)
{
    for(size_t i = 0; i < lines.size(); i++)
    {
        LineType &line = lines[i];
        int len = line.indices.size();
        int step = len / normals_per_line_;
        for(int c = step/2; c < len; c += step)
        {
            NormalType normal;
            normal.line_index = i;
            normal.over_segment = true;
            computeNormal( line.indices[c], normal );
            if( normal.valid )
            {
                normals.push_back( normal );
            }
        }
    }
}

//
void LineBasedPlaneSegmentation::removeDuplicateCandidates( std::vector<NormalType>& normals, std::vector<NormalType>& output)
{
    if( remove_reduplicate_candidate_ )
    {
        if( normals.size() <= 1 )
            return;

        // compute local coordinate
        std::vector<bool> duplicates;
        for( size_t i = 0; i < normals.size(); i++)
        {
            NormalType &normal = normals[i];
            if(!normal.valid)
                continue;
            normal.n << normal.coefficients[0], normal.coefficients[1], normal.coefficients[2];
            normal.d = normal.coefficients[3];
            normalTangentCoordinate( normal, normal.basis );
            duplicates.push_back( false );
        }

        // find duplicate
        for( int i = 0; i < normals.size()-1; i++)
        {
            if(duplicates[i])
                continue;

            for( size_t j = i+1; j < normals.size(); j++)
            {
                if(duplicates[j])
                    continue;

//                Eigen::Vector2f error_dir = normalError( normals[i], normals[j] );
//                float error_d = fabs(normals[i].d - normals[j].d);
//                if( (fabs(error_dir[0]) + fabs(error_dir[1])) < reduplicate_candidate_normal_thresh_
//                        && error_d < reduplicate_candidate_distance_thresh_ )
                if(isSimilarPlanes(normals[i], normals[j], reduplicate_candidate_normal_thresh_, reduplicate_candidate_distance_thresh_))
                {
                    // one with minimal curvature survives
                    if( normals[i].curvature > normals[j].curvature )
                        duplicates[i] = true;
                    else
                        duplicates[j] = true;
                }
            }
        }

        // construct output
        output.clear();
        for( size_t i = 0; i < duplicates.size(); i++)
        {
            if( !duplicates[i] )
                output.push_back( normals[i] );
        }
    }
    else
    {
        output = normals;
    }
}

void LineBasedPlaneSegmentation::planeExtraction(std::vector<LineType> &lines,
                                                 std::vector<NormalType> &line_normals,
                                                 std::vector<NormalType>& normals,
                                                 std::vector<PlaneType>& final_planes,
                                                 bool solve_over_segment)
{
#ifdef DEBUG
    std::cout << WHITE << "Extract planes:" << RESET;
#endif

    int iteration = 0;

#ifdef DEBUG
    ros::Time start_dura = ros::Time::now();
#endif

    std::vector<PlaneType> planes;
    //
    const int min_normal_indices = normal_min_inliers_percentage_ * normal_smoothing_size_ * normal_smoothing_size_;

    // compute possible planes
    extractAllPossiblePlanes(normals, planes);
    //
#ifdef DEBUG
    cout << "EA size: " << BOLDCYAN;
#endif
    for(std::vector<PlaneType>::iterator itp = planes.begin(), end = planes.end(); itp != end; itp++)
    {
        itp->over_segment = false;
#ifdef DEBUG
        cout << " " << itp->indices.size();
#endif
    }
#ifdef DEBUG
    cout << RESET << endl;
#endif


    iteration++;
#ifdef DEBUG
        cout << " EALL:" << BOLDYELLOW << getScopeTime(start_dura) << RESET << endl;
        cout << BOLDWHITE << " Iteration: " << iteration << RESET;
#endif

    // Get one index
    int max_index = getOnePlaneIndex(lines, line_normals, normals, planes, solve_over_segment);


    while(max_index >= 0)
    {
        PlaneType &spl = planes[max_index];
        NormalType &snr = normals[spl.normal_index];
        bool check_os = !spl.over_segment;
        // Save last indices mask
        cv::Mat last_indices_mask;
        indices_mask_.copyTo(last_indices_mask);

        // Extract one plane, delete indices
        PlaneType respl;
        respl.coefficients = spl.coefficients;
        extractFinalPlane(respl, snr.center_index, false);
        //
        if(respl.indices.size() >= min_inliers_)
        {
            deleteInliers(respl.indices);
        }
        else
        {
            spl.valid = false;
#ifdef DEBUG
            cout << BOLDWHITE << " EXT: " << BOLDYELLOW << getScopeTime(start_dura) << RESET;
            cout << endl;
#endif

            iteration++;
#ifdef DEBUG
            cout << BOLDWHITE << " Iteration: " << iteration << RESET;
#endif
            // Get next index
            max_index = getOnePlaneIndex(lines, line_normals, normals, planes, solve_over_segment);
            continue;
        }

        // Check if the result will lead to over segmentation
        std::vector<PlaneType> remain_planes;
        int os_index = -1;
        int size_indices, size_remain;
        for(size_t i = 0; i < planes.size(); i++)
        {
            if(i == max_index)
                continue;

            PlaneType &pl = planes[i];
            NormalType &nr = normals[pl.normal_index];

            if(!pl.valid || !nr.valid)
            {
                pl.valid = false;
                nr.valid = false;
                continue;
            }

            // Invalid selected point, continue
            if(indices_mask_.at<uchar>(nr.center_index) == 0)   // invalid supporting point
            {
                nr.valid = false;
                continue;
            }

            // Invalid normal, continue
            if(countNormalNeighbors(nr.center_index) < min_normal_indices)  // invalid normal
            {
                nr.valid = false;
                continue;
            }

            // Get remaining indices
            PlaneType rpl;
            selectConnectedPlaneRegion(nr.center_index, pl.coefficients, distance_threshold_, rpl.indices);

            if(rpl.indices.size() < min_inliers_)
            {
                continue;
            }

            // Check number of remainning indices
            if( check_os && rpl.indices.size()*1.21 < pl.indices.size() )   // lead to over-segmentation
//            if( check_os && (pl.indices.size() - rpl.indices.size() > cloud_width_ * 3) )   // lead to over-segmentation
            {
                os_index = i;
                size_remain = rpl.indices.size();
                size_indices = pl.indices.size();
                break;
            }
            else
            {
                rpl.valid = true;
                rpl.normal_index = pl.normal_index;
                rpl.coefficients = pl.coefficients;
                rpl.over_segment = nr.over_segment;
                remain_planes.push_back(rpl);
            }
        }

        if(check_os && os_index >= 0)
        {
            //
            iteration++;
            if(iteration>=16)
                break;
#ifdef DEBUG

            cout << BOLDWHITE << " IOS: " << BOLDRED << os_index
                 << BOLDWHITE << " " << BOLDGREEN << size_remain
                 << BOLDWHITE << "/" << BOLDBLUE << size_indices
                 << BOLDWHITE << " T: " << BOLDYELLOW << getScopeTime(start_dura)
                 << BOLDWHITE << " Retry" << RESET << endl;
            //
            cout << BOLDWHITE << " Iteration: " << BOLDCYAN << iteration << RESET;
#endif
            spl.over_segment = true;
            indices_mask_ = last_indices_mask;
            max_index = getOnePlaneIndex(lines, line_normals, normals, planes, solve_over_segment);;
//            max_index = os_index;
//            computePlaneCoefficient(planes[max_index].indices, planes[max_index]);

            continue;
        }

        // save final plane
        final_planes.push_back( respl );
        snr.valid = false;

        //
        planes.swap(remain_planes);

        // Check if solve over segmentation is needed
        if(solve_over_segment)
        {
            int count = 0;
            for( size_t i = 0; i < planes.size(); i++)
            {
                if(planes[i].over_segment)
                    count ++;
            }
            if(count <= 0)
                solve_over_segment = false;
        }


#ifdef DEBUG
        cout << BOLDWHITE << " EXT: " << BOLDYELLOW << getScopeTime(start_dura) << RESET;
        cout << endl;
#endif

        iteration++;
#ifdef DEBUG
        cout << BOLDWHITE << " Iteration: " << iteration << RESET;
#endif
        // Get next index
        max_index = getOnePlaneIndex(lines, line_normals, normals, planes, solve_over_segment);
    }

#ifdef DEBUG
    cout << RESET << endl;
#endif
}

//
void LineBasedPlaneSegmentation::refinePlanes(std::vector<PlaneType>& final_planes)
{
    if(!refine_plane_)
        return;

    const float threshold = distance_threshold_;
    const float square_threshold = neighbor_threshold_ * neighbor_threshold_;

    for(size_t i = 0; i < final_planes.size(); i++)
    {
        PlaneType &pl = final_planes[i];
        // Build search list
        std::stack<cv::Point2i> search_list;
        if(pl.boundary_indices.size() > 0)
        {
            for(std::vector<int>::iterator it = pl.boundary_indices.begin(), end = pl.boundary_indices.end();
                it != end; it++)
            {
                cv::Point2i p2i;
                p2i.y = (*it)/cloud_width_;
                p2i.x = *it - p2i.y*cloud_width_;
                search_list.push(p2i);
            }
        }
        else
        {
            for(std::vector<int>::iterator it = pl.indices.begin(), end = pl.indices.end();
                it != end; it++)
            {
                cv::Point2i p2i;
                p2i.y = (*it)/cloud_width_;
                p2i.x = *it - p2i.y*cloud_width_;
                search_list.push(p2i);
            }
        }

        cv::Mat mask = indices_mask_;   // actually the same mat
        // loop while search list is not empty
        while( search_list.size() > 0)
        {
            cv::Point2i pc = search_list.top();
            const PointT &pcenter = input_->points[ pc.y * cloud_width_ + pc.x];
            search_list.pop();
            for( int i =0; i < 8; i ++)
            {
                // pick one neighbor
                cv::Point2i pn = pc + neighbor8_dir[i];
                if( pn.x < 0 || pn.x >= cloud_width_ || pn.y < 0 || pn.y >= cloud_height_ )
                    continue;
                int idx = pn.y * cloud_width_ + pn.x;
                const PointT &pneighbor = input_->points[idx];
                if( mask.at<uchar>( idx )
                        && checkPointWithDistanceToPlane( idx, pl.coefficients, threshold)
                        && squareDistancePoint2Point( pcenter,  pneighbor) < square_threshold )
                {
                    // add to indices
                    pl.indices.push_back( idx );
                    mask.at<uchar>(idx) = 0;
                    // add to search list
                    search_list.push( pn );
                }
            } // end for
        } // end while

        //
        if(extract_boundary_)
        {
            extractBoundary(pl.indices, pl.boundary_indices, pl.hull_indices);
        }
    }

}
/****************************************************************************************/
/****************************************************************************************/
// Line extraction
void LineBasedPlaneSegmentation::lineRegionRow(int row, std::vector<LineType> &regions)
{
    const float sqare_distance_thresh = line_point_min_distance_ * line_point_min_distance_;
    const unsigned minimum_inlier = line_fitting_min_indices_;

    const int begin = row * camera_info_->width;
    const int end = begin + camera_info_->width - 1;

    LineType region;

    int pre = begin;
    while( !indices_mask_.at<uchar>(pre) )
    {
        pre ++;
        if(pre >= end)
            return;
    }
    region.indices.push_back(pre);
    int cur = pre + 1;
    while(cur <= end)
    {
        while( !indices_mask_.at<uchar>(pre) )
        {
            cur ++;
            if(cur > end)
                break;
        }

        if(cur > end)
            break;

        if(squareDistancePoint2Point( input_->points[pre], input_->points[cur] ) < sqare_distance_thresh)
        {
            region.indices.push_back(cur);
        }
        else
        {
            if(region.indices.size() >= minimum_inlier)
                regions.push_back( region );
            // for next iteration
            region.indices.clear();
        }
        pre = cur;
        cur ++;
    }
    // for the last region
    if( region.indices.size() >= minimum_inlier)
        regions.push_back( region );
}

//
void LineBasedPlaneSegmentation::lineRegionCol(int col, std::vector<LineType> &regions)
{
    const float sqare_distance_thresh = line_point_min_distance_ * line_point_min_distance_;
    const unsigned minimum_inlier = line_fitting_min_indices_;

    const int begin = col;
    const int step = camera_info_->width;
    int count = camera_info_->height;
    LineType region;

    int pre = begin;
    while( !indices_mask_.at<uchar>(pre) )
    {
        pre += step;
        count --;
        if(count <= 0)
            return;
    }
    region.indices.push_back(pre);
    int cur = pre + step;
    count --;
    while( count > 0)
    {
        while( !indices_mask_.at<uchar>(pre) )
        {
            cur += step;
            count --;
            if( count <= 0)
                break;
        }
        if( count <= 0)
            break;

        if(squareDistancePoint2Point( input_->points[pre], input_->points[cur] ) < sqare_distance_thresh)
        {
            region.indices.push_back(cur);
        }
        else
        {
            if(region.indices.size() >= minimum_inlier)
                regions.push_back( region );

            // for next iteration
            region.indices.clear();
        }
        pre = cur;
        cur += step;
        count --;
    }
    // for the last region, get scan point, do line fitting
    if(region.indices.size() >= minimum_inlier)
        regions.push_back( region );

}

void LineBasedPlaneSegmentation::lineRegions(std::vector<LineType>& regions)
{
    std::vector<int> rows = selected_rows_;
    std::vector<int> cols = selected_cols_;
    //
    if( use_horizontal_line_ )
    {
        for(size_t i = 0; i < rows.size(); i++)
        {
            lineRegionRow( rows[i], regions );
        }
    }

    if( use_verticle_line_ )
    {
        for(size_t i = 0; i < cols.size(); i++)
        {
            lineRegionCol( cols[i], regions );
        }
    }
}
//
void LineBasedPlaneSegmentation::lineSegment(std::vector<LineType> regions, std::vector<LineType>& lines)
{
    const float angular_threshold = cos(DEG2RAD(line_fitting_angular_threshold_));
    const int min_size = line_fitting_min_indices_;

    // segment
    for(size_t r = 0; r < regions.size(); r++)
    {
        LineType &region = regions[r];

        // compute normals
        std::vector<Eigen::Vector4f> normals;
        for(std::vector<int>::iterator itt = region.indices.begin(), end = region.indices.end(); itt != end; itt++)
        {
            Eigen::Vector4f n;
            pcl::Normal &nn = normals_->points[*itt];
            if(isnan(nn.normal[0]) || isnan(nn.normal[1]) || isnan(nn.normal[2]) || isnan(nn.normal[3]))
                computeNormal(*itt, n);
            else
                n << nn.normal[0], nn.normal[1], nn.normal[2], nn.normal[3];
            normals.push_back(n);
        }


        int start = 0;
        int end = 0;
        for(size_t i = 1; i < region.indices.size(); i++)
        {
            Eigen::Vector4f &last_normal = normals[end];
            Eigen::Vector4f &normal = normals[i];
            float angle;
            angle = last_normal[0]*normal[0]+last_normal[1]*normal[1]+last_normal[2]*normal[2];
            if(angle > angular_threshold)   // angle < 2.0 degree, so cos(angle)>cos(d2r(2.0))
            {
                end = i;
            }
            else
            {
                int len = end - start + 1;
                if(len>min_size)
                {
                    LineType l;
                    l.indices.resize( len );
                    memcpy( &l.indices[0], &(region.indices[start]), sizeof(int)*len );
                    lines.push_back(l);
                }
                //
                i++;
                start = i;
                end = i;
            }
        }
        // remain line
        int len = end - start + 1;
        if(len>min_size)
        {
            LineType l;
            l.indices.resize( len );
            memcpy( &l.indices[0], &(region.indices[start]), sizeof(int)*len );
            lines.push_back(l);
        }
    }

}

/****************************************************************************************/
// Normal extimation
//
void LineBasedPlaneSegmentation::selectedNormalNeighbors( int index, std::vector<int> &indices )
{
    //
    indices.clear();
    //
    int y = index / cloud_width_;
    int x = index - cloud_width_ * y;
    const int smoothing_size_2 = normal_smoothing_size_ / 2;
    int start_x = x - smoothing_size_2;
    int end_x = x + smoothing_size_2-1;
    int start_y = y - smoothing_size_2;
    int end_y = y + smoothing_size_2-1;
    // check range
    if(start_x < 0)
    {
        start_x = 0;
        end_x = start_x + smoothing_size_2 * 2-1;
    }
    else if(end_x >= cloud_width_)
    {
        end_x = cloud_width_ - 1;
        start_x = end_x - smoothing_size_2 * 2+1;
    }
    // check range
    if(start_y < 0)
    {
        start_y = 0;
        end_y = start_y + smoothing_size_2 * 2-1;
    }
    else if(end_y >= cloud_height_)
    {
        end_y = cloud_height_ - 1;
        start_y = end_y - smoothing_size_2 * 2+1;
    }
    for( int r = start_y; r <= end_y; r++ )
    {
        int rs = r * cloud_width_;
        for( int c = start_x; c <= end_x; c++ )
        {
            int idx = rs + c;
            if( indices_mask_.at<uchar>(idx) )
                indices.push_back( idx );
        }
    }
}

//
void LineBasedPlaneSegmentation::computeNormalCoefficient( std::vector<int> &indices, NormalType &normal)
{
    // Placeholder for the 3x3 covariance matrix at each surface patch
    EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
    // 16-bytes aligned placeholder for the XYZ centroid of a surface patch
    Eigen::Vector4f xyz_centroid;
    pcl::computeMeanAndCovarianceMatrix (*input_, indices, covariance_matrix, xyz_centroid);
    normal.centroid.x = xyz_centroid(0);
    normal.centroid.y = xyz_centroid(1);
    normal.centroid.z = xyz_centroid(2);

    // Get the plane normal and surface curvature
    pcl::solvePlaneParameters (covariance_matrix, xyz_centroid, normal.coefficients, normal.curvature);
    //
    flipNormalTowardsViewpoint(normal.centroid, 0, 0, 0, normal.coefficients[0], normal.coefficients[1], normal.coefficients[2]);
    normal.coefficients[3] = 0;
    normal.coefficients[3] = -1 * normal.coefficients.dot (xyz_centroid);
}

//
unsigned LineBasedPlaneSegmentation::countNormalNeighbors( int index )
{
    unsigned count = 0;

    int y = index / cloud_width_;
    int x = index - cloud_width_ * y;
    const int smoothing_size_2 = normal_smoothing_size_ / 2;
    int start_x = x - smoothing_size_2;
    int end_x = x + smoothing_size_2;
    int start_y = y - smoothing_size_2;
    int end_y = y + smoothing_size_2;
    // check range
    if(start_x < 0)
    {
        start_x = 0;
        end_x = start_x + smoothing_size_2 * 2;
    }
    else if(end_x >= cloud_width_)
    {
        end_x = cloud_width_ - 1;
        start_x = end_x - smoothing_size_2 * 2;
    }
    // check range
    if(start_y < 0)
    {
        start_y = 0;
        end_y = start_y + smoothing_size_2 * 2;
    }
    else if(end_y >= cloud_height_)
    {
        end_y = cloud_height_ - 1;
        start_y = end_y - smoothing_size_2 * 2;
    }
    for( int r = start_y; r <= end_y; r++ )
    {
        int rs = r * cloud_width_;
        for( int c = start_x; c <= end_x; c++ )
        {
            int idx = rs + c;
            if( indices_mask_.at<uchar>(idx) )
                count ++;
        }
    }

    return count;
}



//
void LineBasedPlaneSegmentation::flipNormalTowardsViewpoint (const PointT &point,
                            float vp_x, float vp_y, float vp_z,
                            float &nx, float &ny, float &nz)
{
    // See if we need to flip any plane normals
    vp_x -= point.x;
    vp_y -= point.y;
    vp_z -= point.z;

    // Dot product between the (viewpoint - point) and the plane normal
    float cos_theta = (vp_x * nx + vp_y * ny + vp_z * nz);

    // Flip the plane normal
    if (cos_theta < 0)
    {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }
}

//
void LineBasedPlaneSegmentation::computeNormal( int center_index, NormalType &normal)
{
    normal.center_index = center_index;

    selectedNormalNeighbors( normal.center_index, normal.indices);
    normal.valid = true;
    const PointT &pt = input_->points[normal.center_index];

//    cout << "Normal " << YELLOW << normal.line_index << " indices = " << CYAN << normal.indices.size() << RESET << endl;
    int min_indices = normal_min_inliers_percentage_ * normal_smoothing_size_ * normal_smoothing_size_;
    if( normal.indices.size() < min_indices )
    {
        normal.valid = false;
        return;
    }

    // compute plane parameters
    computeNormalCoefficient( normal.indices, normal );

    if( normal.curvature >  normalCurvatureThreshold(pt.z))    // curvature actually is depth dependent
        normal.valid = false;
}

//
void LineBasedPlaneSegmentation::computeNormal(int center_index, Eigen::Vector4f &coefficients)
{
    //
    std::vector<int> indices;
    //
    int y = center_index / cloud_width_;
    int x = center_index - cloud_width_ * y;
    if(x < 2 || y < 2 || x > (cloud_width_-3) || y > (cloud_height_-3))
    {
        coefficients << 0, 0, 0, std::numeric_limits<float>::max();
        return;
    }

    //
    const int smoothing_size_2 = normal_estimate_smoothing_size_ / 2;
    int start_x = x - smoothing_size_2;
    int end_x = x + smoothing_size_2-1;
    int start_y = y - smoothing_size_2;
    int end_y = y + smoothing_size_2-1;
    //

    // check range
    if(start_x < 0)
    {
        start_x = 0;
        end_x = x + x;
    }
    else if(end_x >= cloud_width_)
    {
        end_x = cloud_width_ - 1;
        start_x = x - (cloud_width_-1-x);
    }
    // check range
    if(start_y < 0)
    {
        start_y = 0;
        end_y = y + y;
    }
    else if(end_y >= cloud_height_)
    {
        end_y = cloud_height_ - 1;
        start_y = y - (cloud_height_-1-y);
    }

    for( int r = start_y; r <= end_y; r++ )
    {
        int rs = r * cloud_width_;
        for( int c = start_x; c <= end_x; c++ )
        {
            int idx = rs + c;
            if( indices_mask_.at<uchar>(idx) )
                indices.push_back( idx );
        }
    }

    //
    const int size_thresh = 0.75 * (end_x-start_x+1) * (end_y-start_y+1);
    if(indices.size() < size_thresh)
    {
        coefficients << 0, 0, 0, std::numeric_limits<float>::max();
        return;
    }

    //
    NormalType n;
    computeNormalCoefficient(indices, n);
    coefficients = n.coefficients;
}

//
void LineBasedPlaneSegmentation::computePointNormal( int center_index, NormalType &normal)
{
    normal.center_index = center_index;

    selectedNormalNeighbors( normal.center_index, normal.indices);
    normal.valid = true;

    if( normal.indices.size() < (normal_min_inliers_percentage_ * normal_smoothing_size_ * normal_smoothing_size_) )
    {
        normal.valid = false;
        return;
    }

    // compute plane parameters
    computeNormalCoefficient( normal.indices, normal );
}

/****************************************************************************************/
// Dureplicated candidate removal
//
void LineBasedPlaneSegmentation::normalTangentCoordinate( NormalType &normal, Eigen::MatrixXf &local)
{
    // Get the unit vector
    Eigen::Vector3f n = normal.n;

    // Get the axis of rotation with the minimum projected length of the point
    Eigen::Vector3f axis( 0, 0, 1);
    float mx = fabs(n.x()), my = fabs(n.y()), mz = fabs(n.z());
    if ((mx <= my) && (mx <= mz)) {
      axis = Eigen::Vector3f(1.0, 0.0, 0.0);
    } else if ((my <= mx) && (my <= mz)) {
      axis = Eigen::Vector3f(0.0, 1.0, 0.0);
    }

    // Choose the direction of the first basis vector b1 in the tangent plane by crossing n with
    // the chosen axis.
    Eigen::Vector3f B1 = vector3fCross( n, axis );

    // Normalize result to get a unit vector: b1 = B1 / |B1|.
    Eigen::Vector3f b1 = normalizeVector3f(B1);

    // Get the second basis vector b2, which is orthogonal to n and b1, by crossing them.
    // No need to normalize this, p and b1 are orthogonal unit vectors.
    Eigen::Vector3f b2 = vector3fCross( n, b1 );

    // Create the basis by stacking b1 and b2.
    local.resize( 3, 2);
    local << b1.x(), b2.x(), b1.y(), b2.y(), b1.z(), b2.z();
}

// detail see: GTSAM4, gtsam::geometry::Unit3::localCoordinates()
//
Eigen::Vector2f LineBasedPlaneSegmentation::normalError( NormalType &local, NormalType &other)
{
    const float x = local.n.dot(other.n);
    // Crucial quantity here is y = theta/sin(theta) with theta=acos(x)
    // Now, y = acos(x) / sin(acos(x)) = acos(x)/sqrt(1-x^2)
    // We treat the special case 1 and -1 below
    const float x2 = x * x;
    const float z = 1 - x2;
    float y;
    if (z < std::numeric_limits<float>::epsilon())
    {
      if (x > 0)  // first order expansion at x=1
        y = 1.0 - (x - 1.0) / 3.0;
      else  // cop out
        return Eigen::Vector2f(M_PI, 0.0);
    }
    else
    {
      // no special case
      y = acos(x) / sqrt(z);
    }
    return local.basis.transpose() * y * (other.n - x * local.n);
}

bool LineBasedPlaneSegmentation::isSimilarPlanes(NormalType &normal1, NormalType &normal2,
                                                 float direction_thresh, float distance_thresh)
{
    Eigen::Vector2f error_dir = normalError( normal1, normal2 );
    float error_d = fabs(normal1.d - normal2.d);
    if( (fabs(error_dir[0]) + fabs(error_dir[1])) < direction_thresh
            && error_d < distance_thresh )
    {
        return true;
    }else{
        return false;
    }
}

/****************************************************************************************/
// Plane extraction
//
bool LineBasedPlaneSegmentation::checkOverSegmentation(std::vector<LineType> &lines,
                                                       std::vector<NormalType> &line_normals,
                                                       std::vector<NormalType> &normals,
                                                       std::vector<PlaneType> &planes)
{
    if(planes.size() <= 1)
        return false;

//    ros::Time start_dura = ros::Time::now();

    // valid line indices
    std::vector< std::vector<int> > valid_line_indices; // for each line normal
    valid_line_indices.reserve(line_normals.size());
//    const unsigned int window_size = slide_window_size_;
    const unsigned int cutoff = 5;//
    for( size_t i = 0; i < line_normals.size(); i++)
    {
        LineType &line = lines[line_normals[i].line_index];
        if(line.valid)
        {
            std::vector<int> valid_indices = getValidIndices(line.indices, cutoff);
            if(valid_indices.size() < 3)
            {
                line.valid = false;
                line_normals[i].valid = false;
                valid_line_indices.push_back(std::vector<int>(0));
            }
            else{
                line_normals[i].valid = true;
                valid_line_indices.push_back(valid_indices);
            }
        }
        else
        {
            line.valid = false;
            line_normals[i].valid = false;
            valid_line_indices.push_back(std::vector<int>(0));
        }
    }

//    cout << BOLDMAGENTA << " LVI: " << BOLDYELLOW << getScopeTime(start_dura) << RESET;

    const int min_normal_indices = normal_smoothing_size_*normal_smoothing_size_* 4 / 5;
    for( size_t i = 0; i < planes.size(); i++ )
    {
        PlaneType &plane = planes[i];
        NormalType &normal = normals[plane.normal_index];
        if(!plane.over_segment || !normal.valid || !plane.valid)
            continue;

        plane.over_segment = false; // Set false first
        for( size_t j = 0; j < line_normals.size(); j++)
        {
            NormalType &line_normal = line_normals[j];
            if(!line_normal.valid || normal.line_index == line_normal.line_index)
                continue;

            //
            std::vector<int> &valid_indices = valid_line_indices[j];

            // If normal and line_normal are similar, discard. Check only the direction
//            float anglediff = acos(normal.coefficients.head<3>().dot(line_normal.coefficients.head<3>()));
            Eigen::Vector2f n_error = normalError(normal, line_normal);
            float anglediff = fabs(n_error[0])+fabs(n_error[2]);
            if(anglediff < 3.0*reduplicate_candidate_normal_thresh_)
                continue;

//            getScopeTime(start_dura);// RESET

            // check if line & plane have intersection point
            float min_p2p = 1.0;
            int min_index = -1;
            for(size_t k = 0; k < valid_indices.size(); k++)
            {
                float p2p = computePointWithDistanceToPlane(valid_indices[k], plane.coefficients);
                if( p2p < distance_threshold_ && p2p < min_p2p)
                {
                    // check if this point is on the plane
                    if(std::find(plane.indices.begin(), plane.indices.end(), valid_indices[k]) == plane.indices.end())
                        continue;

                    min_p2p = p2p;
                    min_index = valid_indices[k];
                }
            }

            if(min_index >= 0)
            {
                // Compute normal of this point
                NormalType point_normal;
                const PointT &pt = input_->points[min_index];
                computePointNormal(min_index, point_normal);

//                cout << endl;
//                cout << WHITE << "  - n" << RED << plane.normal_index << WHITE << "l" << MAGENTA << line_normal.line_index
//                     << CYAN << " " << anglediff << WHITE << "/"<< YELLOW << (2.0*reduplicate_candidate_normal_thresh_)
//                     << GREEN << " " << (point_normal.valid?"true":"false")
//                     << CYAN << " " << point_normal.indices.size() << WHITE << "/" << YELLOW << min_normal_indices
//                     << CYAN << " " << point_normal.curvature << WHITE "/" << YELLOW << normalCurvatureThreshold(pt.z);


                if( point_normal.valid
                        && point_normal.indices.size() > (min_normal_indices)
                        && (point_normal.curvature <  normalCurvatureThreshold(pt.z)) ) // curvature actually is depth dependent
                {
                    // Check
//                    cout << GREEN << " pass";
                    double adiff = acos(point_normal.coefficients.head<3>().dot(line_normal.coefficients.head<3>()));
                    if(adiff < reduplicate_candidate_normal_thresh_)
                    {
                        plane.over_segment = true;
//                        cout << BOLDRED << " OverSeg";
                    }
                }
            }

            if(plane.over_segment)
                break;
        }

        // update to normal
        normal.over_segment = plane.over_segment;
    }

//    cout << endl;
//    cout << BOLDMAGENTA << "  COSL: " << BOLDYELLOW << getScopeTime(start_dura) << RESET;
}

//
inline bool LineBasedPlaneSegmentation::checkPointWithDistanceToPlane( int index, const Eigen::Vector4f &model_coefficients, float threshold)
{
    const PointT &point = input_->points[index];
    float distance = fabs ( model_coefficients[0] * point.x + model_coefficients[1] * point.y + model_coefficients[2] * point.z + model_coefficients[3]);
    return (distance < threshold);
}

// Compare cos(a) value
inline bool LineBasedPlaneSegmentation::checkNormalAngularDistance(int index, Eigen::Vector4f &model_coefficients, float cos_threshold)
{
    pcl::Normal &normal = normals_->points[index];
    float angle = normal.normal[0]*model_coefficients[0]+normal.normal[1]*model_coefficients[1]+normal.normal[2]*model_coefficients[2];
    if(isnan(angle))
        return true;
    return (angle > cos_threshold);
}

// Compare cos(a) value
inline bool LineBasedPlaneSegmentation::checkNormalDistance(int idx1, int idx2, float cos_threshold)
{
    pcl::Normal &n1 = normals_->points[idx1];
    pcl::Normal &n2 = normals_->points[idx2];
    float angle = n1.normal[0]*n2.normal[0]+n1.normal[1]*n2.normal[1]+n1.normal[2]*n2.normal[2];
    if(isnan(angle))
        return true;
    return (angle > cos_threshold);
}

//
int LineBasedPlaneSegmentation::getPlaneWithMostIndices( const std::vector<PlaneType> &planes )
{
    int max_index = -1;
    int max_size = 0;
    for( size_t i = 0; i < planes.size(); i++)
    {
        const PlaneType &plane = planes[i];
        if( plane.valid && plane.indices.size() > max_size)
        {
            max_index = i;
            max_size = plane.indices.size();
        }
    }
    return max_index;
}

//
int LineBasedPlaneSegmentation::getPlaneWithGoodIndices( const std::vector<PlaneType> &planes )
{
    std::vector<int> good_index;
    for(size_t i = 0; i < planes.size(); i++)
    {
        const PlaneType &plane = planes[i];
        if(plane.valid && !plane.over_segment)
            good_index.push_back(i);
    }

    int max_index = -1;
    int max_size = 0;
    if(good_index.size() > 0)
    {
        for(size_t i = 0; i < good_index.size(); i++)
        {
            int idx = good_index[i];
            const PlaneType &plane = planes[idx];
            if( plane.valid && plane.indices.size() > max_size)
            {
                max_index = idx;
                max_size = plane.indices.size();
            }
        }
        return max_index;
    }
    else
    {
        for( size_t i = 0; i < planes.size(); i++)
        {
            const PlaneType &plane = planes[i];
            if( plane.valid && plane.indices.size() > max_size)
            {
                max_index = i;
                max_size = plane.indices.size();
            }
        }
        return max_index;
    }

    return max_index;
}

//
// Only for planeExtraction2
int LineBasedPlaneSegmentation::getOnePlaneIndex(std::vector<LineType> &lines,
                                                 std::vector<NormalType> &line_normals,
                                                 std::vector<NormalType>& normals,
                                                 std::vector<PlaneType>& planes,
                                                 bool solve_over_segment)
{
#ifdef DEBUG
    ros::Time start_dura = ros::Time::now();
#endif
    int max_index = -1;

    if(planes.size() <= 0)
        return -1;

    if(solve_over_segment)
    {
        // Check if there is over-segmentation
        checkOverSegmentation(lines, line_normals, normals, planes);
#ifdef DEBUG
        cout << endl << WHITE << " - Over-segment(red):";

        for( size_t i = 0; i < planes.size(); i++)
        {
            PlaneType &plane = planes[i];
            if(plane.over_segment)
                cout << RED << " " << plane.normal_index;
            else
                cout << GREEN << " " << plane.normal_index;
        }
#endif

    }

#ifdef DEBUG
    cout << BOLDWHITE<< " COS: " << BOLDYELLOW << getScopeTime(start_dura) << RESET;
#endif

    // Select one
    max_index = getPlaneWithGoodIndices(planes);

    // No valid plane, break extraction
    if( max_index < 0 )
        return -1;

#ifdef DEBUG
    cout << WHITE << " Select " << BLUE << planes[max_index].normal_index << RESET;
#endif

    // not check the curvature
    PlaneType &plane = planes[max_index];
    computePlaneCoefficient( plane.indices, plane );

    return max_index;
}

//
void LineBasedPlaneSegmentation::selectConnectedPlaneRegion (int start_index, PlaneCoefficients &coefficients,
                                                             const float threshold, std::vector<int> &indices)
{
    //
    indices.clear();

    const float square_threshold = neighbor_threshold_ * neighbor_threshold_;
    cv::Mat mask = indices_mask_.clone();
//    normal.mask = cv::Mat::zeros( cloud_height_, cloud_width_, CV_8UC1 );
    std::stack<cv::Point2i> search_list;
    // add center point
    if( !mask.at<uchar>( start_index ) || !checkPointWithDistanceToPlane( start_index, coefficients, threshold))
        return;
    indices.push_back( start_index );
    mask.at<uchar>( start_index ) = 0;
    cv::Point2i center_point;
    center_point.y = start_index / cloud_width_;
    center_point.x = start_index - center_point.y * cloud_width_;
    search_list.push( center_point );

    // loop while search list is not empty
    while( search_list.size() > 0)
    {
        cv::Point2i pc = search_list.top();
        const PointT &pcenter = input_->points[ pc.y * cloud_width_ + pc.x];
        search_list.pop();
        for( int i =0; i < 8; i ++)
        {
            // pick one neighbor
            cv::Point2i pn = pc + neighbor8_dir[i];
            if( pn.x < 0 || pn.x >= cloud_width_ || pn.y < 0 || pn.y >= cloud_height_ )
                continue;
            int idx = pn.y * cloud_width_ + pn.x;
            const PointT &pneighbor = input_->points[idx];
            if( mask.at<uchar>( idx )
                    && checkPointWithDistanceToPlane( idx, coefficients, threshold)
                    && squareDistancePoint2Point( pcenter,  pneighbor) < square_threshold )
            {
                // add to indices
                indices.push_back( idx );
                mask.at<uchar>( idx ) = 0;
                // add to search list
                search_list.push( pn );
            }
        } // end for
    } // end while

}


//
void LineBasedPlaneSegmentation::selectConnectedPlaneRegionAndBoundary (int start_index,
                                                                   PlaneCoefficients &coefficients,
                                                                   const float threshold,
                                                                   std::vector<int> &indices,
                                                                   std::vector<int> &boundary_indices,
                                                                   std::vector<int> &hull_indices)
{
    //
    indices.clear();

    const float square_threshold = neighbor_threshold_ * neighbor_threshold_;
    cv::Mat mask = indices_mask_.clone();
    cv::Mat plane_mask = cv::Mat::zeros( cloud_height_, cloud_width_, CV_8UC1 );
    std::stack<cv::Point2i> search_list;
    // add center point
    if( !mask.at<uchar>( start_index ) || !checkPointWithDistanceToPlane( start_index, coefficients, threshold))
        return;
    indices.push_back( start_index );
    mask.at<uchar>( start_index ) = 0;
    cv::Point2i center_point;
    center_point.y = start_index / cloud_width_;
    center_point.x = start_index - center_point.y * cloud_width_;
    search_list.push( center_point );

    // loop while search list is not empty
    while( search_list.size() > 0)
    {
        cv::Point2i pc = search_list.top();
        const PointT &pcenter = input_->points[ pc.y * cloud_width_ + pc.x];
        search_list.pop();
        for( int i =0; i < 8; i ++)
        {
            // pick one neighbor
            cv::Point2i pn = pc + neighbor8_dir[i];
            if( pn.x < 0 || pn.x >= cloud_width_ || pn.y < 0 || pn.y >= cloud_height_ )
                continue;
            int idx = pn.y * cloud_width_ + pn.x;
            const PointT &pneighbor = input_->points[idx];
            if( mask.at<uchar>( idx )
                    && checkPointWithDistanceToPlane( idx, coefficients, threshold)
                    && squareDistancePoint2Point( pcenter,  pneighbor) < square_threshold )
            {
                // add to inlier
                indices.push_back( idx );
                mask.at<uchar>( idx ) = 0;
                // mark inlier
                plane_mask.at<uchar>( idx ) = 255;
                // add to search list
                search_list.push( pn );
            }
        } // end for
    } // end while


    // Extract boundary
    cv::Mat boundary_mat;
    cv::threshold( plane_mask, boundary_mat, 100.0, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> >hulls;
    // find boundary
    cv::findContours( boundary_mat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point(0, 0) );
    for(size_t i = 0; i < contours.size(); i++)
    {
        std::vector<cv::Point> hull;
        cv::convexHull( cv::Mat(contours[i]), hull, true );
        hulls.push_back( hull );
    }
    // save boundary and hull inlier
    for( size_t i = 0; i < contours[0].size(); i++)
    {
        cv::Point &p = contours[0][i];
        boundary_indices.push_back( p.y * cloud_width_ + p.x );
    }
    for(size_t i = 0; i < hulls[0].size(); i++)
    {
        cv::Point &p = hulls[0][i];
        hull_indices.push_back( p.y * cloud_width_ + p.x );
    }

}

//
void LineBasedPlaneSegmentation::computePlaneCoefficient( std::vector<int> &indices, PlaneType &plane)
{
    // Placeholder for the 3x3 covariance matrix at each surface patch
    EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
    // 16-bytes aligned placeholder for the XYZ centroid of a surface patch
    Eigen::Vector4f xyz_centroid;
    pcl::computeMeanAndCovarianceMatrix (*input_, indices, covariance_matrix, xyz_centroid);
    //
    plane.centroid.x = xyz_centroid[0];
    plane.centroid.y = xyz_centroid[1];
    plane.centroid.z = xyz_centroid[2];
    plane.covariance = covariance_matrix;

    // Get the plane normal and surface curvature
    pcl::solvePlaneParameters (covariance_matrix, xyz_centroid, plane.coefficients, plane.curvature);
    // Extract the smallest eigenvalue and its eigenvector
//    EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
//    EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
//    pcl::eigen33 (covariance_matrix, eigen_vectors, eigen_values);
//    //
//    plane.coefficients.head<3>() = eigen_vectors.col(0);
//    // Compute the curvature surface change
//    float eig_sum = covariance_matrix.coeff(0) + covariance_matrix.coeff(4) + covariance_matrix.coeff(8);
//    if (eig_sum != 0)
//      plane.curvature = fabsf(eigen_values(0) / eig_sum);
//    else
//      plane.curvature = 0;
    //
    flipNormalTowardsViewpoint(plane.centroid, 0, 0, 0, plane.coefficients[0], plane.coefficients[1], plane.coefficients[2]);
    plane.coefficients[3] = 0;
    plane.coefficients[3] = -1 * plane.coefficients.dot (xyz_centroid);

//    // Set lamda
//    plane.eigen_values = eigen_values;
//    plane.eigen_vectors.setOnes();
//    if( plane.coefficients.head<3>().dot(eigen_vectors.col(0)) < 0 )
//    {
//        plane.eigen_vectors.col(0) = -1 * eigen_vectors.col(0); // z-axis
//        plane.eigen_vectors.col(1) = eigen_vectors.col(1);      // y-axis
//        plane.eigen_vectors.col(2) = plane.eigen_vectors.col(1).cross( plane.eigen_vectors.col(0) );    // x-axis
//    }
//    else
//    {
//        plane.eigen_vectors = eigen_vectors;
//        plane.eigen_vectors.col(2) = plane.eigen_vectors.col(1).cross( plane.eigen_vectors.col(0) );    // x-axis
//    }
//    // Local coordinate
//    plane.local_coordinate.setIdentity();
//    plane.local_coordinate.col(0).head<3>() = plane.eigen_vectors.col(2);
//    plane.local_coordinate.col(1).head<3>() = plane.eigen_vectors.col(1);
//    plane.local_coordinate.col(2).head<3>() = plane.eigen_vectors.col(0);
//    plane.local_coordinate.col(3).head<3>() = xyz_centroid.head<3>();
}

//
void LineBasedPlaneSegmentation::extractBoundary(std::vector<int> &indices,
                                           std::vector<int> &boundary_indices,
                                           std::vector<int> &hull_indices)
{
    // construct image
    cv::Mat mark = cv::Mat::zeros( cloud_height_, cloud_width_, CV_8UC1 );
    for(size_t i = 0; i < indices.size(); i++)
    {
        mark.at<uchar>(indices[i]) = 255;
    }

    // find boundary
    cv::Mat boundary_mat;
    cv::threshold( mark, boundary_mat, 100.0, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> >hulls;
    //
    cv::findContours( boundary_mat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point(0, 0) );
    //
    for(size_t i = 0; i < contours.size(); i++)
    {
        std::vector<cv::Point> hull;
        cv::convexHull( cv::Mat(contours[i]), hull, true );
        hulls.push_back( hull );
    }

    boundary_indices.clear();
    hull_indices.clear();
    // save boundary and hull indices
    for( size_t i = 0; i < contours[0].size(); i++)
    {
        cv::Point &p = contours[0][i];
        boundary_indices.push_back( p.y * cloud_width_ + p.x );
    }
    for(size_t i = 0; i < hulls[0].size(); i++)
    {
        cv::Point &p = hulls[0][i];
        hull_indices.push_back( p.y * cloud_width_ + p.x );
    }
}

//
void LineBasedPlaneSegmentation::deleteInliers(const std::vector<int> inliers)
{
    for(size_t i = 0; i < inliers.size(); i++)
    {
        indices_mask_.at<uchar>( inliers[i] ) = 0;
    }
}

void LineBasedPlaneSegmentation::extractAllPossiblePlanes(std::vector<NormalType> &normals,
                                                          std::vector<PlaneType> &planes)
{
    planes.clear();
    //
    const int normal_min_inliers = normal_min_inliers_percentage_ * normal_smoothing_size_ * normal_smoothing_size_;
    for(size_t i = 0; i < normals.size(); i++)
    {
        NormalType &normal = normals[i];
        // check if normal valid
        if( normal.valid )
        {
            if( countNormalNeighbors( normal.center_index ) < normal_min_inliers)
            {
                normal.valid = false;
                continue;
            }

            PlaneType plane;
            selectConnectedPlaneRegion( normal.center_index, normal.coefficients, distance_threshold_, plane.indices);

            if( plane.indices.size() < min_inliers_ )
            {
                normal.valid = false;
                continue;
            }
            else
            {
                plane.valid = true;
                plane.coefficients = normal.coefficients;
                plane.normal_index = i;   // which normam r.s.t this plane
                plane.over_segment = normal.over_segment;
                planes.push_back( plane );  // put into stack
            }
        }
    }
}

void LineBasedPlaneSegmentation::extractFinalPlane(PlaneType &plane, int start_index, bool delete_indices)
{
    if( optimize_coefficients_ && start_index >= 0)
    {
        // find planar region using optimized coefficients
        if( extract_boundary_ )
            selectConnectedPlaneRegionAndBoundary( start_index, plane.coefficients, distance_threshold_, plane.indices, plane.boundary_indices, plane.hull_indices );
        else
            selectConnectedPlaneRegion( start_index, plane.coefficients, distance_threshold_, plane.indices);
        computePlaneCoefficient( plane.indices, plane );
    }
    else
    {
        if( extract_boundary_ )
            extractBoundary( plane.indices, plane.boundary_indices, plane.hull_indices);
    }
    //
    if(delete_indices)
        deleteInliers(plane.indices);
}

/****************************************************************************************/
void loadBoolParam(cv::FileStorage &fs, const std::string &name, bool &var, bool default_value)
{
    std::string bstr = fs[name];
    if(bstr.empty()){
        var = default_value;
        cout << WHITE << "  - " << BLUE << name << WHITE << " default " << YELLOW << (default_value?"true":"false") << RESET << endl;
    }else{
        if(!bstr.compare("true")){
            var = true;
            cout << WHITE << "  - " << BLUE << name << WHITE << " load " << CYAN << "true" << RESET << endl;
        }else{
            var = false;
            cout << WHITE << "  - " << BLUE << name << WHITE << " load " << CYAN << "false" << RESET << endl;
        }
    }
}

void loadIntParam(cv::FileStorage &fs, const std::string &name, int &var, int default_value)
{
    cv::FileNode fn = fs[name];
    if(fn.empty()){
        var = default_value;
        cout << WHITE << "  - " << BLUE << name << WHITE << " default " << YELLOW << default_value << RESET << endl;
    }else{
        var = (int)fn;
        cout << WHITE << "  - " << BLUE << name << WHITE << " load " << CYAN << var << RESET << endl;
    }
}

void loadFloatParam(cv::FileStorage &fs, const std::string &name, float &var, float default_value)
{
    cv::FileNode fn = fs[name];
    if(fn.empty()){
        var = default_value;
        cout << WHITE << "  - " << BLUE << name << WHITE << " default " << YELLOW << default_value << RESET << endl;
    }else{
        var = (float)fn;
        cout << WHITE << "  - " << BLUE << name << WHITE << " load " << CYAN << var << RESET << endl;
    }
}

void loadDoubleParam(cv::FileStorage &fs, const std::string &name, double &var, double default_value)
{
    cv::FileNode fn = fs[name];
    if(fn.empty()){
        var = default_value;
        cout << WHITE << "  - " << BLUE << name << WHITE << " default " << YELLOW << default_value << RESET << endl;
    }else{
        var = (double)fn;
        cout << WHITE << "  - " << BLUE << name << WHITE << " load " << CYAN << var << RESET << endl;
    }
}

}

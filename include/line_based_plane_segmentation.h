#ifndef LINE_BASED_PLANE_SEGMANTATION_H
#define LINE_BASED_PLANE_SEGMANTATION_H

#include <pcl/console/print.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/PointIndices.h>
#include <pcl/common/centroid.h>
#include <pcl/features/feature.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "line_extraction.h"
#include <stack>

namespace line_based_plane_segment
{

void loadBoolParam(cv::FileStorage &fs, const std::string &name, bool &var, bool default_value);
void loadIntParam(cv::FileStorage &fs, const std::string &name, int &var, int default_value);
void loadFloatParam(cv::FileStorage &fs, const std::string &name, float &var, float default_value);
void loadDoubleParam(cv::FileStorage &fs, const std::string &name, double &var, double default_value);

//void loadBoolParam(cv::FileStorage &fs, const std::string &name, bool &var, bool default_value);
//template<class T> void loadParam(cv::FileStorage &fs, const std::string &name, T &var, T default_value);

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef typename PointCloud::Ptr PointCloudPtr;
typedef typename PointCloud::ConstPtr PointCloudConstPtr;
typedef boost::shared_ptr<const pcl::PointRepresentation< PointT > > PointRepresentationConstPtr;
typedef Eigen::Vector4f PlaneCoefficients;

struct CAMERA_INFO
{
    int width, height;
    float cx, cy, fx, fy, scale;
    CAMERA_INFO() : width(320), height(240), cx(159.75), cy(119.75), fx(262.5), fy(262.5), scale(1.0) {}
    CAMERA_INFO(int _width, int _height, float _cx, float _cy, float _fx, float _fy, float _scale)
        : width(_width), height(_height), cx(_cx), cy(_cy), fx(_fx), fy(_fy), scale(_scale) {}
    CAMERA_INFO( CAMERA_INFO &other, int skip = 1): width(other.width/skip), height(other.height/skip), cx(other.cx/skip), cy(other.cy/skip), fx(other.fx/skip), fy(other.fy/skip), scale(other.scale) {}
};

struct LineType
{
    LineType() : valid(true) {}

    bool valid;
    std::vector<int> indices;
};

struct NormalType
{
    NormalType() : valid(true), over_segment(true) {}

    bool valid;
    bool over_segment;
    int line_index;
    int center_index;
    std::vector<int> indices;
    //
    PointT centroid;
    Eigen::Vector4f coefficients;
    float curvature;
    //
    Eigen::Vector3f n;
    float d;
    Eigen::MatrixXf basis;
};

struct PlaneType
{
    PlaneType():valid(false), over_segment(true), id(-1), coefficients(0.0,0.0,0.0,1.0),cloud(new PointCloud){}

    //
    bool valid;
    bool over_segment;
    int normal_index;
    //
    int id;
    PlaneCoefficients coefficients;
    PointT centroid;
    Eigen::Matrix3f covariance;
    float curvature;
    //
    PointCloudPtr cloud;
    std::vector<int> indices;
    std::vector<int> boundary_indices;
    std::vector<int> hull_indices;
    //
    Eigen::Vector3f n;
    float d;
    Eigen::MatrixXf basis;

    //
//    cv::Mat distance_mask;
};

class LineBasedPlaneSegmentation
{
public:
    LineBasedPlaneSegmentation(const std::string &setting_file);
    LineBasedPlaneSegmentation(CAMERA_INFO *camera);
    //
    void initialize();  // Build zfactor and update selected rows/cols
    //
    void setCameraInfo(CAMERA_INFO &camera) { camera_info_ = new CAMERA_INFO(camera); initialized_ = false;}
    void updateSelectedRowsAndCols();   // Update selected rows/cols
    void setInputCloud(const PointCloudPtr &cloud) { input_ = cloud; }
    inline PointCloudConstPtr const getInputCloud () { return (input_); }

    void setMask(cv::Mat &mask) { indices_mask_ = mask; }
    inline cv::Mat const getMask(){ return indices_mask_; }

    void setNormalCloud(const pcl::PointCloud<pcl::Normal>::Ptr &normals) { normals_ = normals; }
    inline pcl::PointCloud<pcl::Normal>::Ptr const getNormalCloud() { return(normals_); }

    // Solve the over-segment problem

    void segment(std::vector<PlaneType>& planes);

    void segment(std::vector<LineType>& lines,
                 std::vector<NormalType>& normals,
                 std::vector<PlaneType>& planes);

    void getScanlineCloud(const PointCloudPtr &input,
                          vector<pcl::PointIndices> &indices,
                          vector< vector<ScanPoint> > &scans);

    const CAMERA_INFO &getCameraParameters() { return *camera_info_; }

    // Each individual step of segmentation
    bool initCompute();
    void lineFitting( std::vector<LineType>& lines );
    void normalEstimate( std::vector<LineType>& lines, std::vector<NormalType>& normals);
    void removeDuplicateCandidates( std::vector<NormalType>& normals, std::vector<NormalType>& output);
    void planeExtraction(std::vector<LineType> &lines, std::vector<NormalType> &line_normals,
                         std::vector<NormalType>& normals, std::vector<PlaneType>& final_planes,
                         bool solve_over_segment = true);
    void planeExtraction2(std::vector<LineType> &lines, std::vector<NormalType> &line_normals,
                          std::vector<NormalType>& normals, std::vector<PlaneType>& final_planes,
                          bool solve_over_segment = true);
    void refinePlanes(std::vector<PlaneType>& final_planes);
    void deinitCompute();

    // With normals
    void lineFittingWithNormals(std::vector<LineType>& lines);
    void lineRegionWithNormals(std::vector<LineType>& regions);
    void lineSegmentWithNormals(std::vector<LineType> regions, std::vector<LineType>& lines);

private:
    // Line extraction
    void lineRegionRow(int row, std::vector<LineType> &regions);
    void lineRegionCol(int col, std::vector<LineType> &regions);
    void getRowScanPoints(int row, const std::vector<int> &indices, std::vector<ScanPoint> &scan_points);
    void getColScanPoints(int col, const std::vector<int> &indices, std::vector<ScanPoint> &scan_points);
    void lineSegmentRow(int row, std::vector<LineType>& lines );
    void lineSegmentCol(int col, std::vector<LineType>& lines );
    // Normal extimation
    void selectedNormalNeighbors( int index, std::vector<int> &indices );
    unsigned countNormalNeighbors( int index );
    void computeNormalCoefficient( std::vector<int> &indices, NormalType &normal);
    void flipNormalTowardsViewpoint (const PointT &point, float vp_x, float vp_y, float vp_z, float &nx, float &ny, float &nz);
    void computeNormal(int center_inlier, NormalType &normal);
    void computeNormal(int center_index, Eigen::Vector4f &coefficients);
    void computePointNormal( int center_index, NormalType &normal);
    void computeNormalsFromLine( LineType& line, std::vector<NormalType> & normals);
    // Reduplicated normal removal
    void normalTangentCoordinate( NormalType &normal, Eigen::MatrixXf &local);
    Eigen::Vector2f normalError( NormalType &local, NormalType &other);
    bool isSimilarPlanes(NormalType &normal1, NormalType &normal2,
                         float direction_thresh, float distance_thresh);
    // Plane extraction
    bool checkOverSegmentation(std::vector<LineType> &lines,
                               std::vector<NormalType> &line_normals,
                               std::vector<NormalType> &normals,
                               std::vector<PlaneType> &planes);
    inline bool checkPointWithDistanceToPlane( int index, const Eigen::Vector4f &model_coefficients, float threshold);
    inline bool checkNormalAngularDistance(int index, Eigen::Vector4f &model_coefficients, float cos_threshold);
    inline bool checkNormalDistance(int idx1, int idx2, float cos_threshold);
    int getPlaneWithMostIndices( const std::vector<PlaneType> &planes );
    int getPlaneWithGoodIndices( const std::vector<PlaneType> &planes );
    int getOnePlaneIndex(std::vector<LineType> &lines,
                         std::vector<NormalType> &line_normals,
                         std::vector<NormalType>& normals,
                         std::vector<PlaneType>& planes,
                         bool solve_over_segment);
    void selectConnectedPlaneRegion ( int start_index, PlaneCoefficients &coefficients,
                                    const float threshold, std::vector<int> &indices);
    void selectConnectedPlaneRegionAndBoundary ( int start_index,
                                                 PlaneCoefficients &coefficients,
                                                 const float threshold,
                                                 std::vector<int> &indices,
                                                 std::vector<int> &boundary_indices,
                                                 std::vector<int> &hull_indices);
    void computePlaneCoefficient( std::vector<int> &indices, PlaneType &plane);
    void extractBoundary(std::vector<int> &indices,
                         std::vector<int> &boundary_indices,
                         std::vector<int> &hull_indices);
    void deleteInliers(const std::vector<int> indices);
    void extractAllPossiblePlanes(std::vector<NormalType> &normals,
                                  std::vector<PlaneType> &planes);
    void extractFinalPlane(PlaneType &plane, int start_index, bool delete_indices = true);

    /***********************************************************************************/
    // All the inline functions
    // Check if indices mask is ok
    inline bool validMask()
    {
        return (!indices_mask_.empty());
    }
    // Check if point is valid
    inline bool isValidPoint(const PointT &p)
    {
        return (prttcp_->isValid(p) && p.z > 0);
    }
    // Squared distance
    inline float squareDistancePoint2Point(const PointT &p1, const PointT &p2)
    {
        Eigen::Vector3f delta= p1.getVector3fMap() - p2.getVector3fMap();
        return ( delta.dot(delta) );
    }
    // Distance
    inline float distancePoint2Point(const PointT &p1, const PointT &p2)
    {
        Eigen::Vector3f delta= p1.getVector3fMap() - p2.getVector3fMap();
        return ( sqrt( delta.dot(delta) ) );
    }
    //
    inline float computePointWithDistanceToPlane( int index, const Eigen::Vector4f &model_coefficients)
    {
        const PointT &point = input_->points[index];
        return fabs( model_coefficients[0] * point.x + model_coefficients[1] * point.y + model_coefficients[2] * point.z + model_coefficients[3]);
    }
    // Cross
    inline Eigen::Vector3f vector3fCross( Eigen::Vector3f& p, Eigen::Vector3f & q)
    {
        return Eigen::Vector3f (p.y() * q.z() - p.z() * q.y(),
                                p.z() * q.x() - p.x() * q.z(),
                                p.x() * q.y() - p.y() * q.x());
    }
    // Normalize
    inline Eigen::Vector3f normalizeVector3f( Eigen::Vector3f &p)
    {
        return p/sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
    }
    // Get valid indices
    inline std::vector<int> getValidIndices(std::vector<int> &input)
    {
        std::vector<int> output;
        for(std::vector<int>::iterator it = input.begin(), end = input.end();
            it != end; it++)
        {
            if(indices_mask_.at<uchar>(*it))
            {
                output.push_back(*it);
            }
        }
        return output;
    }
    // Get valid indices
    inline std::vector<int> getValidIndices(std::vector<int> &input, unsigned int cutoff)
    {
        std::vector<int> output;
        if(input.size() <= cutoff*2)
            return output;
        for(std::vector<int>::iterator it = input.begin()+cutoff, end = input.end()-cutoff;
            it != end; it++)
        {
            if(indices_mask_.at<uchar>(*it))
            {
                output.push_back(*it);
            }
        }
        return output;
    }
    //
    inline float normalCurvatureThreshold(float depth)
    {
        return std::max((float)normal_maximum_curvature_, (float)(normal_maximum_curvature_+0.005*(depth-1.0)));
    }

public:
    bool use_normal_cloud_;
    //
    bool use_horizontal_line_;
    bool use_verticle_line_;
    int y_interval_;
    int x_interval_;

    /** \brief Line extraction */
    float line_point_min_distance_;
    int slide_window_size_;
    int line_min_inliers_;
    float line_fitting_threshold_;
    void setLineRegressionParams(int window_size, float fidelity, int min_inlier)
    {
        line_segmentor_->setParameters(window_size, fidelity, min_inlier);
        slide_window_size_ = window_size;
        line_fitting_threshold_ = fidelity;
        line_min_inliers_ = min_inlier;
    }

    /** \brief Normals detection */
    int normals_per_line_;
    int normal_smoothing_size_;
    float normal_min_inliers_percentage_;
    float normal_maximum_curvature_;

    /** \brief Remove reduplicated candidates */
    bool remove_reduplicate_candidate_;
    float reduplicate_candidate_normal_thresh_;
    float reduplicate_candidate_distance_thresh_;

    /** \brief Plane extraction */
    bool plane_extraction_use_normal_;
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

    //
    bool line_fitting_use_normal_;
    int line_fitting_normal_smoothing_size_;
    float line_fitting_angular_threshold_;
    int line_fitting_min_indices_;
    float angular_threshold_;

    //
    //
    int normal_estimate_method_;
    float normal_estimate_depth_change_factor_;
    float normal_estimate_smoothing_size_;

private:
    bool initialized_;
    bool compute_initialized_;
    bool use_normals_;
    //
    PointCloudConstPtr input_;
    cv::Mat indices_mask_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    //
    LineRegressionSegmentation* line_segmentor_;
    PointRepresentationConstPtr prttcp_;
    CAMERA_INFO* camera_info_;
    int cloud_width_;
    int cloud_height_;
    std::vector<int> selected_rows_;
    std::vector<int> selected_cols_;
    std::vector<float> z_factor_row_;
    std::vector<float> z_factor_col_;
    //
    //
    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne_;
};

//the following are UBUNTU/LINUX ONLY terminal color
#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */

}

#endif

#ifndef PLANE_VIEWER_H
#define PLANE_VIEWER_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <pangolin/pangolin.h>
#include <pcl/point_cloud.h>
#include <stdlib.h>
#include <thread>
#include <list>
#include <mutex>
#include "line_based_plane_segmentation.h"
#include "organized_multi_plane_segmentor.h"

using namespace std;
using namespace line_based_plane_segment;

class PlaneSegment;

// RGB Value
typedef union
{
    struct /*anonymous*/
    {
        unsigned char Blue;
        unsigned char Green;
        unsigned char Red;
        unsigned char Alpha;
    };
    float float_value;
    long long_value;
} RGBValue;

typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointCloud<PointType> PointCloudType;
typedef PointCloudType::Ptr PointCloudTypePtr;
typedef PointCloudType::ConstPtr PointCloudTypeConstPtr;
//
typedef std::vector<LineType> VectorLines;
typedef std::vector<NormalType> VectorNormals;
typedef std::vector<PlaneType> VectorPlanes;

class PlaneViewer
{
public:
    PlaneViewer()
        : use_omps_(false)
        , omps_segmentation_(NULL)
        , plane_segmentation_(NULL)
        , finished_(false)
        , viewer_width_(1280)
        , viewer_height_(720)
        , ui_width_(320)
        , viewpoint_x_(0.0f)
        , viewpoint_y_(-0.0f)
        , viewpoint_z_(-0.1f)
        , viewpoint_f_(460.0f)
        , cloud_point_size_(1.0f)
        , line_point_size_(4.0f)
        , normal_point_size_(2.0f)
        , plane_point_size_(1.0f)
        , boundary_point_size_(2.0f)
        , hull_point_size_(4.0f)
        , image_rgb_(cv::Mat())
        , cloud_(new PointCloudType)
    {}

    PlaneViewer(const std::string &file_setting)
        : PlaneViewer()
    {
        cv::FileStorage fs(file_setting, cv::FileStorage::READ);

        loadIntParam(fs, "Viewer.width", viewer_width_, viewer_width_);
        loadIntParam(fs, "Viewer.height", viewer_height_, viewer_height_);
        loadIntParam(fs, "Viewer.ui_width", ui_width_, ui_width_);
        //
        loadFloatParam(fs, "Viewer.viewpoint_x", viewpoint_x_, viewpoint_x_);
        loadFloatParam(fs, "Viewer.viewpoint_y", viewpoint_y_, viewpoint_y_);
        loadFloatParam(fs, "Viewer.viewpoint_z", viewpoint_z_, viewpoint_z_);
        loadFloatParam(fs, "Viewer.viewpoint_f", viewpoint_f_, viewpoint_f_);
        //
        loadFloatParam(fs, "Viewer.cloud_point_size", cloud_point_size_, cloud_point_size_);
        loadFloatParam(fs, "Viewer.line_point_size", line_point_size_, line_point_size_);
        loadFloatParam(fs, "Viewer.normal_point_size", normal_point_size_, normal_point_size_);
        loadFloatParam(fs, "Viewer.plane_point_size", plane_point_size_, plane_point_size_);
        loadFloatParam(fs, "Viewer.boundary_point_size", boundary_point_size_, boundary_point_size_);
        loadFloatParam(fs, "Viewer.hull_point_size", hull_point_size_, hull_point_size_);

    }


    void setPlaneSegmentor(PlaneSegment* segmentation){ plane_segmentation_ = segmentation;}
    void run();
    void spin();
    void finish(){ finished_ = true; run_thread_->detach();}


    PointCloudTypePtr getCloud();
    void setCloud(const PointCloudTypePtr &cloud);
    cv::Mat getImageRGB();
    void setImageRGB(const cv::Mat &rgb);
    cv::Mat getImageDepth();
    void setImageDepth(const cv::Mat &depth);

    void setSegmentResult(VectorLines &lines, VectorNormals &normals, VectorPlanes &planes);
    void setRuntimes(std::vector<std::string>& steps, std::vector<float> &runtimes);

    /// For omps
    void ompsRun();
    //
    void setPlaneSegmentor(OrganizedPlaneSegmentor* segmentation){ omps_segmentation_ = segmentation;}
    inline void setUseOmps(bool use){use_omps_ = use;}
    bool isUseOmps() const {return use_omps_;}

private:
    void showPlanes(pangolin::View &d_cam, pangolin::OpenGlRenderState &s_cam,
                    bool display_cloud = false, bool display_lines = true, bool display_normals = true,
                    bool display_planes = true, bool display_boundary = true, bool display_hull = true,
                    bool project_points = false, bool display_residual = false);
    void showRuntimes(pangolin::View &d_cam, pangolin::OpenGlRenderState &s_cam, bool display_runtime = true);
    void showImageRGB(pangolin::View &d_cam, cv::Mat &rgb, pangolin::GlTexture &texture);
    void showPointCloud(pangolin::View &d_cam, pangolin::OpenGlRenderState &s_cam, PointCloudTypePtr &cloud);
    //
    void drawAxis( float scale = 1.0, float thickness = 2.0);
    void draw3DText( float x, float y, float z, const std::string &text);
    void drawWindowText( float u, float v, const std::string &text);
    void drawPointCloud(const pcl::PointCloud<pcl::PointXYZRGBA> &cloud,
                        float point_size = 1.0f);
    void drawPointCloud(const pcl::PointCloud<pcl::PointXYZRGBA> &cloud,
                        const std::vector<int> &indices,
                        float point_size = 1.0f);
    template<typename PT> void drawPointCloud(const pcl::PointCloud<PT> &cloud,
                                              GLubyte r, GLubyte g, GLubyte b, GLubyte a = 255,
                                              float point_size = 1.0f);
    template<typename PT> void drawPointCloud(const pcl::PointCloud<PT> &cloud,
                                              const std::vector<int> &indices,
                                              GLubyte r, GLubyte g, GLubyte b, GLubyte a = 255,
                                              float point_size = 1.0f);
    template<typename PT> void drawCloudHull(const pcl::PointCloud<PT> &cloud,
                                             const std::vector<int> &indices,
                                             GLubyte r, GLubyte g, GLubyte b, GLubyte a = 255,
                                             float line_width = 1.0f);
    void projectPoints(const PointCloudType &input, const std::vector<int> &inlier,
                       const Eigen::Vector4f &model_coefficients,
                       PointCloudType &projected_points );
    std::string timeToStr();

private:
    bool use_omps_;
    OrganizedPlaneSegmentor *omps_segmentation_;
    PlaneSegment *plane_segmentation_;
    bool finished_;
    thread* run_thread_;
    //
    int viewer_width_;
    int viewer_height_;
    int ui_width_;
    float viewpoint_x_;
    float viewpoint_y_;
    float viewpoint_z_;
    float viewpoint_f_;
    //
    float cloud_point_size_;
    float line_point_size_;
    float normal_point_size_;
    float plane_point_size_;
    float boundary_point_size_;
    float hull_point_size_;

    //
    cv::Mat image_rgb_;
    std::mutex image_rgb_mutex_;
    std::list<cv::Mat> image_rgb_list_;
    //
    cv::Mat image_depth_;
    std::mutex image_depth_mutex_;
    std::list<cv::Mat> image_depth_list_;
    //
    PointCloudTypePtr cloud_;
    std::mutex cloud_mutex_;
    std::list<PointCloudTypePtr> cloud_list_;
    //
    std::mutex plane_mutex_;
    VectorLines lines_;
    VectorNormals normals_;
    VectorPlanes planes_;
    //
    std::mutex runtime_mutex_;
    std::vector<std::string> steps_;
    std::vector<float> runtimes_;
};



#endif // PLANE_VIEWER_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <pangolin/pangolin.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <stdlib.h>
#include <thread>

#include "plane_viewer.h"
#include "plane_segment.h"

using namespace std;

void PlaneViewer::run()
{
    // Initialize viewer
    pangolin::CreateWindowAndBind("Plane Segment",viewer_width_,viewer_height_);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(viewer_width_-ui_width_*2,viewer_height_,viewpoint_f_,viewpoint_f_,(viewer_width_-ui_width_*2)/2.0f,viewer_height_/2.0f,0.1,1000),
                pangolin::ModelViewLookAt(viewpoint_x_,viewpoint_y_,0, 0,0,-viewpoint_z_, 0.0,-1.0, 0.0)
                );

    pangolin::OpenGlRenderState s_cam2(
                pangolin::ProjectionMatrix(ui_width_,ui_width_*0.75f,262.5,262.5,ui_width_/2.0f,ui_width_*0.375f,0.1,1000),
//                pangolin::ModelViewLookAt(0,0,-2.0, 0,0,0, 0.0,-1.0,0.0)
                pangolin::ModelViewLookAt(0,-0.1,-0.6, 0,-0.1,0, 0.0,-1.0,0.0)
                );

    pangolin::OpenGlRenderState s_cam3(
                pangolin::ProjectionMatrix(ui_width_,ui_width_*0.75f,262.5,262.5,ui_width_/2.0f,ui_width_*0.375f,0.1,1000),
//                pangolin::ModelViewLookAt(0,0,-2.0, 0,0,0, 0.0,-1.0,0.0)
                pangolin::ModelViewLookAt(0,-0.1,-0.6, 0,-0.1,0, 0.0,-1.0,0.0)
                );


    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::Display("planes")
            .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(viewer_width_-ui_width_*2))
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View& d_cam2 = pangolin::Display("cloud")
            .SetBounds(0.0, pangolin::Attach::Pix(ui_width_*0.75), pangolin::Attach::Pix(viewer_width_-ui_width_), 1.0)
            .SetHandler(new pangolin::Handler3D(s_cam2));

    pangolin::View& d_cam3 = pangolin::Display("image")
            .SetBounds(pangolin::Attach::Pix(ui_width_*0.75), pangolin::Attach::Pix(ui_width_*1.5), pangolin::Attach::Pix(viewer_width_-ui_width_), 1.0)
            .SetHandler(new pangolin::Handler3D(s_cam3));

    pangolin::Display("multi")
        .SetBounds(0.0, 1.0, 0.0, 1.0)
        .SetLayout(pangolin::LayoutOverlay)
        .AddDisplay(d_cam)
        .AddDisplay(d_cam2)
        .AddDisplay(d_cam3);

    pangolin::CreatePanel("menu3").SetBounds( 0, 1.0, pangolin::Attach::Pix(viewer_width_ - ui_width_*2), pangolin::Attach::Pix(viewer_width_ - ui_width_));
    pangolin::Var<bool> ps_use_horizontal_line("menu3.Use Horizontal Line", plane_segmentation_->use_horizontal_line_, true);
    pangolin::Var<bool> ps_use_verticle_line("menu3.Use Verticle Line", plane_segmentation_->use_verticle_line_, true);
    pangolin::Var<int> ps_y_interval("menu3.Y Interval", plane_segmentation_->y_interval_, 5, 300);
    pangolin::Var<int> ps_x_interval("menu3.X Interval", plane_segmentation_->x_interval_, 5, 300);
    /** \brief Line extraction */
    pangolin::Var<float> ps_line_point_min_distance("menu3.Line Point Min Distance", plane_segmentation_->line_point_min_distance_, 0.01, 0.2);
    pangolin::Var<float> ps_line_fitting_angular_threshold("menu3.Line Fitting Angular Thresh", plane_segmentation_->line_fitting_angular_threshold_, 1.0, 10.0);
    pangolin::Var<int> ps_line_fitting_min_inliers("menu3.Line Min Size", plane_segmentation_->line_fitting_min_indices_, 7, 37);

    /** \brief Normals per line */
    pangolin::Var<int> ps_normals_per_line("menu3.Normals Per Line", plane_segmentation_->normals_per_line_, 1, 5);
    pangolin::Var<int> ps_normal_smoothing_size("menu3.Normal Smooth Size", plane_segmentation_->normal_smoothing_size_, 5, 40);
    pangolin::Var<float> ps_normal_min_inliers_percentage("menu3.Normal Min Indices Alpha", plane_segmentation_->normal_min_inliers_percentage_, 0.5, 0.99);
    pangolin::Var<float> ps_normal_maximum_curvature("menu3.Normal Max Curvature", plane_segmentation_->normal_maximum_curvature_, 0.0001, 0.005);
    /** \brief Remove duplicate candidate if True */
    pangolin::Var<bool> ps_remove_reduplicate_candidate("menu3.Reduplicated Candidate Removal", plane_segmentation_->remove_reduplicate_candidate_, true);
    pangolin::Var<float> ps_reduplicate_candidate_normal_thresh("menu3.Reduplicated Direction Thresh", plane_segmentation_->reduplicate_candidate_normal_thresh_, 0.01, 0.2);
    pangolin::Var<float> ps_reduplicate_candidate_distance_thresh("menu3.Reduplicated Distance Thresh", plane_segmentation_->reduplicate_candidate_distance_thresh_, 0.01, 0.1);
    /** \brief Plane extraction */
    pangolin::Var<int> ps_min_inliers("menu3.Plane Min Size", plane_segmentation_->min_inliers_, 100, 10000);
    pangolin::Var<float> ps_max_curvature("menu3.Plane Max Curvature", plane_segmentation_->max_curvature_, 0.001, 0.02);
    pangolin::Var<float> ps_distance_threshold("menu3.Plane Distance Thresh", plane_segmentation_->distance_threshold_, 0.005, 0.05);
    pangolin::Var<float> ps_neighbor_threshold("menu3.Plane Neighbor Thresh", plane_segmentation_->neighbor_threshold_, 0.005, 0.5);
    pangolin::Var<float> ps_angular_threshold("menu3.Plane Angular Thresh", plane_segmentation_->angular_threshold_, 1.0, 20);
    /** \brief Refine Plane segmentation result. */
    pangolin::Var<bool> ps_solve_over_segment("menu3.Solve Over Segment", plane_segmentation_->solve_over_segment_, true);
    pangolin::Var<bool> ps_refine_plane("menu3.Refine Plane", plane_segmentation_->refine_plane_, true);
    pangolin::Var<bool> ps_optimize_coefficients("menu3.Optimize Coefficients", plane_segmentation_->optimize_coefficients_, true);
    pangolin::Var<bool> ps_project_points("menu3.Project Points(GUI)", plane_segmentation_->project_points_, true);
    pangolin::Var<bool> ps_extract_boundary("menu3.Extract Boundary", plane_segmentation_->extract_boundary_, true);
    //
    pangolin::Var<bool> psUpdate("menu3.Update", false, false);

    pangolin::CreatePanel("menu1").SetBounds( pangolin::Attach::Pix(ui_width_*1.5), 1.0, pangolin::Attach::Pix(viewer_width_-ui_width_), pangolin::Attach::Pix(viewer_width_-ui_width_/2.0));
//    pangolin::Var<bool> menuDisplayImageRGB("menu1.Display RGB",true,true);
    pangolin::Var<bool> menuDisplayCloud("menu1.Display Cloud",false,true);
    pangolin::Var<bool> menuDisplayLines("menu1.Display Lines",true,true);
    pangolin::Var<bool> menuDisplayNormals("menu1.Display Normals",true,true);
    pangolin::Var<bool> menuDisplayPlanes("menu1.Display Planes",true,true);
    pangolin::Var<bool> menuDisplayBoundary("menu1.Display Boundary",true,true);
    pangolin::Var<bool> menuDisplayHull("menu1.Display Hull",false,true);
    pangolin::Var<bool> menuDisplayRuntimes("menu1.Display Runtimes",true,true);
    pangolin::Var<bool> menuDisplayResidual("menu1.Display Residual",false,true);


    pangolin::CreatePanel("menu2").SetBounds( pangolin::Attach::Pix(ui_width_*1.5), 1.0, pangolin::Attach::Pix(viewer_width_-ui_width_/2.0), 1.0);
    pangolin::Var<bool> exitViewer("menu2.Exit",false,false);
    pangolin::Var<bool> saveWindow("menu2.Save Window",false,false);
    pangolin::Var<bool> recordWindow("menu2.Record Window",false,false);
    pangolin::Var<bool> menuResetView("menu2.Reset View",false,false);
    pangolin::Var<bool> menuStopResume("menu2.Stop & Resume", false, false);
    pangolin::Var<bool> menuSaveData("menu2.Save Data", false, false);
//    pangolin::Var<bool> menuReset("menu2.Reset",false,false);
    pangolin::Var<int> menuSkipPixel("menu2.Skip Pixels", plane_segmentation_->skip_pixel_, 1, 16);


    pangolin::GlTexture imageTexture(640,480,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
//    pangolin::GlTexture imageDepthTexture(640,480,GL_INTENSITY16,false,0,GL_RGB,GL_UNSIGNED_BYTE);

    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        plane_segmentation_->setSkipPixel(menuSkipPixel);

        if( pangolin::Pushed(psUpdate) )
        {
            //
            plane_segmentation_->use_horizontal_line_ = ps_use_horizontal_line;
            plane_segmentation_->use_verticle_line_ = ps_use_verticle_line;
            plane_segmentation_->y_interval_ = ps_y_interval;
            plane_segmentation_->x_interval_ = ps_x_interval;
            /** \brief Line extraction */
            plane_segmentation_->line_point_min_distance_ = ps_line_point_min_distance;
            plane_segmentation_->line_fitting_angular_threshold_ = ps_line_fitting_angular_threshold;
            plane_segmentation_->line_fitting_min_indices_ = ps_line_fitting_min_inliers;
            /** \brief Normals per line */
            plane_segmentation_->normals_per_line_ = ps_normals_per_line;
            plane_segmentation_->normal_smoothing_size_ = ps_normal_smoothing_size;
            plane_segmentation_->normal_min_inliers_percentage_ = ps_normal_min_inliers_percentage;
            plane_segmentation_->normal_maximum_curvature_ = ps_normal_maximum_curvature;
            /** \brief Remove duplicate candidate if True */
            plane_segmentation_->remove_reduplicate_candidate_ = ps_remove_reduplicate_candidate;
            plane_segmentation_->reduplicate_candidate_normal_thresh_ = ps_reduplicate_candidate_normal_thresh;
            plane_segmentation_->reduplicate_candidate_distance_thresh_ = ps_reduplicate_candidate_distance_thresh;
            /** \brief Plane extraction */
            plane_segmentation_->min_inliers_ = ps_min_inliers;
            plane_segmentation_->max_curvature_ = ps_max_curvature;
            plane_segmentation_->distance_threshold_ = ps_distance_threshold;
            plane_segmentation_->neighbor_threshold_ = ps_neighbor_threshold;
            plane_segmentation_->angular_threshold_ = ps_angular_threshold;
            /** \brief Refine Plane segmentation result. Note: Not Valid. */
            plane_segmentation_->solve_over_segment_ = ps_solve_over_segment;
            plane_segmentation_->refine_plane_ = ps_refine_plane;
            plane_segmentation_->optimize_coefficients_ = ps_optimize_coefficients;
            plane_segmentation_->project_points_ = ps_project_points;
            plane_segmentation_->extract_boundary_ = ps_extract_boundary;
        }
        if( pangolin::Pushed(menuSaveData) )
        {
            cv::Mat rgb = getImageRGB();
            cv::Mat depth = getImageDepth();
            PointCloudTypePtr cloud = getCloud();
            //
//            cout << "Image rgb: " << rgb.type() << " " << rgb.cols << " " << rgb.rows << " " << rgb.channels() << endl;
//            cout << "Image depth: " << depth.type() << " " << depth.cols << " " << depth.rows << " " << depth.channels() << endl;
            //
            std::string posfix = "_"+timeToStr();
            std::string filergb = "rgb"+posfix+".pgm";
            std::string filedepth = "depth"+posfix+".exr";
            std::string filepcd = "cloud"+posfix+".pcd";
            cout << "Save image rgb & depth: '" << filergb << "', '" << filedepth << "' and '" << filepcd << "'." << endl;
            cv::imwrite(filergb, rgb);
            cv::imwrite(filedepth, depth);
            //
            pcl::io::savePCDFileASCII(filepcd, *cloud);
        }
        if( pangolin::Pushed(exitViewer) )
        {
            finished_ = true;
            exit(0);
            break;
        }
        if( pangolin::Pushed(menuStopResume) )
        {
            if(!plane_segmentation_->isStopped())
            {
                plane_segmentation_->setStop();
            }
            else
            {
                plane_segmentation_->setResume();
            }
        }

        if( pangolin::Pushed(saveWindow) )
            pangolin::SaveWindowOnRender("window_"+timeToStr());

        if( pangolin::Pushed(recordWindow) )
            pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=30,bps=8388608,unique_filename]//cap_"+timeToStr()+".avi");

        if( pangolin::Pushed(menuResetView) )
        {
            d_cam.Activate(s_cam);
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(viewpoint_x_,viewpoint_y_,0, 0,0,-viewpoint_z_, 0.0,-1.0, 0.0));
            d_cam2.Activate(s_cam2);
            s_cam2.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-0.1,-0.6, 0,-0.1,0, 0.0,-1.0,0.0));
        }

        // RGB image render
        cv::Mat rgb = getImageRGB();
        showImageRGB(d_cam3, rgb, imageTexture);

        // Cloud render
        PointCloudTypePtr cloud = getCloud();
        showPointCloud(d_cam2, s_cam2, cloud);

        // Planes render
        showPlanes(d_cam, s_cam, menuDisplayCloud, menuDisplayLines, menuDisplayNormals,
                   menuDisplayPlanes, menuDisplayBoundary, menuDisplayHull, ps_project_points, menuDisplayResidual);

        // Runtimes
        showRuntimes(d_cam, s_cam, menuDisplayRuntimes);

        pangolin::FinishFrame();

        //
        usleep(20000);

        if(finished_)
            break;
    }
}

void PlaneViewer::ompsRun()
{
    // Initialize viewer
    pangolin::CreateWindowAndBind("OPMS Segment",viewer_width_,viewer_height_);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(viewer_width_-ui_width_*2,viewer_height_,viewpoint_f_,viewpoint_f_,(viewer_width_-ui_width_*2)/2.0f,viewer_height_/2.0f,0.1,1000),
                pangolin::ModelViewLookAt(viewpoint_x_,viewpoint_y_,0, 0,0,-viewpoint_z_, 0.0,-1.0, 0.0)
                );

    pangolin::OpenGlRenderState s_cam2(
                pangolin::ProjectionMatrix(ui_width_,ui_width_*0.75f,262.5,262.5,ui_width_/2.0f,ui_width_*0.375f,0.1,1000),
//                pangolin::ModelViewLookAt(0,0,-2.0, 0,0,0, 0.0,-1.0,0.0)
                pangolin::ModelViewLookAt(0,-0.1,-0.6, 0,-0.1,0, 0.0,-1.0,0.0)
                );

    pangolin::OpenGlRenderState s_cam3(
                pangolin::ProjectionMatrix(ui_width_,ui_width_*0.75f,262.5,262.5,ui_width_/2.0f,ui_width_*0.375f,0.1,1000),
//                pangolin::ModelViewLookAt(0,0,-2.0, 0,0,0, 0.0,-1.0,0.0)
                pangolin::ModelViewLookAt(0,-0.1,-0.6, 0,-0.1,0, 0.0,-1.0,0.0)
                );


    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::Display("planes")
            .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(viewer_width_-ui_width_*2))
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View& d_cam2 = pangolin::Display("cloud")
            .SetBounds(0.0, pangolin::Attach::Pix(ui_width_*0.75), pangolin::Attach::Pix(viewer_width_-ui_width_), 1.0)
            .SetHandler(new pangolin::Handler3D(s_cam2));

    pangolin::View& d_cam3 = pangolin::Display("image")
            .SetBounds(pangolin::Attach::Pix(ui_width_*0.75), pangolin::Attach::Pix(ui_width_*1.5), pangolin::Attach::Pix(viewer_width_-ui_width_), 1.0)
            .SetHandler(new pangolin::Handler3D(s_cam3));

    pangolin::Display("multi")
        .SetBounds(0.0, 1.0, 0.0, 1.0)
        .SetLayout(pangolin::LayoutOverlay)
        .AddDisplay(d_cam)
        .AddDisplay(d_cam2)
        .AddDisplay(d_cam3);

    pangolin::CreatePanel("menu3").SetBounds( 0, 1.0, pangolin::Attach::Pix(viewer_width_ - ui_width_*2), pangolin::Attach::Pix(viewer_width_ - ui_width_));
    pangolin::Var<int> omps_ne_method("menu3.Ne Method", omps_segmentation_->ne_method_, 0, 3);  // 0: COVARIANCE_MATRIX, 1: AVERAGE_3D_GRADIENT, 2: AVERAGE_DEPTH_CHANGE, 3: SIMPLE_3D_GRADIENT
    pangolin::Var<float> omps_ne_max_depth_change_factor("menu3.Ne Depth Change Factor", omps_segmentation_->ne_max_depth_change_factor_, 0.001, 0.5);
    pangolin::Var<int> omps_ne_normal_smoothing_size("menu3.Ne Smoothing Size", omps_segmentation_->ne_normal_smoothing_size_, 5, 50);
    pangolin::Var<int> omps_min_inliers("menu3.Min Inliers", omps_segmentation_->min_inliers_, 400, 10000);
    pangolin::Var<float> omps_angular_threshold("menu3.Angular Threshold", omps_segmentation_->angular_threshold_, 1.0, 10.0); // indegree
    pangolin::Var<float> omps_distance_threshold("menu3.Distance Threshold", omps_segmentation_->distance_threshold_, 0.005, 0.5);
    pangolin::Var<bool> omps_project_bounding_points("menu3.Project Boundary", omps_segmentation_->project_bounding_points_, false, true);
    pangolin::Var<bool> omps_project_points("menu3.Project Points", false, true);
    //
    pangolin::Var<bool> ompsUpdate("menu3.Update", false, false);

    pangolin::CreatePanel("menu1").SetBounds( pangolin::Attach::Pix(ui_width_*1.5), 1.0, pangolin::Attach::Pix(viewer_width_-ui_width_), pangolin::Attach::Pix(viewer_width_-ui_width_/2.0));
//    pangolin::Var<bool> menuDisplayImageRGB("menu1.Display RGB",true,true);
    pangolin::Var<bool> menuDisplayCloud("menu1.Display Cloud",false,true);
    pangolin::Var<bool> menuDisplayPlanes("menu1.Display Planes",true,true);
    pangolin::Var<bool> menuDisplayBoundary("menu1.Display Boundary",true,true);
    pangolin::Var<bool> menuDisplayHull("menu1.Display Hull",false,true);
    pangolin::Var<bool> menuDisplayRuntimes("menu1.Display Runtimes",true,true);
    pangolin::Var<bool> menuDisplayResidual("menu1.Display Residual",false,true);


    pangolin::CreatePanel("menu2").SetBounds( pangolin::Attach::Pix(ui_width_*1.5), 1.0, pangolin::Attach::Pix(viewer_width_-ui_width_/2.0), 1.0);
    pangolin::Var<bool> exitViewer("menu2.Exit",false,false);
    pangolin::Var<bool> saveWindow("menu2.Save Window",false,false);
    pangolin::Var<bool> recordWindow("menu2.Record Window",false,false);
    pangolin::Var<bool> menuResetView("menu2.Reset View",false,false);
    pangolin::Var<bool> menuStop("menu2.Stop", false, false);
    pangolin::Var<bool> menuResume("menu2.Resume", false, false);
    pangolin::Var<bool> menuSaveData("menu2.Save Data", false, false);
//    pangolin::Var<bool> menuReset("menu2.Reset",false,false);
    pangolin::Var<int> menuSkipPixel("menu2.Skip Pixels", plane_segmentation_->skip_pixel_, 1, 16);


    pangolin::GlTexture imageTexture(640,480,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
//    pangolin::GlTexture imageDepthTexture(640,480,GL_INTENSITY16,false,0,GL_RGB,GL_UNSIGNED_BYTE);

    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        plane_segmentation_->setSkipPixel(menuSkipPixel);

        if( pangolin::Pushed(ompsUpdate) )
        {
            omps_segmentation_->ne_method_ = omps_ne_method;
            omps_segmentation_->ne_max_depth_change_factor_ = omps_ne_max_depth_change_factor;
            omps_segmentation_->ne_normal_smoothing_size_ = (int)omps_ne_normal_smoothing_size;
            omps_segmentation_->angular_threshold_ = omps_angular_threshold;
            omps_segmentation_->distance_threshold_ = omps_distance_threshold;
            omps_segmentation_->min_inliers_ = omps_min_inliers;
            omps_segmentation_->project_bounding_points_ = omps_project_bounding_points;
            //
            omps_segmentation_->is_update_omps_parameters_ = true;  // Parameters will be updated before next segmentation
        }
        if( pangolin::Pushed(menuSaveData) )
        {
            cv::Mat rgb = getImageRGB();
            cv::Mat depth = getImageDepth();
            PointCloudTypePtr cloud = getCloud();
            //
//            cout << "Image rgb: " << rgb.type() << " " << rgb.cols << " " << rgb.rows << " " << rgb.channels() << endl;
//            cout << "Image depth: " << depth.type() << " " << depth.cols << " " << depth.rows << " " << depth.channels() << endl;
            //
            std::string posfix = "_"+timeToStr();
            std::string filergb = "rgb"+posfix+".pgm";
            std::string filedepth = "depth"+posfix+".exr";
            std::string filepcd = "cloud"+posfix+".pcd";
            cout << "Save image rgb & depth: '" << filergb << "', '" << filedepth << "' and '" << filepcd << "'." << endl;
            cv::imwrite(filergb, rgb);
            cv::imwrite(filedepth, depth);
            //
            pcl::io::savePCDFileASCII(filepcd, *cloud);
        }
        if( pangolin::Pushed(exitViewer) )
        {
            finished_ = true;
            exit(0);
            break;
        }
        if( pangolin::Pushed(menuStop) )
        {
            if(!plane_segmentation_->isStopped())
            {
                plane_segmentation_->setStop();
                menuStop = pangolin::Var<bool>("menu2.Resume", false, false);
            }
            else
            {
                plane_segmentation_->setResume();
                menuStop = pangolin::Var<bool>("menu2.Stop", false, false);
            }
        }
        if( pangolin::Pushed(menuResume) )
        {
            if(!plane_segmentation_->isStopped())
            {
                plane_segmentation_->setStop();
                menuResume = pangolin::Var<bool>("menu2.Resume", false, false);
            }
            else
            {
                plane_segmentation_->setResume();
                menuResume = pangolin::Var<bool>("menu2.Stop", false, false);
            }
        }
        if( pangolin::Pushed(saveWindow) )
            pangolin::SaveWindowOnRender("window_"+timeToStr());

        if( pangolin::Pushed(recordWindow) )
            pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=30,bps=8388608,unique_filename]//cap_"+timeToStr()+".avi");

        if( pangolin::Pushed(menuResetView) )
        {
            d_cam.Activate(s_cam);
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(viewpoint_x_,viewpoint_y_,0, 0,0,-viewpoint_z_, 0.0,-1.0, 0.0));
            d_cam2.Activate(s_cam2);
            s_cam2.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-0.8,-3.6, 0,-0.8,0, 0.0,-1.0,0.0));
        }

        // RGB image render
        cv::Mat rgb = getImageRGB();
        showImageRGB(d_cam3, rgb, imageTexture);

        // Cloud render
        PointCloudTypePtr cloud = getCloud();
        showPointCloud(d_cam2, s_cam2, cloud);

        // Planes render
        showPlanes(d_cam, s_cam, menuDisplayCloud, false, false,
                   menuDisplayPlanes, menuDisplayBoundary, menuDisplayHull, omps_project_points, menuDisplayResidual);

        // Runtimes
        showRuntimes(d_cam, s_cam, menuDisplayRuntimes);

        pangolin::FinishFrame();

        //
        usleep(20000);

        if(finished_)
            break;
    }
}

void PlaneViewer::spin()
{
    if(!use_omps_)
        run_thread_ = new thread(&PlaneViewer::run, this);
    else
        run_thread_ = new thread(&PlaneViewer::ompsRun, this);
}

void PlaneViewer::setSegmentResult(VectorLines &lines, VectorNormals &normals, VectorPlanes &planes)
{
    unique_lock<mutex> lock(plane_mutex_);
    lines_ = lines;
    normals_ = normals;
    planes_ = planes;
}

void PlaneViewer::setRuntimes(std::vector<string> &steps, std::vector<float> &runtimes)
{
    unique_lock<mutex> lock(runtime_mutex_);
    steps_ = steps;
    runtimes_ = runtimes;
}

PointCloudTypePtr PlaneViewer::getCloud()
{
    unique_lock<mutex> lock(cloud_mutex_);
    if(cloud_list_.size() > 0)
    {
        cloud_ = cloud_list_.front();
        cloud_list_.pop_front();
    }
    return cloud_;
}
void PlaneViewer::setCloud(const PointCloudTypePtr &cloud)
{
    unique_lock<mutex> lock(cloud_mutex_);
    cloud_list_.push_back(cloud);
}
cv::Mat PlaneViewer::getImageRGB()
{
    unique_lock<mutex> lock(image_rgb_mutex_);
    if(image_rgb_list_.size() > 0)
    {
        image_rgb_ = image_rgb_list_.front();
        image_rgb_list_.pop_front();
    }
    return image_rgb_;
}
void PlaneViewer::setImageRGB(const cv::Mat &rgb)
{
    unique_lock<mutex> lock(image_rgb_mutex_);
    image_rgb_list_.push_back(rgb);
}
cv::Mat PlaneViewer::getImageDepth()
{
    unique_lock<mutex> lock(image_depth_mutex_);
    if(image_depth_list_.size() > 0)
    {
        image_depth_ = image_depth_list_.back();
        image_depth_list_.clear();
//        image_depth_ = image_depth_list_.front();
//        image_depth_list_.pop_front();
    }
    return image_depth_;
}
void PlaneViewer::setImageDepth(const cv::Mat &depth)
{
    unique_lock<mutex> lock(image_depth_mutex_);
    image_depth_list_.push_back(depth);
}
/***********************************************************************************************/
//void PlaneViewer::showPlanes(pangolin::View &d_cam, pangolin::OpenGlRenderState &s_cam,
//                             )
void PlaneViewer::showPlanes(pangolin::View &d_cam, pangolin::OpenGlRenderState &s_cam,
                             bool display_cloud, bool display_lines, bool display_normals,
                             bool display_planes, bool display_boundary, bool display_hull,
                             bool project_points, bool display_residual)
{
    d_cam.Activate(s_cam);
    drawAxis(0.2, 5.0);

    PointCloudTypePtr cloud = getCloud();
    if(cloud == NULL || cloud->empty())
        return;
    if( display_cloud )
        drawPointCloud(*cloud, cloud_point_size_);

    unique_lock<mutex> lock(plane_mutex_);
    // Info
    int height = d_cam.GetBounds().h - 20;
    glColor3f(0.0f, 0.0f, 1.0f);
    drawWindowText(400, height, "Segment Result");
//    glColor3f(0.4f, 0.1f, 0.4f);
    glColor3ub(255, 140, 0);
    int temp_h = height-20;
    if(display_lines){
        drawWindowText(400, temp_h, "Lines:   "+std::to_string(lines_.size()));
        temp_h -= 20;
    }
    if(display_normals){
        drawWindowText(400, temp_h, "Normals: "+std::to_string(normals_.size()));
        temp_h -= 20;
    }
    if(display_planes){
        drawWindowText(400, temp_h, "Planes:  "+std::to_string(planes_.size()));
        temp_h -= 20;
        //
        glColor3ub(100, 0, 155);
        drawWindowText(400, temp_h, "Sizes:");
        temp_h -= 20;
        for(size_t i = 0; i < planes_.size(); i++)
        {
            drawWindowText(420, temp_h, std::to_string(i)+": "+std::to_string(planes_[i].indices.size()));
            temp_h -= 20;
        }
    }

    //
    if(display_lines)
    {
        for( size_t i = 0; i < lines_.size(); i++)
        {
            LineType &line = lines_[i];
            if(line.indices.size() == 0)
                continue;
            PointType &pt = cloud->points[line.indices[0]];
            drawPointCloud(*cloud, line.indices, pt.r, pt.g, pt.b, 255, line_point_size_);
            // line id
            glColor4ub(76, 0, 153, 255);
            draw3DText(pt.x, pt.y, pt.z, std::to_string(i));
        }
    }
    if(display_normals)
    {
        for( size_t i = 0; i < normals_.size(); i++)
        {
            NormalType &normal = normals_[i];
            if(normal.indices.size() == 0)
                continue;
            drawPointCloud(*cloud, normal.indices, 153, 153, 255, 255, normal_point_size_);
            // normal id
            glColor4ub(255, 128, 0, 255);
            draw3DText(normal.centroid.x, normal.centroid.y,
                    normal.centroid.z, std::to_string(i));
        }
    }
    if(display_planes)
    {
        for( size_t i = 0; i < planes_.size(); i++)
        {
            PlaneType &plane = planes_[i];
            if(plane.indices.size() == 0)
                continue;
            PointType &pt = cloud->points[plane.indices[0]];
            if(project_points){
                PointCloudPtr pcloud(new PointCloudType);
                projectPoints(*cloud, plane.indices, plane.coefficients, *pcloud);
                drawPointCloud(*pcloud, plane_point_size_);
            }else
                drawPointCloud(*cloud, plane.indices, pt.r, pt.g, pt.b, pt.a, plane_point_size_);
            // plane id
            glColor3f(1.0f, 0.0f, 0.0f);
            draw3DText(plane.centroid.x+plane.coefficients[0]*0.1,
                    plane.centroid.y+plane.coefficients[1]*0.1,
                    plane.centroid.z+plane.coefficients[2]*0.1, std::to_string(i));
        }
    }
    if(display_boundary)
    {
        for( size_t i = 0; i < planes_.size(); i++)
        {
            PlaneType &plane = planes_[i];
            if(plane.boundary_indices.size() == 0)
                continue;
            drawPointCloud(*cloud, plane.boundary_indices, 255, 0, 0, 255, boundary_point_size_);
//            drawCloudHull(*cloud, plane.boundary_indices, 255, 80, 80, 255, cloud_point_size_);
        }
    }
    if(display_hull)
    {
        for( size_t i = 0; i < planes_.size(); i++)
        {
            PlaneType &plane = planes_[i];
            if(plane.hull_indices.size() == 0)
                continue;
            drawPointCloud(*cloud, plane.hull_indices, 0, 0, 255, 255, hull_point_size_);
            drawCloudHull(*cloud, plane.hull_indices, 255, 80, 80, 255, cloud_point_size_);
        }
    }
    if(display_residual)
    {
        pcl::IndicesPtr indices(new std::vector<int>);
        for( size_t i = 0; i < planes_.size(); i++)
        {
            PlaneType &plane = planes_[i];
            if(plane.indices.size() == 0)
                continue;
            ;
            for(std::vector<int>::iterator it = plane.indices.begin(), end = plane.indices.end();
                it != end; it++)
            {
                indices->push_back(*it);
            }
        }

//        PointCloudTypePtr cloud_r(new PointCloudType);
        std::vector<int> indices_r;
        pcl::ExtractIndices<PointType> ei;

        // Create the filtering object
        // Extract the inliers
        ei.setInputCloud( cloud );
        ei.setIndices( indices );
        ei.setNegative( true );
        ei.filter(indices_r);
//        ei.filter( *cloud_r );

        //
        drawPointCloud(*cloud, indices_r, 20, 80, 50, 255, cloud_point_size_);
    }

}

void PlaneViewer::showRuntimes(pangolin::View &d_cam, pangolin::OpenGlRenderState &s_cam, bool display_runtime)
{
    if(!display_runtime)
        return;

    d_cam.Activate(s_cam);
    unique_lock<mutex> lock(runtime_mutex_);
    // show runtimes
    int height = d_cam.GetBounds().h - 20;
//    int height = 100+20*std::min(steps_.size(),runtimes_.size())+20;
    glColor3f(0.0f, 0.0f, 1.0f);
    drawWindowText(10, height, "Runtimes");
    glColor3ub(255, 140, 0);
    float total = 0;
    for( size_t i = 0; i < steps_.size() && i < runtimes_.size(); i++)
    {
        total += runtimes_[i];
        drawWindowText(10, height - 20*(i+1), steps_[i]+": "+std::to_string(runtimes_[i])+" ms");
    }
    glColor3f(1.0f, 0.0f, 1.0f);
    drawWindowText(10, height - 20*(std::min(steps_.size(),runtimes_.size())+1), "Total: "+std::to_string(total)+" ms");
}


void PlaneViewer::showImageRGB(pangolin::View &d_cam, cv::Mat &rgb, pangolin::GlTexture &texture)
{
    if(rgb.empty())
        return;
    texture.Upload(rgb.data, 0, 0, 640, 480, GL_BGR, GL_UNSIGNED_BYTE);
    d_cam.Activate();
    glColor3f(1.0f, 1.0f, 1.0f);
    texture.RenderToViewportFlipY();
}

void PlaneViewer::showPointCloud(pangolin::View &d_cam, pangolin::OpenGlRenderState &s_cam, PointCloudTypePtr &cloud)
{
    d_cam.Activate(s_cam);
    drawAxis(0.5, 2.0);

    if(cloud == NULL || cloud->empty())
        return;
    drawPointCloud(*cloud, cloud_point_size_);
}

/***********************************************************************************************/
void PlaneViewer::drawAxis( float scale, float thickness)
{
    glLineWidth(thickness);
    pangolin::glDrawAxis(scale);
}
void PlaneViewer::draw3DText( float x, float y, float z, const std::string &text)
{
    pangolin::GlText gltext = pangolin::GlFont::I().Text(text);
    gltext.Draw(x, y, z);
}

void PlaneViewer::drawWindowText( float u, float v, const std::string &text)
{
    pangolin::GlText gltext = pangolin::GlFont::I().Text(text);
    gltext.DrawWindow(u, v);
}

void PlaneViewer::drawPointCloud(const pcl::PointCloud<pcl::PointXYZRGBA> &cloud,
                                 float point_size)
{
    glPointSize(point_size);
    glBegin(GL_POINTS);

    pcl::PointCloud<pcl::PointXYZRGBA>::const_iterator end = cloud.end();
    for(pcl::PointCloud<pcl::PointXYZRGBA>::const_iterator it = cloud.begin(); it != end; it++)
    {
        const pcl::PointXYZRGBA &pt = *it;
        if(isnan(pt.z) || isnan(pt.x) || isnan(pt.y))
            continue;
        glColor4ub(pt.r, pt.g, pt.b, pt.a);
        glVertex3f(pt.x, pt.y, pt.z);
    }

    glEnd();
}

void PlaneViewer::drawPointCloud(const pcl::PointCloud<pcl::PointXYZRGBA> &cloud,
                                 const std::vector<int> &indices,
                                 float point_size)
{
    glPointSize(point_size);
    glBegin(GL_POINTS);

    for(size_t i = 0; i < indices.size(); i++)
    {
        const pcl::PointXYZRGBA &pt = cloud.points[indices[i]];
        if(isnan(pt.z) || isnan(pt.x) || isnan(pt.y))
            continue;
        glColor4ub(pt.r, pt.g, pt.b, pt.a);
        glVertex3f(pt.x, pt.y, pt.z);
    }

    glEnd();
}

template<typename PT>
void PlaneViewer::drawPointCloud(const pcl::PointCloud<PT> &cloud,
                                 GLubyte r, GLubyte g, GLubyte b, GLubyte a,
                                 float point_size)
{
    glPointSize(point_size);
    glColor4ub(r, g, b, a);
    glBegin(GL_POINTS);

    typename pcl::PointCloud<PT>::const_iterator end = cloud.end();
    for(typename pcl::PointCloud<PT>::const_iterator it = cloud.begin(); it != end; it++)
    {
        const PT &pt = *it;
        if(isnan(pt.z) || isnan(pt.x) || isnan(pt.y))
            continue;
        glVertex3f(pt.x, pt.y, pt.z);
    }

    glEnd();
}

template<typename PT>
void PlaneViewer::drawPointCloud(const pcl::PointCloud<PT> &cloud,
                                 const std::vector<int> &indices,
                                 GLubyte r, GLubyte g, GLubyte b, GLubyte a,
                                 float point_size)
{
    glPointSize(point_size);
    glColor4ub(r, g, b, a);
    glBegin(GL_POINTS);
    for(size_t i = 0; i < indices.size(); i++)
    {
        const pcl::PointXYZRGBA &pt = cloud.points[indices[i]];
        if(isnan(pt.z) || isnan(pt.x) || isnan(pt.y))
            continue;
        glVertex3f(pt.x, pt.y, pt.z);
    }

    glEnd();
}

template<typename PT>
void PlaneViewer::drawCloudHull(const pcl::PointCloud<PT> &cloud,
                                const std::vector<int> &indices,
                                GLubyte r, GLubyte g, GLubyte b, GLubyte a,
                                float line_width)
{
    glLineWidth(line_width);
    glColor4ub(r, g, b, a);
    glBegin(GL_LINE_LOOP);
    for(size_t i = 0; i < indices.size(); i++)
    {
        const pcl::PointXYZRGBA &pt = cloud.points[indices[i]];
        if(isnan(pt.z) || isnan(pt.x) || isnan(pt.y))
            continue;
        glVertex3f(pt.x, pt.y, pt.z);
    }

    glEnd();
}

void PlaneViewer::projectPoints(const PointCloudType &input, const std::vector<int> &inlier,
                                const Eigen::Vector4f &model_coefficients,
                                PointCloudType &projected_points )
{
    projected_points.header = input.header;
    projected_points.is_dense = input.is_dense;

    Eigen::Vector4f mc(model_coefficients[0], model_coefficients[1], model_coefficients[2], 0);

    // normalize the vector perpendicular to the plane...
    mc.normalize();
    // ... and store the resulting normal as a local copy of the model coefficients
    Eigen::Vector4f tmp_mc = model_coefficients;
    tmp_mc[0] = mc[0];
    tmp_mc[1] = mc[1];
    tmp_mc[2] = mc[2];

    // Allocate enough space and copy the basics
    projected_points.points.resize (inlier.size ());
    projected_points.width    = static_cast<uint32_t> (inlier.size ());
    projected_points.height   = 1;

    typedef typename pcl::traits::fieldList<PointType>::type FieldList;
    // Iterate over each point
    for (size_t i = 0; i < inlier.size (); ++i)
        // Iterate over each dimension
        pcl::for_each_type <FieldList> (pcl::NdConcatenateFunctor <PointType, PointType> (input.points[inlier[i]], projected_points.points[i]));

    // Iterate through the 3d points and calculate the distances from them to the plane
    for (size_t i = 0; i < inlier.size (); ++i)
    {
        // Calculate the distance from the point to the plane
        Eigen::Vector4f p (input.points[inlier[i]].x,
                            input.points[inlier[i]].y,
                            input.points[inlier[i]].z,
                            1);
        // use normalized coefficients to calculate the scalar projection
        float distance_to_plane = tmp_mc.dot (p);

        pcl::Vector4fMap pp = projected_points.points[i].getVector4fMap ();
        pp.matrix () = p - mc * distance_to_plane;        // mc[3] = 0, therefore the 3rd coordinate is safe
    }
}

std::string PlaneViewer::timeToStr()
{
    std::stringstream msg;
    const boost::posix_time::ptime now=
        boost::posix_time::second_clock::local_time();
    boost::posix_time::time_facet *const f=
        new boost::posix_time::time_facet("%Y-%m-%d-%H-%M-%S");
    msg.imbue(std::locale(msg.getloc(),f));
    msg << now;
    return msg.str();
}

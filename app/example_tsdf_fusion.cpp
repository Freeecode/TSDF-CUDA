//
// Created by will on 19-10-17.
//

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "Timer.h"
#include "CUDATSDFIntegrator.h"
#include "Reader.h"

using namespace std;


int main(int argc, char* argv[])
{
    // Path for TSDF 
    std::string dataPath;
    std::string configFile;
    if(argc < 3)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./tsdf_fusion configFile dataPath" << std::endl;
        exit(-1);
    }

    configFile = argv[1];
    dataPath = argv[2];
    std::cout << "Data Path : " << dataPath << std::endl;
    std::cout << "ConfigPath: " << configFile << std::endl;

    CUDATSDFIntegrator Fusion(configFile);
    
    int base_frame_idx = 150;
    int first_frame_idx = 150;
    float num_frames = 50;
    int im_width = 640;
    int im_height = 480;
    float depthfactor = 1000.0f;

    std::ostringstream base_frame_prefix;
    base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;
    std::string base2world_file = dataPath + "frame-" + base_frame_prefix.str() + ".pose.txt";
    float base2world[4 * 4] = {0};
    LoadMatrix(base2world_file, base2world);
    // Invert base frame camera pose to get world-to-base frame transform 
    float base2world_inv[16] = {0};
    invert_matrix(base2world, base2world_inv);

    float depth[im_width * im_height];
    float pose[4 * 4] = {0};
    uchar4* color = new uchar4[im_width * im_height];

    for(int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; ++frame_idx)
    {
 
        std::ostringstream curr_frame_prefix;
        curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
        
        // Path
        std::string depth_im_file = dataPath + "frame-" + curr_frame_prefix.str() + ".depth.png";
        std::string color_im_file = dataPath + "frame-" + curr_frame_prefix.str() + ".color.png";
        std::string pose_file     = dataPath + "frame-" + curr_frame_prefix.str() + ".pose.txt";
        //std::cout << depth_im_file << std::endl;
        // Read current frame
        cv::Mat depth_im = cv::imread(depth_im_file, CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat color_im = cv::imread(color_im_file);
        //std::cout << "Channels: " << color_im.channels() << std::endl;
        cv::cvtColor(color_im, color_im, CV_BGR2RGBA);
        memcpy(color, color_im.data, sizeof(uchar4) * im_height * im_width); 
        
        ReadDepth(depth_im, im_width, im_height, depth, depthfactor);
        LoadMatrix(pose_file, pose);

        float came2base[16]= {0};
        multiply_matrix(base2world_inv, pose, came2base);

        //cv::Mat rgb(im_height, im_width, CV_8UC4);
        //memcpy(rgb.data, color, sizeof(uchar4) * im_width * im_height);
        //cv::imshow("color", rgb);
        Fusion.integrate(depth, color, came2base);
    }

    Fusion.SaveVoxelGrid2SurfacePointCloud(0.2, 0.0);

    return 0;
}
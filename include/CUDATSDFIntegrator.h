//
// Created by will on 20-1-9.
//

#ifndef TSDF_CUDATSDFINTEGRATOR_H
#define TSDF_CUDATSDFINTEGRATOR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "Utils.h"
#include "Voxel.h"

class CUDATSDFIntegrator
{
public:

    CUDATSDFIntegrator();

    CUDATSDFIntegrator(std::string &strSettingFile);

    virtual ~CUDATSDFIntegrator();

public:
    void Initialize();

    void integrate(float* depth_cpu_data, uchar4* color_cpu_data, float* pose);

    void deIntegrate(float* depth_cpu_data, uchar4* color_cpu_data, float* pose);

    void SaveVoxelGrid2SurfacePointCloud(float tsdf_thresh, float weight_thresh);

private:

//host    
    // fx,fy,cx,cy
    float h_camK[4]; 

    // Image resolution
    int h_width;
    int h_height;

    // VoxelSize
    float h_voxelSize;

    // Truncation
    float h_truncation;

    // Grid size
    int h_gridSize;

    // Location of voxel grid origin in base frame coordinates
    float h_grid_origin_x;
    float h_grid_origin_y;
    float h_grid_origin_z;

    // TSDF model
    Voxel* h_SDFBlocks;
    
//host

    int FrameId;
   
//device   
    float* d_camK;
    
    float* d_depth;

    float* d_pose;

    uchar4* d_color;

    Voxel* d_SDFBlocks;
//device 

};

#endif //TSDF_CUDATSDFINTEGRATOR_H

//
// Created by will on 19-11-15.
//

#pragma once

#ifndef TSDF_VOXEL_H
#define TSDF_VOXEL_H

#include <cuda_runtime.h>
#include <cstring>

/**
 * @brief Voxel defination
 */
struct Voxel
{
    float sdf;
    float weight;
    uchar4 color;
    //unsigned char color[4];//R,G,B,A

    // Default Constructor
    __device__ __host__ Voxel()
    {
        sdf = 1.0f; // signed distance function
        weight = 0.0f; // accumulated weight
        color = make_uchar4(0, 0, 0, 0); //R,G,B
    }

    __device__ __host__ void operator=(struct Voxel &v)
    {
        this->sdf = v.sdf;
        this->weight = v.weight;
        this->color = v.color;
    }
};

#endif //TSDF_VOXEL_H

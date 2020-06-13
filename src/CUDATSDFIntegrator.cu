//
// Created by will on 20-1-9.
//

#include "CUDATSDFIntegrator.h"

// CUDA kernel function to integrate a TSDF voxel volume given depth images and color images
__global__ void IntegrateDepthMapKernel(float* d_cam_K, float* d_pose, float* d_depth, uchar4* d_color, 
                                        float voxel_size, float truncation, int height, int width,
                                        int grid_dim_x, int grid_dim_y, int grid_dim_z,
                                        float grid_origin_x, float grid_origin_y, float grid_origin_z, Voxel* d_SDFBlocks)
{
    int pt_grid_z = blockIdx.x;
    int pt_grid_y = threadIdx.x;
    
    
    for(int pt_grid_x = 0; pt_grid_x < grid_dim_x; pt_grid_x++) 
    {
        // Converter voxel center from grid voxel coordinates to real world coordinates
        float pt_x = grid_origin_x + pt_grid_x * voxel_size;
        float pt_y = grid_origin_y + pt_grid_y * voxel_size;
        float pt_z = grid_origin_z + pt_grid_z * voxel_size;

        // Converter world coordinates to current camera coordinates
        float tmp[3] = {0};
        tmp[0] = pt_x - d_pose[0 * 4 + 3];
        tmp[1] = pt_y - d_pose[1 * 4 + 3];
        tmp[2] = pt_z - d_pose[2 * 4 + 3];
        float cam_pt_x = d_pose[0 * 4 + 0] * tmp[0] + d_pose[1 * 4 + 0] * tmp[1] + d_pose[2 * 4 + 0] * tmp[2];
        float cam_pt_y = d_pose[0 * 4 + 1] * tmp[0] + d_pose[1 * 4 + 1] * tmp[1] + d_pose[2 * 4 + 1] * tmp[2];
        float cam_pt_z = d_pose[0 * 4 + 2] * tmp[0] + d_pose[1 * 4 + 2] * tmp[1] + d_pose[2 * 4 + 2] * tmp[2];
        
        if(cam_pt_z <= 0)
            continue;
       
        // d_camK: fx, fy, cx, cy
        int pt_pix_x = roundf(d_cam_K[0] * (cam_pt_x / cam_pt_z) + d_cam_K[2]);
        int pt_pix_y = roundf(d_cam_K[1] * (cam_pt_y / cam_pt_z) + d_cam_K[3]);
        if(pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
            continue;

        //printf("%d, %d\n", pt_pix_x, pt_pix_y);
        float depth_val = d_depth[pt_pix_y * width + pt_pix_x];
        if(depth_val <= 0 || depth_val > 6)
            continue;
        
        float diff = depth_val - cam_pt_z;
        if(diff <= -truncation)
            continue;

        int volume_idx = pt_grid_z * grid_dim_x * grid_dim_y + pt_grid_y * grid_dim_x + pt_grid_x;

        // Integrate TSDF
        float dist = fmin(1.0f, diff / truncation);
        float weight_old = d_SDFBlocks[volume_idx].weight;
        float weight_new = weight_old + 1.0f;
        d_SDFBlocks[volume_idx].weight = weight_new;
        d_SDFBlocks[volume_idx].sdf = (d_SDFBlocks[volume_idx].sdf * weight_old + dist) / weight_new;

        // Integrate Color
        uchar4 RGB = d_color[pt_pix_y * width + pt_pix_x];
        float3 cur_color = make_float3(RGB.x, RGB.y, RGB.z);
        float3 old_color = make_float3(d_SDFBlocks[volume_idx].color.x,
                    d_SDFBlocks[volume_idx].color.y, d_SDFBlocks[volume_idx].color.z);
        float3 new_color;
        new_color.x = fmin(roundf((old_color.x * weight_old + cur_color.x)/weight_new), 255.0f);
        new_color.y = fmin(roundf((old_color.y * weight_old + cur_color.y)/weight_new), 255.0f);;            
        new_color.z = fmin(roundf((old_color.z * weight_old + cur_color.z)/weight_new), 255.0f);; 
        d_SDFBlocks[volume_idx].color = make_uchar4(new_color.x, new_color.y,new_color.z, 255);
    }

}

extern "C" void IntegrateDepthMapCUDA(float* d_cam_K, float* d_pose, float* d_depth, uchar4* d_color, 
                                      float voxel_size, float truncation, int height, int width, int grid_dim, 
                                      float grid_origin_x, float grid_origin_y, float grid_origin_z, Voxel* d_SDFBlocks)
{
   
    const dim3 gridSize(grid_dim);
    const dim3 blockSize(grid_dim);

    //std::cout << "Launch Kernel..." << std::endl;
    IntegrateDepthMapKernel <<< gridSize, blockSize >>> (d_cam_K, d_pose, d_depth, d_color, 
                                                         voxel_size, truncation, height, width, grid_dim, grid_dim, grid_dim, 
                                                         grid_origin_x, grid_origin_y, grid_origin_z, d_SDFBlocks); 

    //cudaError_t status = cudaGetLastError();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
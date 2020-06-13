//
// Created by will on 20-1-9.
//

#include "CUDATSDFIntegrator.h"
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

extern "C" void IntegrateDepthMapCUDA(float* d_cam_K, float* d_pose, float* d_depth, uchar4* d_color, 
                                      float voxel_size, float truncation, int height, int width, int grid_dim, 
                                      float gird_origin_x, float gird_origin_y, float gird_origin_z, Voxel* d_SDFBlocks);

extern "C" void deIntegrateDepthMapCUDA();

CUDATSDFIntegrator::CUDATSDFIntegrator(std::string &strSettingFile)
{   
    //std::cout << strSettingFile  << std::endl;
    cv::FileStorage fSettings(strSettingFile, cv::FileStorage::READ);

    std::cout << "Fusion Params:" << std::endl;
    // Camera Intrinsics
    h_camK[0] = fSettings["Camera.fx"];
    h_camK[1] = fSettings["Camera.fy"];
    h_camK[2] = fSettings["Camera.cx"];
    h_camK[3] = fSettings["Camera.cy"];
    std::cout << "[fx,fy,cx,cy]: " << h_camK[0]<<","<< h_camK[1] <<","<< h_camK[2] <<","<< h_camK[3]<< std::endl;  

    // Image resolution
    h_width  = fSettings["Camera.width"];
    h_height = fSettings["Camera.height"];
    std::cout << "[Width,Height]: "<< h_width << "," << h_height << std::endl;

    // Voxel Size
    h_voxelSize = fSettings["VoxelSize"];
    std::cout << "VoxelSize: " << h_voxelSize << std::endl;

    // Truncation
    h_truncation = fSettings["Truncation"];
    std::cout << "Truncation: " << h_truncation << std::endl;

    // Grid Size
    h_gridSize = fSettings["GridSize"];
    std::cout << "GridSize: " << h_gridSize << std::endl;

    std::cout << "Initialize TSDF ..." << std::endl;
    Initialize();
}

void CUDATSDFIntegrator::Initialize()
{
    // Location of voxel grid origin
    h_grid_origin_x = -1.50f;
    h_grid_origin_y = -1.50f;
    h_grid_origin_z = 0.5f;

    FrameId = 0;

    // allocate memory on CPU
    // TSDF model
    h_SDFBlocks = new Voxel[h_gridSize * h_gridSize * h_gridSize];

    // allocate memory on GPU
    checkCudaErrors(cudaMalloc(&d_camK, 4 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_camK, h_camK, 4 * sizeof(float), cudaMemcpyHostToDevice));
    // TSDF model
    checkCudaErrors(cudaMalloc(&d_SDFBlocks, h_gridSize * h_gridSize * h_gridSize * sizeof(Voxel)));
    // depth data
    checkCudaErrors(cudaMalloc(&d_depth, h_height * h_width * sizeof(float)));
    // color data
    checkCudaErrors(cudaMalloc(&d_color, h_height * h_width * sizeof(uchar4)));
    // pose in base coordinates
    checkCudaErrors(cudaMalloc(&d_pose, 4 * 4 * sizeof(float)));
}

// Integrate depth and color into TSDF model
void CUDATSDFIntegrator::integrate(float* depth_cpu_data, uchar4* color_cpu_data, float* pose_cpu)
{
    //std::cout << "Fusing color image and depth" << std::endl;

    // copy data to gpu
    checkCudaErrors(cudaMemcpy(d_depth, depth_cpu_data, h_height * h_width * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_color, color_cpu_data, h_height * h_width * sizeof(uchar4), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_pose, pose_cpu, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice));

    // Integrate function
    IntegrateDepthMapCUDA(d_camK, d_pose, d_depth, d_color, h_voxelSize, h_truncation, h_height,
                        h_width, h_gridSize, h_grid_origin_x, h_grid_origin_y, h_grid_origin_z, d_SDFBlocks);

    FrameId++;
    std::cout << "Frame Index:" << FrameId << std::endl;
}

// deIntegrate depth and color from TSDF model
void CUDATSDFIntegrator::deIntegrate(float* depth_cpu_data, uchar4* color_cpu_data, float* pose)
{

}

// Compute surface points from TSDF voxel grid and save points to point cloud file
void CUDATSDFIntegrator::SaveVoxelGrid2SurfacePointCloud(float tsdf_thresh, float weight_thresh)
{

    checkCudaErrors(cudaMemcpy(h_SDFBlocks, d_SDFBlocks, 
                        h_gridSize * h_gridSize * h_gridSize * sizeof(Voxel), cudaMemcpyDeviceToHost));
                    
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
    for(int i = 0; i < h_gridSize * h_gridSize * h_gridSize; i++)
    {
        if(std::abs(h_SDFBlocks[i].sdf) < tsdf_thresh && h_SDFBlocks[i].weight > weight_thresh)
        {
            // Compute voxel indices in int for higher positive number range
            int z = floor(i / (h_gridSize * h_gridSize));
            int y = floor((i - (z * h_gridSize * h_gridSize)) / h_gridSize);
            int x = i - (z * h_gridSize * h_gridSize) - (y * h_gridSize);

            // Convert voxel indices to float, and save coordinates to ply file
            float pt_base_x = h_grid_origin_x + (float) x * h_voxelSize;
            float pt_base_y = h_grid_origin_y + (float) y * h_voxelSize;
            float pt_base_z = h_grid_origin_z + (float) z * h_voxelSize;

            pcl::PointXYZRGB point;
            point.x = pt_base_x;
            point.y = pt_base_y;
            point.z = pt_base_z;
            point.r = h_SDFBlocks[i].color.x;
            point.g = h_SDFBlocks[i].color.y;
            point.b = h_SDFBlocks[i].color.z;
            pointcloud.push_back(point);
        }
    }
    std::cout << pointcloud.size() << std::endl;
    pcl::io::savePCDFileBinary("tsdf.pcd", pointcloud);
    
    /*
    // Count total number of points in point cloud
    int num_pts = 0;
    for(int i = 0; i < h_gridSize * h_gridSize * h_gridSize; i++)
    {
        if(std::abs(h_SDFBlocks[i].sdf) < tsdf_thresh && h_SDFBlocks[i].weight > weight_thresh)
            num_pts++;
    }
    
    std::cout << num_pts << std::endl;
    
    // Create header for .ply file
    FILE *fp = fopen("tsdf.ply", "w");
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n", num_pts);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "end_header\n");

    for(int i = 0; i < h_gridSize * h_gridSize * h_gridSize; i++)
    {
        if(std::abs(h_SDFBlocks[i].sdf) < tsdf_thresh && h_SDFBlocks[i].weight > weight_thresh)
        {
            // Compute voxel indices in int for higher positive number range
            int z = floor(i / (h_gridSize * h_gridSize));
            int y = floor((i - (z * h_gridSize * h_gridSize)) / h_gridSize);
            int x = i - (z * h_gridSize * h_gridSize) - (y * h_gridSize);

            // Convert voxel indices to float, and save coordinates to ply file
            float pt_base_x = h_grid_origin_x + (float) x * h_voxelSize;
            float pt_base_y = h_grid_origin_y + (float) y * h_voxelSize;
            float pt_base_z = h_grid_origin_z + (float) z * h_voxelSize;
            fwrite(&pt_base_x, sizeof(float), 1, fp);
            fwrite(&pt_base_y, sizeof(float), 1, fp);
            fwrite(&pt_base_z, sizeof(float), 1, fp);
        }
    }
    fclose(fp);*/

}

// Default deconstructor
CUDATSDFIntegrator::~CUDATSDFIntegrator()
{
    free(h_SDFBlocks);
    checkCudaErrors(cudaFree(d_camK));
    checkCudaErrors(cudaFree(d_SDFBlocks));
    checkCudaErrors(cudaFree(d_depth));
    checkCudaErrors(cudaFree(d_color));
}

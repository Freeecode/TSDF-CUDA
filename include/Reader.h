//
// Created by will on 20-1-9.
//

#ifndef TSDF_READER_H_
#define TSDF_READER_H_

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

void LoadMatrix(std::string filename, float* pose);

bool invert_matrix(const float m[16], float invOut[16]);

void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]);

void ReadDepth(cv::Mat& depth_mat, int width, int height, 
               float* depth, float depthfactor);

#endif
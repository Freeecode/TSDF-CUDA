//
// Created by will on 20-1-9.
//

#ifndef TSDF_TIMER_H
#define TSDF_TIMER_H

#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

/**
 * @brief  GPUTimer class defination
 *
 */

class GPUTimer
{
public:
    GPUTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void GPUStart()
    {
        cudaEventRecord(start, 0);
    }

    void GPUStop()
    {
        cudaEventRecord(stop, 0);
    }

    void GPUElapsed()
    {
        float elapsed; //ms
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);

        std::cout << "Total cost on GPU: " << elapsed << " ms" << std::endl;
        //return elapsed;
    }

    ~GPUTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

private:
    cudaEvent_t start;
    cudaEvent_t stop;
};

/**
 * @brief CPUTimer class defination
 *
 */

class CPUTimer
{
public:
    
    CPUTimer()
    {

    }

    void CPUStart()
    {
        start = std::chrono::system_clock::now();
    }

    void CPUStop()
    {
        stop = std::chrono::system_clock::now();
    }

    void CPUElapsed()
    {
        std::chrono::duration<double> elapsed_seconds =
                std::chrono::duration_cast<std::chrono::duration<double >>(stop - start);
        std::cout << "Total cost on CPU: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;
    }

    virtual ~CPUTimer()
    {
    }

private:
    //std::chrono::system_clock::time_point start, stop;
    std::chrono::time_point<std::chrono::system_clock> start, stop;
};

#endif //TSDF_TIMER_H

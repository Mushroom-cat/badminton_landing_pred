#pragma once
/*
 * @file	
 * @brief	
 * @author	sjtu 3-122b 
 * @date	2025
 * @copyright	All rights reserved
 * @details	
*/
#include <chrono>
#include <opencv2/opencv.hpp>
/************************************************************************/
/*                                                                      
                                                                        */
/************************************************************************/
using namespace cv;
using namespace std;

struct MyTimer
{
    void reset()
    {
        start = std::chrono::high_resolution_clock::now();
    }

    double elapse()const //ms
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
};

struct InferPose  //人体+球拍关键点检测接口
{
    virtual vector<vector<Point2f>> run(const Mat& img) = 0;
};

using InferPosePtr = std::shared_ptr<InferPose>;
InferPosePtr infer_pose_create_ocv_rt(const string& model);    //关键点检测  OpenCV   后端推理


struct InferLandPoint  //落点预测接口
{
    virtual Point3f run(const vector<vector<Point3f>>& framPoseList, const vector<Point3f>& ballList) = 0;
};

using InferLandPointPtr = std::shared_ptr<InferLandPoint>;  

InferLandPointPtr infer_landpoint_create_onnx_rt(const string& model, bool isBefore);   //落点预测 OnnxRuntime 后端推理


struct InferDet  //羽毛球检测
{
    virtual vector<Rect> run(const Mat& img) = 0;
};

using InferDetPtr = std::shared_ptr<InferDet>;

InferDetPtr infer_det_create_ocv_rt(const string& model);    //目标检测  OpenCV   后端推理


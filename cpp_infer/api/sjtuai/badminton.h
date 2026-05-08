#pragma once
/*
 * @file	
 * @brief	
 * @author	sjtu 3-122b 
 * @date	2025
 * @copyright	All rights reserved
 * @details	
*/
#include <deque> //双端队列,支持快速的从 头部、尾部修改数据
#include <opencv2/opencv.hpp>
#include "decl.h"
/************************************************************************/
/*   
羽毛球落点预测
                                                                        */
/************************************************************************/
NAMESPACE_SJTU_AI_BEGIN
using cv::Mat; using cv::Rect; using cv::Point3f; using cv::Point2f; 
using std::string; using std::vector;


using KeyPose = vector<Point2f>;            
using KeyPoseList = std::deque<KeyPose>;    
using FramePose = vector<Point3f>;
using FramePoseList = std::deque<FramePose>;
using BallList = std::deque<Point3f>;

struct SJTU_AI_DECL Badminton
{
    Badminton();
    ~Badminton();

    //实时识别图像
        //1. 调用 姿态识别
        //2. 调用 羽毛球检测
        //3. 内部判断是否触发落点预测,并返回落点
    bool recoImg(const Mat& leftImg, const Mat& rightImg, Point3f& landingPos);


    //左右相机 人体+球拍姿态识别
    bool detPose(const Mat& leftImg, const Mat& rightImg, vector<Point2f>& leftPose, vector<Point2f>& rightPose);
    
    //左右相机 羽毛球检测: 根据姿态的球拍位置触发
    bool detBall(const Mat& leftImg, const Mat& rightImg, 
            const vector<Point2f>& leftPose, const vector<Point2f>& rightPose,  
            Rect& leftRect, Rect& rightRect);

    //落点预测：回归落点   根据球与拍子的距离ROI信息，才检测和激活检测:  0. 距离    1. 方向改变
        //根据输入的羽毛球的位置,进行落点预测
    bool landPointPredict(const Rect& leftRect, const Rect& rightRect, Point3f& landingPos); //直接根据回归落点


    Badminton& clear();
    //////////////////////////////////////////////////////////////////////////
    //相机参数
    Badminton& setCameraIntrinsic(const string& fName); //设置相机内参
    Badminton& setCameraExtrinsic(const string& fName); //设置相机外参

    //////////////////////////////////////////////////////////////////////////
    // 姿态检测 相关设置
    Badminton& setPoseLeftRoi(const Rect& roi);  //设置左相机的ROI
    Badminton& setPoseRightRoi(const Rect& roi); //设置右相机的ROI
    Badminton& setPoseModel(const string& fName); //设置Pose检测模型
    
    const FramePoseList& getFramePoseList()const;   //获取所有的3D坐标序列
    const KeyPoseList& getLeftKeyPoseList()const;  //获取左相机的2D关键点序列
    const KeyPoseList& getRightKeyPoseList()const; //获取右相机的2D关键点序列

    //////////////////////////////////////////////////////////////////////////
    //羽毛球检测
    Badminton& setDetModel(const string& fName); //设置Rect检测模型

    //////////////////////////////////////////////////////////////////////////
    // 落点预测 相关设置
    Badminton& setDistBatBall2D(double dist); //设置触发落点预测的 球拍和球的像素距离
    Badminton& setDistBatBall3D(double dist); //设置触发落点预测的 球拍和球的空间距离

    Badminton& setBeforeLandModel(const string& fName); //设置击球前 落点预测模型
    Badminton& setAfterLandModel(const string& fName);  //设置击球后 落点预测模型

private:
    struct Impl;
    std::shared_ptr<Impl> pImpl;
};

using BadmintonPtr = std::shared_ptr<Badminton>;

NAMESPACE_SJTU_AI_END
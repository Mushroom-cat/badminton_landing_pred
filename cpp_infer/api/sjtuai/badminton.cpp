/*
 * @file	
 * @brief	
 * @author	sjtu 3-122b 
 * @date	2025
 * @copyright	All rights reserved
 * @details	
*/
#include "decl.h"
#include "badminton.h"
#include "detail/utils.h"
#include "detail/infer.h"
#include "detail/BS_thread_pool.hpp"
/************************************************************************/
/*                                                                      
                                                                        */
/************************************************************************/

NAMESPACE_SJTU_AI_BEGIN

using mutex_lock_guard = std::lock_guard<std::mutex>;
using mutex_lock = mutex_lock_guard;

static int const BAT_POSE_KEY_POINTS = 21;
static int const BAT_QUEUE_SIZE = 100;


struct Badminton::Impl
{
    //相机标定
    CamCalib camCalib;

    //姿态识别
    Rect leftPoseRoi;
    Rect rightPoseRoi;
    Size poseRoiSize = { 280, 280 };
    FramePoseList poseList;
    KeyPoseList leftPoseList, rightPoseList;
    InferPosePtr inferPose; //姿态检测模型推理
    std::mutex poseMutex;

    //羽毛球检测
    Size ballRoiSize = { 224, 224 };
    InferDetPtr inferDet; //目标检测模型推理
    std::mutex detMutex;
    
    Mat leftLastImg, rightLastImg;
    int binThresh = 10;
    BallList ballList;

    //击球时刻检测
    double distBatBall2D = 20;
    double distBatBall3D = 30;

    //落点预测
    int FrameCount = 50;
    InferLandPointPtr inferBeforeLand, inferAfterLand;

    BS::light_thread_pool pool = BS::light_thread_pool(4);

    bool needInferLandPoint(const Point2f& ballLeft, const Point2f& ballRight)const
    { 
        if (poseList.size() < FrameCount)
            return false;

        double dist3d = 0;
        //3D 空间距离
        {
            auto& pose = poseList.front();
            auto& pt1 = pose.at(17);
            auto& pt2 = pose.at(18);
            auto& pt3 = pose.at(19);
            auto& pt4 = pose.at(20);

            Point3f cent;
            cent.x = (pt1.x + pt2.x + pt3.x + pt4.x) / 4;
            cent.y = (pt1.y + pt2.y + pt3.y + pt4.y) / 4;
            cent.z = (pt1.z + pt2.z + pt3.z + pt4.z) / 4;

            //X,Y轴       Z轴方向距离
            auto ballPoint = camCalib.cvtPoint(ballLeft, ballRight);
            float dx = cent.x - ballPoint.x;
            float dy = cent.y - ballPoint.y;
            float dz = cent.z - ballPoint.z;
            dist3d = std::sqrt(dx * dx + dy * dy + dz * dz);

            cout << "球拍3D位置:" << cent  << ", 球与拍3D距离:" << dist3d << ", (阈值:" << distBatBall3D << ")" << endl;
        }

        double dist2dLeft = 0;
        {
            auto& keyLeft = leftPoseList.front();
            auto& pt1 = keyLeft.at(17);
            auto& pt2 = keyLeft.at(18);
            auto& pt3 = keyLeft.at(19);
            auto& pt4 = keyLeft.at(20);

            Point2f cent;
            cent.x = (pt1.x + pt2.x + pt3.x + pt4.x) / 4;
            cent.y = (pt1.y + pt2.y + pt3.y + pt4.y) / 4;

            //X,Y轴
            float dx = cent.x - ballLeft.x;
            float dy = cent.y - ballLeft.y;
            dist2dLeft = std::sqrt(dx * dx + dy * dy);

            cout << "球拍Left 2D位置:" << cent << ", 球与拍2D距离:" << dist2dLeft << ", (阈值:" << distBatBall2D << ")" << endl;
        }

        double  dist2dRight = 0;
        {
            auto& keyRight = rightPoseList.front();
            auto& pt1 = keyRight.at(17);
            auto& pt2 = keyRight.at(18);
            auto& pt3 = keyRight.at(19);
            auto& pt4 = keyRight.at(20);

            Point2f cent;
            cent.x = (pt1.x + pt2.x + pt3.x + pt4.x) / 4;
            cent.y = (pt1.y + pt2.y + pt3.y + pt4.y) / 4;

            //X,Y轴 
            float dx = cent.x - ballRight.x;
            float dy = cent.y - ballRight.y;
            dist2dRight = std::sqrt(dx * dx + dy * dy);

            cout << "球拍Left 2D位置:" << cent << ", 球与拍2D距离:" << dist2dRight << ", (阈值:" << distBatBall2D << ")" << endl;
        }

        bool OK = dist3d <= distBatBall3D && dist2dLeft <= distBatBall2D && dist2dRight <= distBatBall2D;
        if (OK)
            cout << "=== 距离近, 此时是击球时刻, 触发落点预测!!!!" << endl;
        else
            cout << "=== 距离远，无法触发落点预测" << endl;

        return OK;
    }
};


Badminton::Badminton() :pImpl(new Impl)
{
}

Badminton::~Badminton()
{
}

bool Badminton::recoImg(const Mat& leftImg, const Mat& rightImg, Point3f& landingPos)
{
    bool isOK = false;
    vector<Point2f> leftPose, rightPose;
    Rect leftBall, rightBall;

    //姿态检测
    bool ok = detPose(leftImg, rightImg, leftPose, rightPose);
    if (!ok) return false;

    //羽毛球检测
    ok = detBall(leftImg, rightImg, leftPose, rightPose, leftBall, rightBall);
    if (!ok) return false;

    //落点预测    
    ok = landPointPredict(leftBall, rightBall, landingPos);
    if (!ok) return false;

    return isOK;
}

bool Badminton::detPose(const Mat& leftImg, const Mat& rightImg, vector<Point2f>& leftPts, vector<Point2f>& rightPts)  //左右相机 人体+球拍姿态识别
{
    auto leftRoi = pImpl->leftPoseRoi;
    auto rightRoi = pImpl->rightPoseRoi;

    //if (!pImpl->leftPoseList.empty())
    //    leftRoi = points_gen_roi(pImpl->leftPoseList.front(), pImpl->poseRoiSize);

    //if (!pImpl->rightPoseList.empty())
    //    rightRoi = points_gen_roi(pImpl->rightPoseList.front(), pImpl->poseRoiSize);

    //cout << "leftRoi:" << leftRoi << endl;
    //cout << "rightRoi:" << rightRoi << endl;

    std::future<KeyPointsList> leftTask = pImpl->pool.submit_task([=] {
        mutex_lock ml(pImpl->poseMutex);
        return pImpl->inferPose->run(leftImg(leftRoi));
        }
    );

    std::future<KeyPointsList> rightTask = pImpl->pool.submit_task([=] {
        mutex_lock ml(pImpl->poseMutex);
        return pImpl->inferPose->run(rightImg(rightRoi));
        }
    );

    auto leftPoseList = leftTask.get();
    auto rightPoseList = rightTask.get();

    if (leftPoseList.empty() || rightPoseList.empty())
        return false;

    auto& leftPose = leftPoseList.front();
    auto& rightPose = rightPoseList.front();
    if (leftPose.size() != BAT_POSE_KEY_POINTS || rightPose.size() != BAT_POSE_KEY_POINTS)
        return false;

    for (auto& pt : leftPose)
    {
        pt.x += leftRoi.x;
        pt.y += leftRoi.y;
    }

    for (auto& pt : rightPose)
    {
        pt.x += rightRoi.x;
        pt.y += rightRoi.y;
    }
    cout << "左边点:" << leftPose.size() << "                 右边点:" << rightPose.size() << endl;

    auto framePose = pImpl->camCalib.cvtPoint(leftPose, rightPose);

    for (int i = 0; i < leftPose.size() && i < rightPose.size() && i < framePose.size(); ++i)
    {
        cout << leftPose[i].x << "," << leftPose[i].y
            << "          " << rightPose[i].x << ", " << rightPose[i].y
            << "          " << framePose[i].x << "," << framePose[i].y << "," << framePose[i].z
            << endl;
    }

    if (!framePose.empty())
    {
        leftPts = leftPose;
        rightPts = rightPose;
        pImpl->leftPoseList.push_front(leftPose);
        pImpl->rightPoseList.push_front(rightPose);
        pImpl->poseList.push_front(framePose);
        if (pImpl->leftPoseList.size() > BAT_QUEUE_SIZE) pImpl->leftPoseList.resize(BAT_QUEUE_SIZE);
        if (pImpl->rightPoseList.size() > BAT_QUEUE_SIZE) pImpl->rightPoseList.resize(BAT_QUEUE_SIZE);
        if (pImpl->poseList.size() > BAT_QUEUE_SIZE) pImpl->poseList.resize(BAT_QUEUE_SIZE);
        cout << "添加姿态到队列, 队列尺寸:" << pImpl->poseList.size() << "\n";
    }
    return true;
}


bool Badminton::detBall(const Mat& leftImg, const Mat& rightImg,
    const vector<Point2f>& leftPose, const vector<Point2f>& rightPose,
    Rect& leftRect, Rect& rightRect)
{
    auto& ballRoi = pImpl->ballRoiSize;

    float cx_l = (leftPose[20].x + leftPose[19].x + leftPose[18].x + leftPose[17].x) / 4;
    float cy_l = (leftPose[20].y + leftPose[19].y + leftPose[18].y + leftPose[17].y) / 4;

    float cx_r = (rightPose[20].x + rightPose[19].x + rightPose[18].x + rightPose[17].x) / 4;
    float cy_r = (rightPose[20].y + rightPose[19].y + rightPose[18].y + rightPose[17].y) / 4;

    Rect leftRoi, rightRoi;

    leftRoi.x = cx_l - ballRoi.width / 2;
    leftRoi.y = cy_l - ballRoi.height / 2;
    leftRoi.width = ballRoi.width;
    leftRoi.height = ballRoi.height;
    rect_normal(leftRoi, leftImg);

    rightRoi.x = cx_r - ballRoi.width / 2;
    rightRoi.y = cy_r - ballRoi.height / 2;
    rightRoi.width = ballRoi.width;
    rightRoi.height = ballRoi.height;
    rect_normal(rightRoi, rightImg);

    cout << "detBall leftRoi:" << leftRoi << endl;
    cout << "detBall rightRoi:" << rightRoi << endl;

    Mat leftDifImg, rightDifImg;

    if (pImpl->leftLastImg.empty())
    {
        leftDifImg = leftImg;
    }else {
        leftDifImg = calcDiffGray(pImpl->leftLastImg, leftImg, pImpl->binThresh);
    }

    if (pImpl->rightLastImg.empty())
    {
        rightDifImg = rightImg;
    }else {
        rightDifImg = calcDiffGray(pImpl->rightLastImg, rightImg, pImpl->binThresh);
    }


    std::future<vector<Rect>> leftTask = pImpl->pool.submit_task([=] {
        mutex_lock ml(pImpl->detMutex);
        return pImpl->inferDet->run(leftDifImg(leftRoi));
        }
    );

    std::future<vector<Rect>> rightTask = pImpl->pool.submit_task([=] {
        mutex_lock ml(pImpl->detMutex);
        return pImpl->inferDet->run(rightDifImg(rightRoi));
        }
    );

    auto leftRs = leftTask.get();
    auto rightRs = rightTask.get();
    //auto leftRs = pImpl->leftInferDet->run(leftImg(leftRoi));
    //auto rightRs = pImpl->rightInferDet->run(rightImg(rightRoi));

    cout << "left Det size=" << leftRs.size() << ", right Det size=" << rightRs.size() << endl;

    pImpl->leftLastImg = leftImg;
    pImpl->rightLastImg = rightImg;

    
    Point3f ballPt;
    if (leftRs.empty() || rightRs.empty())
    {
        pImpl->ballList.push_front(ballPt);
        if (pImpl->ballList.size() > BAT_QUEUE_SIZE) pImpl->ballList.resize(BAT_QUEUE_SIZE);
        return false;
    }
        

    float minDist = INT_MAX;
    for (auto& r : leftRs)
    {
        r.x += leftRoi.x;
        r.y += leftRoi.y;

        float cx1 = r.x + r.width / 2;
        float cy1 = r.y + r.height / 2;
        auto dist = (cx_l - cx1)* (cx_l - cx1) + (cy1 - cy1) * (cy_l - cy1);
        if (minDist > dist)
        {
            minDist = dist;
            leftRect = r;
        }
    }

    minDist = INT_MAX;
    for (auto& r : rightRs)
    {
        r.x += rightRoi.x;
        r.y += rightRoi.y;

        float cx1 = r.x + r.width / 2;
        float cy1 = r.y + r.height / 2;
        auto dist = (cx_r - cx1) * (cx_r - cx1) + (cy1 - cy1) * (cy_r - cy1);
        if (minDist > dist)
        {
            minDist = dist;
            rightRect = r;
        }
    }

    pImpl->ballList.emplace_front( pImpl->camCalib.cvtPoint(rect_center(leftRect), rect_center(rightRect)));
    if (pImpl->ballList.size() > BAT_QUEUE_SIZE) pImpl->ballList.resize(BAT_QUEUE_SIZE);
    return true;
}


bool Badminton::landPointPredict(const Rect& leftRect, const Rect& rightRect, Point3f& landingPos) //直接根据回归落点
{
    if (!pImpl->needInferLandPoint(rect_center(leftRect), rect_center(rightRect)))
        return false;

    cout << "pose list size=" << pImpl->poseList.size() << ", ball list size=" << pImpl->ballList.size() << endl;

    if (pImpl->poseList.size() != pImpl->ballList.size())
    {
        cout << "warning!!! pose and ball size not equal!" << endl;
        return false;
    }

    vector<FramePose> framePose;
    vector<Point3f> ballPose;
    for (int i = pImpl->FrameCount - 1; i >= 0; --i)
    {
        framePose.push_back(pImpl->poseList[i]);
        ballPose.push_back(pImpl->ballList[i]);
    }
    landingPos = pImpl->inferBeforeLand->run(framePose, ballPose);
    return true;
}

Badminton& Badminton::clear()
{
    pImpl->poseList.clear();
    pImpl->leftPoseList.clear();
    pImpl->rightPoseList.clear();
    return *this;
}

Badminton& Badminton::setCameraIntrinsic(const string& fName) //设置相机内参
{
    pImpl->camCalib.setIntrinsicFile(fName);
    return *this;
}

Badminton& Badminton::setCameraExtrinsic(const string& fName) //设置相机外参
{
    pImpl->camCalib.setExtrinsicFile(fName);
    return *this;
}

Badminton& Badminton::setPoseLeftRoi(const Rect& roi)  //设置左相机的ROI
{
    pImpl->leftPoseRoi = roi;
    cout << "=== 设置左相机Pose ROI:" << roi << "\n\n";
    return *this;
}

Badminton& Badminton::setPoseRightRoi(const Rect& roi) //设置右相机的ROI
{
    pImpl->rightPoseRoi = roi;
    cout << "=== 设置右相机Pose ROI:" << roi << "\n\n";
    return *this;
}

Badminton& Badminton::setPoseModel(const string& fName) //设置姿态检测模型
{
    pImpl->inferPose = infer_pose_create_ocv_rt(fName);
    return *this;
}


const FramePoseList& Badminton::getFramePoseList()const   //获取所有的3D坐标序列
{
    return pImpl->poseList;
}

const KeyPoseList& Badminton::getLeftKeyPoseList()const  //获取左相机的2D关键点序列
{
    return pImpl->leftPoseList;
}

const KeyPoseList& Badminton::getRightKeyPoseList()const //获取右相机的2D关键点序列
{
    return pImpl->rightPoseList;
}

Badminton& Badminton::setDistBatBall2D(double dist) //设置触发落点预测的 球拍和球的像素距离
{
    pImpl->distBatBall2D = dist;
    cout << "=== 球与拍的触发2D距离:" << dist << "\n\n";
    return *this;
}

Badminton& Badminton::setDistBatBall3D(double dist) //设置触发落点预测的 球拍和球的空间距离
{
    pImpl->distBatBall3D = dist;
    cout << "=== 球与拍的触发3D距离:" << dist << "\n\n";
    return *this;
}


Badminton& Badminton::setBeforeLandModel(const string& fName) //设置击球前 落点预测模型
{
    pImpl->inferBeforeLand = infer_landpoint_create_onnx_rt(fName, true);
    return *this;
}

Badminton& Badminton::setAfterLandModel(const string& fName) //设置击球后 落点预测模型
{
    pImpl->inferAfterLand = infer_landpoint_create_onnx_rt(fName, false);
    return *this;
}

Badminton& Badminton::setDetModel(const string& fName) //设置Rect检测模型
{
    pImpl->inferDet = infer_det_create_ocv_rt(fName);
    return *this;
}

NAMESPACE_SJTU_AI_END
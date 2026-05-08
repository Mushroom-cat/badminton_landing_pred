/*
 * @file	
 * @brief	
 * @author	Liu Kuan
 * @date	2025
 * @copyright	All rights reserved
 * @details	
*/
#include <dbg.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include "api/sjtuai/badminton.h"

using namespace std;
using namespace cv;
/************************************************************************/
/*                                                                      
                                                                        */
/************************************************************************/

bool s_contains(const string& txt, const string& pat);
string s_replace(const string& txt, const string& oldFmt, const string& newFmt);
bool s_ends(const string& txt, const string& pattern);
vector<string> files_under(const string& dirName);

int main(int argc, char** argv)
{
    string dirName = R"(data\test_seq\round1_data_session01-000766\left)";

    string cameraIntrinsic = R"(data/20260202_intrinsic.yml)", 
            cameraExtrinsic = R"(data/20260202_extrinsic.yml)";

    string poseModel = R"(data/badminton_pose_20260312.onnx)", 
            detModel = R"(data/badminton_det_20260317_diff_gray.onnx)", 
            landBeforeModel = R"(data/before_20260323_165346.onnx)",
            landAfterModel = R"(data/after_20260323_175728.onnx)"
            ;

    double dist2d = 20;
    double dist3d = 20;
    sjtu_ai::BadmintonPtr badminton;
    cv::Rect leftRoi = { 139, 609, 986 - 139, 958 - 609 },
            rightRoi = { 698, 567, 1709 - 698, 987 - 567 };

    badminton.reset(new sjtu_ai::Badminton);
    badminton->setCameraIntrinsic(cameraIntrinsic);
    badminton->setCameraExtrinsic(cameraExtrinsic);
    badminton->setPoseLeftRoi(leftRoi);
    badminton->setPoseRightRoi(rightRoi);
    badminton->setPoseModel(poseModel);
    badminton->setDetModel(detModel);
    badminton->setBeforeLandModel(landBeforeModel);
    badminton->setAfterLandModel(landAfterModel);
    badminton->setDistBatBall2D(dist2d);
    badminton->setDistBatBall3D(dist3d);

    vector<string> fImgList = files_under(dirName);

    for (auto& fImg : fImgList)
    {
        string leftF, rightF;
        if (s_contains(fImg, "left")) leftF = fImg;
        if (s_contains(fImg, "right")) rightF = fImg;
        if (leftF.empty()) leftF = s_replace(rightF, "right", "left");
        if (rightF.empty()) rightF = s_replace(leftF, "left", "right");
        //KERR_CHECK(!leftF.empty() && !rightF.empty());

        auto leftImg = cv::imread(leftF);
        auto rightImg = cv::imread(rightF);
        cout << "load img pair:\n" << leftF << "\n" << rightF << endl;
         
        //×ËÌ¬¼ì²â
        Mat leftImgDraw, rightImgDraw;
        do {
            bool isOK = false;
            vector<Point2f> leftPose, rightPose;
            Rect leftBall, rightBall;
            Point3f landPoint;

            isOK = badminton->detPose(leftImg, rightImg, leftPose, rightPose);
            cout << "det pose:" << isOK;
            if (!isOK)  
                break;

            //ÓðÃ«Çò¼ì²â
            isOK = badminton->detBall(leftImg, rightImg, leftPose, rightPose, leftBall, rightBall);
            cout << "det ball:" << isOK;
            if (!isOK)  
                break;

            cv::rectangle(leftImgDraw, leftBall, cv::Scalar(0, 0, 255), 2);
            cv::rectangle(rightImgDraw, rightBall, cv::Scalar(0, 0, 255), 2);

            //ÂäµãÔ¤²â    
            isOK = badminton->landPointPredict(leftBall, rightBall, landPoint);
            cout << "land point:" << isOK;
            if (!isOK)  
                break;

            cout << "Ô¤²âÂäµã:" << landPoint << endl;

        } while (false);
    }
    return 0;
}


bool s_contains(const string& txt, const string& pat)
{
    return boost::algorithm::contains(txt, pat);
}

string s_replace(const string& txt, const string& oldFmt, const string& newFmt)
{
    return boost::algorithm::replace_all_copy(txt, oldFmt, newFmt);
}

bool s_ends(const string& txt, const string& pattern)
{
    return boost::algorithm::ends_with(txt, pattern);
}

vector<string> files_under(const string& dirName)
{
    vector<string> fImgList;
    boost::filesystem::recursive_directory_iterator dis(dirName), die;
    for (; dis != die; ++dis)
    {
        auto pf = dis->path();
        string pfstr = pf.string();
        if (boost::filesystem::is_regular_file(pf) && s_ends(pfstr, ".jpg"))
            fImgList.push_back(pfstr);
    }
    return fImgList;
}

#include <_auto_inc/OpenCV.h>
#include <_auto_inc/onnxruntime.h>

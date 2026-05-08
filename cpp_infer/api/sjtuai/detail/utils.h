#pragma once
/*
 * @file	
 * @brief	
 * @author	sjtu 3-122b 
 * @date	2025
 * @copyright	All rights reserved
 * @details	
*/

#include <opencv2/opencv.hpp>
/************************************************************************/
/*                                                                      
                                                                        */
/************************************************************************/
using namespace cv;
using namespace std;

using KeyPoints = vector<Point2f>;
using KeyPointsList = vector<KeyPoints>;

inline Point rect_center(const Rect& r)
{
    return Point(r.x + r.width/2, r.y + r.height/2);
}

inline bool hit_point(const vector<Point3f>& pts)
{
    return true;
}


inline Rect points_gen_roi(const vector<Point2f>& pts, Size sz)
{
    Point2f cent;

    for (auto& p : pts)
        cent += p;

    if (!pts.empty())
    {
        cent.x /= pts.size();
        cent.y /= pts.size();
    }

    return Rect( cent.x - sz.width/2, cent.y - sz.height/2,  sz.width, sz.height);
}

inline Rect& rect_normal(Rect& r, const Size& sz)
{
    if (r.x < 0) r.x = 0;
    if (r.y < 0) r.y = 0;
    if (r.x + r.width >= sz.width) r.width = sz.width - r.x;
    if (r.y + r.height >= sz.height) r.height = sz.height - r.y;
    return r;
}

inline Rect& rect_normal(Rect& r, const Mat& img)
{
    return rect_normal(r, img.size());
}

struct CamCalib
{
    bool setIntrinsicFile(const string& fname)
    {
        cv::FileStorage fs(fname, cv::FileStorage::READ);
        fs["CM1"] >> CM1; CM1.convertTo(CM1, CV_32F);
        fs["D1"] >> D1; D1.convertTo(D1, CV_32F);
        fs["R1"] >> R1; R1.convertTo(R1, CV_32F);
        fs["P1"] >> P1; P1.convertTo(P1, CV_32F);
        fs["CM2"] >> CM2; CM2.convertTo(CM2, CV_32F);
        fs["D2"] >> D2; D2.convertTo(D2, CV_32F);
        fs["R2"] >> R2; R2.convertTo(R2, CV_32F);
        fs["P2"] >> P2; P2.convertTo(P2, CV_32F);
        fs["Q"] >> Q; Q.convertTo(Q, CV_32F);

        cout << "===读取相机内参:\n";
        cout << "CM1:" << CM1 << "\n\n";
        cout << "D1:" << D1 << "\n\n";
        cout << "R1:" << R1 << "\n\n";
        cout << "P1:" << P1 << "\n\n";
        cout << "CM2:" << CM2 << "\n\n";
        cout << "D2:" << D2 << "\n\n";
        cout << "R2:" << R2 << "\n\n";
        cout << "P2:" << P2 << "\n\n";
        cout << "Q:" << Q << "\n\n";
        cout << "\n\n";
        return true;
    }

    bool setExtrinsicFile(const string& fname)
    {
        cv::FileStorage fs(fname, cv::FileStorage::READ);
        fs["ThreeDimTransMat"] >> trans_mat;  trans_mat.convertTo(trans_mat, CV_32F);

        cout << "===读取相机外参:\n" << trans_mat << "\n\n";
        return true;
    }

    Point3f cvtPoint(const Point2f left_point, const Point2f right_point)const
    {
        Point3f pt3;

        // 准备点数据
        std::vector<cv::Point2f> pts_left, pts_right;
        pts_left.push_back(left_point);
        pts_right.push_back(right_point);

        // 转换为适合undistortPoints的格式
        cv::Mat pts_left_mat(1, 1, CV_32FC2, &pts_left[0]);
        cv::Mat pts_right_mat(1, 1, CV_32FC2, &pts_right[0]);

        // 去畸变和校正
        cv::Mat rect_left, rect_right;
        cv::undistortPoints(pts_left_mat, rect_left, CM1, D1, R1, P1);
        cv::undistortPoints(pts_right_mat, rect_right, CM2, D2, R2, P2);

        // 打印校正前后的点
        //std::cout << "before: " << left_point << ", " << right_point << std::endl;

        cv::Point2f u_left_rect(rect_left.at<cv::Vec2f>(0, 0)[0], rect_left.at<cv::Vec2f>(0, 0)[1]);
        cv::Point2f u_right_rect(rect_right.at<cv::Vec2f>(0, 0)[0], rect_right.at<cv::Vec2f>(0, 0)[1]);

        //std::cout << "after: " << u_left_rect << ", " << u_right_rect << std::endl;

        // 计算视差
        float disparity = u_left_rect.x - u_right_rect.x;
        //std::cout << u_left_rect.x << ", " << u_right_rect.x << ", " << disparity << ", 1.0" << std::endl;

        // 计算左摄像头的3D点
        cv::Mat point_2d_disp_left = (cv::Mat_<float>(4, 1) << u_left_rect.x, u_left_rect.y, disparity, 1.0);
        cv::Mat point_3d_hom_left = Q * point_2d_disp_left;
        cv::Point3f point_3d_l(
            point_3d_hom_left.at<float>(0) / point_3d_hom_left.at<float>(3),
            point_3d_hom_left.at<float>(1) / point_3d_hom_left.at<float>(3),
            point_3d_hom_left.at<float>(2) / point_3d_hom_left.at<float>(3)
        );
        //std::cout << "Left 3D Point (X, Y, Z): " << point_3d_l << std::endl;

        // 计算右摄像头的3D点
        cv::Mat point_2d_disp_right = (cv::Mat_<float>(4, 1) << u_right_rect.x, u_right_rect.y, disparity, 1.0);
        cv::Mat point_3d_hom_right = Q * point_2d_disp_right;
        cv::Point3f point_3d_r(
            point_3d_hom_right.at<float>(0) / point_3d_hom_right.at<float>(3),
            point_3d_hom_right.at<float>(1) / point_3d_hom_right.at<float>(3),
            point_3d_hom_right.at<float>(2) / point_3d_hom_right.at<float>(3)
        );
        //std::cout << "Right 3D Point (X, Y, Z): " << point_3d_r << std::endl;

        // 取平均
        cv::Point3f point_3d(
            (point_3d_l.x + point_3d_r.x) / 2.0f,
            (point_3d_l.y + point_3d_r.y) / 2.0f,
            (point_3d_l.z + point_3d_r.z) / 2.0f
        );
        //std::cout << "Averaged 3D Point (X, Y, Z): " << point_3d << std::endl;

        // 读取外参并应用变换
        // 转换为齐次坐标
        cv::Mat point_homogeneous = (cv::Mat_<float>(4, 1) << point_3d.x, point_3d.y, point_3d.z, 1.0f);

        // 应用外参变换矩阵
        cv::Mat world_point = trans_mat * point_homogeneous;

        //std::cout << "World Coordinates (X, Y, Z): "
        //    << world_point.at<float>(0) << ", "
        //    << world_point.at<float>(1) << ", "
        //    << world_point.at<float>(2) << std::endl;
        pt3.x = world_point.at<float>(0);
        pt3.y = world_point.at<float>(1);
        pt3.z = world_point.at<float>(2);
        return pt3;
    }

    vector<Point3f> cvtPoint(const vector<Point2f>& left, const vector<Point2f>& right)const
    {
        vector<Point3f> pts;
        for (int i = 0; i < left.size() && i < right.size(); ++i)
            pts.push_back(cvtPoint(left[i], right[i]));
        return pts;
    }
private:

    Mat CM1, D1, R1, P1;
    Mat CM2, D2, R2, P2;
    Mat Q;
    Mat trans_mat;
};



inline Mat calcDiffGray(const Mat& lastImg, const Mat& img, int binThresh)
{
    Mat imgDiff;
    cv::subtract(img, lastImg, imgDiff);
    cv::threshold(imgDiff, imgDiff, binThresh, 255, cv::THRESH_TOZERO);

    Mat imgDiffGray, imgGray;
    cv::cvtColor(imgDiff, imgDiffGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    Mat imgGray2 = imgGray + imgDiffGray;

    Mat dstImg;
    cv::merge(vector<Mat>{ imgDiffGray, imgGray, imgGray2 }, dstImg);
    return dstImg;
}



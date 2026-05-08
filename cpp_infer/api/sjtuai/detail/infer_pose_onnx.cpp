/*
 * @file	
 * @brief	
 * @author	sjtu 3-122b 
 * @date	2025
 * @copyright	All rights reserved
 * @details	
*/

#include "infer.h"
#include "Yolos/yolos.hpp"
#include <_auto_inc/OpenCV.h>
/************************************************************************/
/*                                                                      
                                                                        */
/************************************************************************/

struct InferPoseYoloOcv : InferPose
{
    typedef yolos::pose::YOLOPoseDetector Alg;
    std::shared_ptr<Alg> alg;
    float score_threshold = 0.25;
    float iou_threshold = 0.45;
    int num_points = 21;
    int dim_point = 2;
    bool is_gpu = true;

    bool init(const string& fModel)
    {
        cout << "===球拍姿态模型初始化:" << fModel << endl;
        alg.reset(new Alg(fModel, "", is_gpu));

        alg->FEATURES_PER_KEYPOINT = dim_point;
        alg->NUM_KEYPOINTS = num_points;

        return alg.get();
    }

    vector<vector<Point2f>> run(const Mat& img) override
    {
        vector<vector<Point2f>> ptsList;

        MyTimer t;
        auto output = alg->detect(img, score_threshold, iou_threshold);
        auto useTime = t.elapse();
        cout << "pose bat, size:" << output.size() << ", time:" << useTime << "(ms)" << endl;
        std::sort(output.begin(), output.end(), [](const auto& p1, const auto& p2) { return p1.conf > p2.conf; });

        for (auto& ot : output)
        {
            vector<Point2f> pts;
            for (auto& pt : ot.keypoints)
                pts.emplace_back(pt.x, pt.y);
            ptsList.emplace_back(pts);
        }
        return ptsList;
    }
};

shared_ptr<InferPose> infer_pose_create_ocv_rt(const string& model)
{
    auto alg = make_shared<InferPoseYoloOcv>();
    alg->init(model);
    return alg;
}
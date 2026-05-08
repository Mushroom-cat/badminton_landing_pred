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

struct InferDetYoloOcv : InferDet
{
    typedef yolos::det::YOLODetector Alg;
    std::shared_ptr<Alg> alg;
    float score_threshold = 0.25;
    float iou_threshold = 0.45;
    bool is_gpu = true;

    bool init(const string& fModel)
    {
        cout << "===羽球检测模型初始化:" << fModel << endl;
        alg.reset(new Alg(fModel, "", is_gpu));
        return alg.get();
    }

    vector<Rect> run(const Mat& img) override
    {
        vector<Rect> rs;

        MyTimer t;
        auto output = alg->detect(img, score_threshold, iou_threshold);
        auto useTime = t.elapse();
        cout << "det ball, size:" << output.size() << ", time:" << useTime << "(ms)" << endl;
        std::sort(output.begin(), output.end(), [](const auto& p1, const auto& p2) { return p1.conf > p2.conf; });
        for (auto& ot : output)
        {
            Rect r = {ot.box.x, ot.box.y, ot.box.width, ot.box.height};
            rs.emplace_back(r);
        }

        return rs;
    }
};

InferDetPtr infer_det_create_ocv_rt(const string& model)    //目标检测  OpenCV   后端推理
{
    auto alg = make_shared<InferDetYoloOcv>();
    alg->init(model);
    return alg;
}
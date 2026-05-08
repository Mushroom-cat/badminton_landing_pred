/*
 * @file	
 * @brief	
 * @author	Liu Kuan
 * @date	2025
 * @copyright	All rights reserved
 * @details	
*/
#include <filesystem>
#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>
#include <_auto_inc/onnxruntime.h>
#include "infer.h"
/************************************************************************/
/*                                                                      
                                                                        */
/************************************************************************/
using namespace std;

static vector<unsigned char> file_read_bin(const string& fileName)
{
    vector<unsigned char> data;
    if (std::filesystem::exists(fileName))
    {
        std::streamsize fz = std::filesystem::file_size(fileName);
        data.resize(fz);
        std::ifstream file(fileName, std::ios::binary);
        file.read((char*)&data[0], fz);
    }
    return data;
}


static string s_fmt_shape(const std::vector<long long>& v)
{
    stringstream ss;
    ss << "(";
    for (size_t i = 0; i < v.size(); i++)
    {
        ss << v[i];
        if (i != v.size() - 1)
            ss << ", ";
    }
    ss << ")";
    return ss.str();
}


static bool frame_is_empty(const vector<Point3f>& frame)
{
    if (frame.empty())
        return true;

    bool isAllZero = true;

    for (auto& pt : frame)
    {
        if (abs(pt.x) > 1e-6 || abs(pt.x) > 1e-6 || abs(pt.x) > 1e-6)
        {
            isAllZero = false;
            break;
        }
    }
    return isAllZero;
}

template<class T>
inline double product(const vector<T>& v) ///< 求乘积
{
    double total = 1;
    for (auto& i : v)total *= i;
    return total;
}


struct OrtRuner
{
    Ort::Env env = { ORT_LOGGING_LEVEL_ERROR, "klib" };
    Ort::Session* _session = nullptr;
    vector<Ort::Value> outValues;

    string modelFile;					//模型路径
    int numThread = -1;					//是否多线程
    int graphOptLevel = -1;	            //模型图优化
    int execMode = -1;				    //执行方式
    int provider = -1;					//运行的后台

    vector<float> inTenData1;		//输入数据1: float
    vector<uint8_t> inTenData2;        //输入数据2: bool
    vector<string> inNames;				//模型的名字,读取模型后会动态修改
    vector<string> outNames;			//模型输出的名字,读取模型后会动态修改
    vector<vector<int64_t>> inShapes;	//输入形状,读取模型后会动态修改
    vector<vector<int64_t>> outShapes;	//输出形状,读取模型后会动态修改

    bool init()
    {
        Ort::SessionOptions _opt;
        if (numThread > 0)  _opt.SetIntraOpNumThreads(numThread);
        if (graphOptLevel > -1)    _opt.SetGraphOptimizationLevel((GraphOptimizationLevel)graphOptLevel);
        if (execMode > -1) _opt.SetExecutionMode((ExecutionMode)execMode);

        switch (provider)
        {
        case 0: {
            OrtCUDAProviderOptions pOpt;
            _opt.AppendExecutionProvider_CUDA(pOpt);
        }break;
        case 1: {
            OrtOpenVINOProviderOptions pOpt;
            _opt.AppendExecutionProvider_OpenVINO(pOpt);
        }break;
        case 2: {
            OrtTensorRTProviderOptions pOpt;
            _opt.AppendExecutionProvider_TensorRT(pOpt);
        }break;
        case 3: {
            OrtMIGraphXProviderOptions pOpt;
            _opt.AppendExecutionProvider_MIGraphX(pOpt);
        }break;
        case 4: {
            OrtROCMProviderOptions pOpt;
            _opt.AppendExecutionProvider_ROCM(pOpt);
        }break;
        }

        auto data = file_read_bin(modelFile);
        _session = new Ort::Session(env, data.data(), data.size(), _opt);

        inNames.clear();
        inShapes.clear();
        outNames.clear();
        outShapes.clear();

        size_t numInputNodes = _session->GetInputCount();
        size_t numOutputNodes = _session->GetOutputCount();

        Ort::AllocatorWithDefaultOptions allocator;
        for (int i = 0; i < numInputNodes; i++)
        {
            auto inN = _session->GetInputNameAllocated(i, allocator);
            inNames.push_back(inN.get());
            Ort::TypeInfo input_type_info = _session->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();
            inShapes.push_back(input_dims);
        }

        for (int i = 0; i < numOutputNodes; i++)
        {
            auto outN = _session->GetOutputNameAllocated(i, allocator);
            outNames.push_back(outN.get());
            Ort::TypeInfo output_type_info = _session->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            auto output_dims = output_tensor_info.GetShape();
            outShapes.push_back(output_dims);
        }

        return true;
    }

    ~OrtRuner()
    {
        delete _session;
    }

    void print(std::ostream& ss = std::cout)
    {
        //print name/shape of inputs
        ss << "Input Node Name/Shape (" << inNames.size() << "):\n";
        for (size_t i = 0; i < inNames.size(); i++) {
            ss << "\t" << inNames[i] << " : " << s_fmt_shape(inShapes[i]) << "\n";
        }

        // print name/shape of outputs
        ss << "Output Node Name/Shape (" << outNames.size() << "):\n";
        for (size_t i = 0; i < outNames.size(); i++) {
            ss << "\t" << outNames[i] << " : " << s_fmt_shape(outShapes[i]) << "\n";
        }
    }

    Point3f run()
    {
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        vector<Ort::Value> input_tensor_;

        input_tensor_.push_back(Ort::Value::CreateTensor<float>(allocator_info,
            inTenData1.data(), inTenData1.size(),
            inShapes[0].data(), inShapes[0].size()));

        input_tensor_.push_back(Ort::Value::CreateTensor<bool>(allocator_info,
            (bool*)inTenData2.data(), inTenData2.size(),
            inShapes[1].data(), inShapes[1].size()));

        std::vector<const char*> input_names_(inNames.size(), nullptr);
        for (int i = 0; i < inNames.size(); ++i)
        {
            input_names_[i] = inNames[i].c_str();
            //cout << "in [" << i << "]" << input_names_[i] << endl;
        }


        std::vector<const char*> output_names_(outNames.size());
        for (int i = 0; i < outNames.size(); ++i)
        {
            output_names_[i] = outNames[i].c_str();
            //cout << "out [" << i << "]" << output_names_[i] << endl;
        }


        outValues = _session->Run(Ort::RunOptions{ nullptr }, input_names_.data(),
            input_tensor_.data(), input_names_.size(),
            output_names_.data(), output_names_.size());

        auto xyz = outValues.at(0).GetTensorMutableData<double>();
        auto var = outValues.at(1).GetTensorMutableData<double>();
        auto time = outValues.at(2).GetTensorMutableData<double>();
        auto direction = outValues.at(3).GetTensorMutableData<float>();
        //cout << "====xyz(" << xyz[0] << "," << xyz[1] << "," << xyz[2] << ")";
        //cout << ", var(" << var[0] << "," << var[1] << "," << var[2] << ")";
        //cout << ", time(" << time[0] << ")";
        //cout << ", direction(" << direction[0] << "," << direction[1] << ")" << endl;
        return Point3f(xyz[0], xyz[1], xyz[2]);
    }
};



struct InferLandPointOnnxRt : InferLandPoint
{
    std::shared_ptr<OrtRuner> runner;
    int seqLen = 50;
    int featLen = 0;
    const int AFTER_LEN = 66;

    bool init(const string& fModel, bool isBefore = true)
    {
        cout << "===落点预测模型初始化:" << fModel << endl;
        this->featLen = isBefore ? 63 : 66;

        runner.reset(new OrtRuner);
        runner->modelFile = fModel;
        runner->init();
        runner->inShapes.at(0).at(0) = 1;
        runner->inShapes.at(0).at(1) = seqLen;
        runner->inShapes.at(0).at(2) = featLen;
        runner->inShapes.at(1).at(0) = 1;
        runner->inShapes.at(1).at(1) = seqLen;

        runner->print();
        return true;
    }

    bool setBeforeData(const vector<vector<Point3f>>& frmPsList)
    {
        cout << "run before, frame size =" << frmPsList.size() << endl;

        auto& pointData = runner->inTenData1;
        pointData.resize(product(runner->inShapes.at(0)));

        auto& maskData = runner->inTenData2;
        maskData.resize(product(runner->inShapes.at(1)));

        int j = 0;
        for (int idx = 0; idx < frmPsList.size(); ++idx)
        {
            auto& frame = frmPsList[idx];

            for (auto& pt : frame)
            {
                pointData[j++] = pt.x;
                pointData[j++] = pt.y;
                pointData[j++] = pt.z;
            }
            maskData[idx] = true;
        }

        return true;
    }

    bool setAfterData(const vector<vector<Point3f>>& frmPsList, const vector<Point3f>& ballList)
    {
        cout << "frame size =" << frmPsList.size() << ", ball size =" << ballList.size() << endl;

        auto& pointData = runner->inTenData1;
        pointData.resize(product(runner->inShapes.at(0)));
        cout << "point data size=" << pointData.size() << endl;

        auto& maskData = runner->inTenData2;
        maskData.resize(product(runner->inShapes.at(1)));
        cout << "mask data size=" << maskData.size() << endl;

        int j = 0;
        for (int idx = 0; idx < frmPsList.size(); ++idx)
        {
            auto& frame = frmPsList[idx];

            for (auto& pt : frame)
            {
                pointData[j++] = pt.x;
                pointData[j++] = pt.y;
                pointData[j++] = pt.z;
            }

            if (idx + 5 >= this->seqLen)
            {
                auto& pt = ballList[idx];
                pointData[j++] = pt.x;
                pointData[j++] = pt.y;
                pointData[j++] = pt.z;
            }
            else {
                pointData[j++] = 0;
                pointData[j++] = 0;
                pointData[j++] = 0;
            }

            //cout << "\n" << j << endl;
            maskData[idx] = true;// !frame_is_empty(frame);
        }
        return true;
    }

    Point3f run(const vector<vector<Point3f>>& framPoseList, const vector<Point3f>& ballList)override
    {
        if (featLen == AFTER_LEN)
        {
            setAfterData(framPoseList, ballList);
        }
        else {
            setBeforeData(framPoseList);
        }

        //打印 
        //auto& pointData = runner->inTenData1;
        //cout << "point data:" << endl;
        //for (int i = 0; i < seqLen; ++i)
        //{
        //    cout << i << ":  ";
        //    for (int j = 0; j < featLen; ++j)
        //    {
        //        cout << pointData[j + i * featLen];
        //        if (j != featLen - 1)
        //            cout << ",";
        //    }
        //    cout << "\n\n";
        //}

        MyTimer t;
        Point3f landPt = runner->run();
        auto useTime = t.elapse();
        cout << "land point infer time:" << useTime << "(ms)" << endl;

        landPt.z = 0;
        return landPt;
    }
};



shared_ptr<InferLandPoint> infer_landpoint_create_onnx_rt(const string& model, bool isBefore)
{
    auto alg = make_shared<InferLandPointOnnxRt>();
    alg->init(model, isBefore);
    return alg;
}
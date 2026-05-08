#pragma once

// ============================================================================
// YOLOs-CPP - Unified YOLO Inference Library
// ============================================================================
// Master include header for all YOLO tasks.
//
// Usage:
//   #include "yolos.hpp"  // Include all tasks
//   or include specific tasks:
//   #include "tasks/detection.hpp"
//   #include "tasks/segmentation.hpp"
//   #include "tasks/pose.hpp"
//   #include "tasks/obb.hpp"
//   #include "tasks/classification.hpp"
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

// Core components
#include "core/types.hpp"
#include "core/version.hpp"
#include "core/utils.hpp"
#include "core/preprocessing.hpp"
#include "core/nms.hpp"
#include "core/drawing.hpp"
#include "core/session_base.hpp"

// Task-specific implementations
#include "tasks/detection.hpp"
#include "tasks/segmentation.hpp"
#include "tasks/pose.hpp"
#include "tasks/obb.hpp"
#include "tasks/classification.hpp"

// ============================================================================
// Namespace Aliases for Convenience
// ============================================================================
namespace yolos {

// Detection task aliases
using Detection = det::Detection;
using YOLODetector = det::YOLODetector;
using YOLO26Detector = det::YOLO26Detector;

// Segmentation task aliases
using Segmentation = seg::Segmentation;
using YOLOSegDetector = seg::YOLOSegDetector;

// Pose estimation task aliases
using PoseResult = pose::PoseResult;
using YOLOPoseDetector = pose::YOLOPoseDetector;

// OBB detection task aliases
using OBBResult = obb::OBBResult;
using YOLOOBBDetector = obb::YOLOOBBDetector;

// Classification task aliases
using ClassificationResult = cls::ClassificationResult;
using YOLOClassifier = cls::YOLOClassifier;
using YOLO26Classifier = cls::YOLO26Classifier;

} // namespace yolos

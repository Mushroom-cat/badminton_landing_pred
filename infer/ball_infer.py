#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2026/1/6 16:15:05

@author: LiuKuan
@copyright: Apache License, Version 2.0
'''
import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO
from pathlib import Path
import numpy as np

# ===============================================================================
'''
修改说明：
1. 适配YOLO检测框模型，不再处理姿态关键点，改为计算检测框中点作为关键点
2. 函数名和注释全部更新，移除pose相关表述，替换为box检测相关
3. 核心逻辑从解析keypoints改为解析boxes，计算每个框的中点坐标
4. 过滤逻辑改为基于检测框的置信度，而非关键点置信度
5. 保留批量处理、格式保存、统计信息等原有实用功能
'''
# ===============================================================================

def infer_box_and_save_json(fImg, fModel='../cfg/ball_det.pt', conf_thres=0.5, show_img=False):
    """
    执行单张图片的YOLO检测框模型推理，将每个检测框的中点作为关键点保存为JSON文件
    :param fImg: 输入图片路径
    :param fModel: YOLO检测框模型路径（如yolov8n.pt）
    :param conf_thres: 检测框置信度阈值，低于此值的框会被过滤
    :param show_img: 是否显示检测结果图片（批量处理时建议设为False）
    :return: 处理状态（True=成功，False=失败）
    """
    try:
        # 1. 加载YOLO检测框模型并执行检测
        model = YOLO(fModel)
        results = model.predict(fImg, verbose=False, conf=conf_thres)
        result = results[0]  # 获取第一张图片的检测结果

        # 2. 提取图片基础信息
        image_path = Path(fImg)
        image_name = image_path.name  # 例如：000682.jpg
        image_height, image_width = result.orig_shape  # 原始图片尺寸 (高, 宽)

        # 3. 构建指定格式的JSON基础结构
        json_data = {
            "version": "0.0.0",
            "flags": [],
            "shapes": [],
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
            "hit": False,
            "hard": False
        }

        # 4. 解析YOLO检测框结果（核心修改：计算检测框中点）
        if result.boxes is not None and len(result.boxes) > 0:
            # 提取检测框坐标和置信度：boxes.xyxy是[N, 4]数组（N个框，x1/y1/x2/y2坐标）
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # 转为numpy数组
            boxes_conf = result.boxes.conf.cpu().numpy()  # 检测框置信度 [N,]

            # 遍历所有检测框
            for box_idx in range(len(boxes_xyxy)):
                box_coords = boxes_xyxy[box_idx]  # 当前框的坐标 [x1, y1, x2, y2]
                box_conf = boxes_conf[box_idx]    # 当前框的置信度

                # 过滤低置信度检测框
                if box_conf >= conf_thres:
                    # 计算检测框中点坐标 (x_center, y_center)
                    x_center = (box_coords[0] + box_coords[2]) / 2.0
                    y_center = (box_coords[1] + box_coords[3]) / 2.0
                    # 保存中点坐标（转为float确保JSON序列化正常）
                    mid_point = [float(x_center), float(y_center)]

                    # 5. 构建shape对象（保持原有JSON格式）
                    shape = {
                        "label": "human2",  # 可根据实际需求修改标签名
                        "points": [mid_point],  # 中点作为关键点
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": [],
                        "mask": None
                    }
                    json_data["shapes"].append(shape)

        # 6. 保存JSON文件（与原图同目录，同名不同后缀）
        json_save_path = image_path.with_suffix('.json')
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # 7. 可选：显示检测结果（批量处理时建议关闭）
        if show_img:
            result.show()

        # 输出单张处理结果（更新统计维度为检测框数）
        box_count = len(json_data['shapes'])
        print(f"✅ 处理完成：{image_path} | 检测框数：{box_count} | 有效中点数：{box_count}")
        return True

    except Exception as e:
        print(f"❌ 处理失败：{fImg} | 错误信息：{str(e)}")
        return False


def batch_infer_box_from_folder(root_dir, fModel='../cfg/ball_det.pt', conf_thres=0.5, show_img=False):
    """
    批量处理指定文件夹及其所有子文件夹中的图片（检测框版本）
    :param root_dir: 根文件夹路径（会遍历所有子文件夹）
    :param fModel: YOLO检测框模型路径
    :param conf_thres: 检测框置信度阈值
    :param show_img: 是否显示每张图片的检测结果（批量处理建议False）
    :return: 处理统计（成功数、失败数、总数）
    """
    # 定义需要处理的图片格式（覆盖常见格式）
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    # 转换为Path对象，方便路径操作
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"❌ 根文件夹不存在：{root_dir}")
        return 0, 0, 0

    # 遍历根文件夹及其所有子文件夹的图片（递归遍历）
    img_files = []
    for ext in img_extensions:
        img_files.extend(root_path.rglob(f"*{ext.lower()}"))  # 匹配小写后缀
        img_files.extend(root_path.rglob(f"*{ext.upper()}"))  # 匹配大写后缀（如.JPG）

    # 去重（避免大小写后缀重复匹配）
    img_files = list(set(img_files))
    total_count = len(img_files)
    success_count = 0
    fail_count = 0

    if total_count == 0:
        print(f"⚠️  在 {root_dir} 及其子文件夹中未找到任何图片文件")
        return 0, 0, 0

    print(f"\n🚀 开始批量处理：共发现 {total_count} 张图片")
    print("-" * 80)

    # 逐个处理图片
    for img_file in img_files:
        if infer_box_and_save_json(str(img_file), fModel, conf_thres, show_img):
            success_count += 1
        else:
            fail_count += 1

    # 输出批量处理统计结果
    print("-" * 80)
    print(f"\n📊 批量处理完成 | 总数：{total_count} | 成功：{success_count} | 失败：{fail_count}")
    return success_count, fail_count, total_count


if __name__ == '__main__':
    # ====================== 批量处理配置（修改这里即可） ======================
    root_folder = "./"  # 待处理的根文件夹（会遍历所有子文件夹）
    model_path = "../cfg/ball_det.pt"  # YOLO检测框模型路径
    conf_threshold = 0.3  # 检测框置信度阈值
    show_image = False  # 批量处理时建议设为False（避免弹窗）

    # 执行批量处理
    batch_infer_box_from_folder(
        root_dir=root_folder,
        fModel=model_path,
        conf_thres=conf_threshold,
        show_img=show_image
    )
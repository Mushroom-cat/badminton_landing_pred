#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2026/1/6 16:15:05

@author: LiuKuan
@copyright: Apache License, Version 2.0
'''
import laok as lk
import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO
from pathlib import Path
import numpy as np

# ===============================================================================
'''
修改说明：
1. 新增批量处理函数，支持遍历指定文件夹及其所有子文件夹的图片
2. 自动匹配常见图片格式（jpg/jpeg/png/bmp等），无需手动指定单张图片
3. 保留单张图片处理的核心逻辑，批量处理时复用该逻辑
4. 批量模式下默认关闭图片可视化（避免弹窗刷屏），可通过参数开启
5. 增加批量处理统计信息，方便查看处理进度和结果
6. 核心修改：只保留检测到的最上面的一个人体（按y坐标最小判断，y越小越靠上）
'''


# ===============================================================================

def infer_pose_and_save_json(fImg, fModel='../cfg/bat_pose.pt', conf_thres=0.5, show_img=False):
    """
    执行单张图片的YOLO-Pose关键点检测，并将关键点保存为指定格式的JSON文件
    核心修改：仅保留检测到的最上面的一个人体
    :param fImg: 输入图片路径
    :param fModel: YOLO-Pose模型路径（需为pose模型，如yolov8n-pose.pt）
    :param conf_thres: 关键点置信度阈值，低于此值的关键点会被过滤
    :param show_img: 是否显示检测结果图片（批量处理时建议设为False）
    :return: 处理状态（True=成功，False=失败）
    """
    try:
        # 1. 加载YOLO-Pose模型并执行关键点检测
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

        # 4. 解析YOLO-Pose关键点结果（核心逻辑：只保留最上面的人体）
        if result.keypoints is not None and len(result.keypoints) > 0:
            # 提取关键点坐标和置信度：keypoints.xy是[N, K, 2]数组（N个人，K个关键点，x/y坐标）
            keypoints_xy = result.keypoints.xy.cpu().numpy()  # 转为numpy数组，方便处理
            keypoints_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

            # ========== 核心修改部分 start ==========
            # 步骤1：计算每个人体的顶部位置（最小y坐标，y越小越靠上）
            person_top_y = []  # 存储每个人体的顶部y坐标
            valid_person_indices = []  # 存储有效人体的索引
            for idx, person_kp in enumerate(keypoints_xy):
                # 过滤掉坐标为0的无效关键点，只计算有效点的最小y值
                valid_kp = person_kp[(person_kp[:, 0] > 0) & (person_kp[:, 1] > 0)]
                if len(valid_kp) > 0:
                    top_y = np.min(valid_kp[:, 1])  # 最小y坐标=最顶部的位置
                    person_top_y.append(top_y)
                    valid_person_indices.append(idx)

            # 步骤2：找到最上面的人体（最小top_y对应的索引）
            if len(valid_person_indices) > 0:
                min_top_y_idx = np.argmin(person_top_y)  # 最小y值的索引
                target_person_idx = valid_person_indices[min_top_y_idx]  # 最上面人体的原始索引

                # 步骤3：只处理这个最上面的人体
                person_keypoints = keypoints_xy[target_person_idx]  # 最上面人体的所有关键点 (K, 2)
                person_conf = keypoints_conf[target_person_idx] if keypoints_conf is not None else np.ones_like(
                    person_keypoints[:, 0])

                # 过滤低置信度关键点，只保留置信度≥阈值的
                valid_points = []
                for kp_coord, kp_c in zip(person_keypoints, person_conf):
                    if kp_c >= conf_thres:  # 仅保留高置信度关键点
                        valid_points.append([float(kp_coord[0]), float(kp_coord[1])])

                # 5. 构建shape对象（严格匹配指定格式）
                shape = {
                    "label": "human2",  # 固定为示例中的"human2"
                    "points": valid_points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": [],
                    "mask": None
                }
                json_data["shapes"].append(shape)
            # ========== 核心修改部分 end ==========

        # 6. 保存JSON文件（与原图同目录，同名不同后缀）
        json_save_path = image_path.with_suffix('.json')
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # 7. 可选：显示检测结果（批量处理时建议关闭）
        if show_img:
            result.show()

        # 输出单张处理结果
        person_count = len(json_data['shapes'])
        kp_count = len(json_data['shapes'][0]['points']) if json_data['shapes'] else 0
        print(f"✅ 处理完成：{image_path} | 人体数：{person_count} | 有效关键点数：{kp_count}")
        return True

    except Exception as e:
        print(f"❌ 处理失败：{fImg} | 错误信息：{str(e)}")
        return False


def batch_infer_pose_from_folder(root_dir, fModel='../cfg/bat_pose.pt', conf_thres=0.5, show_img=False):
    """
    批量处理指定文件夹及其所有子文件夹中的图片
    :param root_dir: 根文件夹路径（会遍历所有子文件夹）
    :param fModel: YOLO-Pose模型路径
    :param conf_thres: 关键点置信度阈值
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
        if infer_pose_and_save_json(str(img_file), fModel, conf_thres, show_img):
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
    model_path = "../cfg/bat_pose.pt"  # YOLO-Pose模型路径
    conf_threshold = 0.3  # 关键点置信度阈值
    show_image = False  # 批量处理时建议设为False（避免弹窗）

    # 执行批量处理
    batch_infer_pose_from_folder(
        root_dir=root_folder,
        fModel=model_path,
        conf_thres=conf_threshold,
        show_img=show_image
    )
#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2026/1/6 16:19:02

@author: LiuKuan
@copyright: Apache License, Version 2.0
'''
import laok
from util import CamCalib
from laok.ext import lxml_
import laok as lk
import os  # 用于路径/文件名处理

# ===============================================================================
r'''
命令行例子:

支持文件夹中json:
    python .\cvt_2d_3d.py cvt_dir_json .\data\bat_2d_to_3d 3d_bat

文件夹中xml:
    python .\cvt_2d_3d.py cvt_dir_xml .\data\ball_2d_to_3d 3d_ball

支持ball文件    
    python .\cvt_2d_3d.py cvt_file .\data\ball_2d_to_3d\left\000769.xml

支持json文件
    python .\cvt_2d_3d.py cvt_file .\data\bat_2d_to_3d\left\000680.json

转换点:
    python .\cvt_2d_3d.py cvt_point [1,2] [3,4]
'''


# ===============================================================================

def cvt_dir_json(dirSrc, dirDst, fIntrinsic="./cfg/20250809_intrinsic.yml",
                 fExtrinsic="./cfg/20250809_extrinsic.yml", max_lines=105):
    """
    转换文件夹中的JSON文件，汇总结果到txt文件（最多105行/文件）
    :param dirSrc: 源JSON文件夹路径
    :param dirDst: 结果保存文件夹路径
    :param fIntrinsic: 内参文件路径
    :param fExtrinsic: 外参文件路径
    :param max_lines: 单个txt文件最大行数（默认105）
    """
    dirSrc1 = laok.path_abs(dirSrc)
    dirDst1 = laok.path_abs(dirDst)
    laok.path_make(dirDst1)  # 确保目标文件夹存在

    cam = CamCalib()
    cam.setIntrinsicFile(fIntrinsic)
    cam.setExtrinsicFile(fExtrinsic)

    # 初始化汇总变量
    line_count = 0  # 当前文件已写入行数
    file_index = 1  # 输出文件序号（如output_1.txt）
    current_file_path = os.path.join(dirDst1, f"output_{file_index}.txt")
    current_file = open(current_file_path, 'w', encoding='utf-8')  # 打开第一个文件

    # 遍历所有JSON文件
    for fjson in lk.files_under(dirSrc, '.json'):
        # 只处理left文件夹下的文件
        if 'left' not in fjson:
            continue

        # 1. 提取无后缀文件名
        file_name_no_ext = os.path.splitext(os.path.basename(fjson))[0]
        # 2. 转换3D坐标
        pt3List = _cvt_json(fjson, cam)
        # 3. 拼接该行内容（文件名: 坐标点）
        points_content = ', '.join([','.join(str(p) for p in pt3) for pt3 in pt3List])
        line_content = f"{file_name_no_ext}: {points_content}\n"

        # 4. 判断是否需要新建文件（超过max_lines行）
        if line_count >= max_lines:
            current_file.close()  # 关闭当前文件
            file_index += 1  # 序号+1
            current_file_path = os.path.join(dirDst1, f"output_{file_index}.txt")
            current_file = open(current_file_path, 'w', encoding='utf-8')  # 新建文件
            line_count = 0  # 重置行数

        # 5. 写入当前行
        current_file.write(line_content)
        line_count += 1

    # 关闭最后一个文件
    current_file.close()
    print(f"✅ JSON转换完成！结果保存至: {dirDst1}")
    print(f"   生成文件数: {file_index}，最后一个文件行数: {line_count}")


def cvt_dir_xml(dirSrc, dirDst, fIntrinsic="./cfg/20250809_intrinsic.yml",
                fExtrinsic="./cfg/20250809_extrinsic.yml", max_lines=105):
    """
    转换文件夹中的XML文件，汇总结果到txt文件（最多105行/文件）
    :param dirSrc: 源XML文件夹路径
    :param dirDst: 结果保存文件夹路径
    :param fIntrinsic: 内参文件路径
    :param fExtrinsic: 外参文件路径
    :param max_lines: 单个txt文件最大行数（默认105）
    """
    dirSrc1 = laok.path_abs(dirSrc)
    dirDst1 = laok.path_abs(dirDst)
    laok.path_make(dirDst1)  # 确保目标文件夹存在

    cam = CamCalib()
    cam.setIntrinsicFile(fIntrinsic)
    cam.setExtrinsicFile(fExtrinsic)

    # 初始化汇总变量
    line_count = 0
    file_index = 1
    current_file_path = os.path.join(dirDst1, f"output_ball_{file_index}.txt")
    current_file = open(current_file_path, 'w', encoding='utf-8')

    # 遍历所有XML文件
    for fXml in lk.files_under(dirSrc, '.xml'):
        if 'left' not in fXml:
            continue

        # 1. 提取无后缀文件名
        file_name_no_ext = os.path.splitext(os.path.basename(fXml))[0]
        # 2. 转换3D坐标
        pt3 = _cvt_xml(fXml, cam)
        # 3. 拼接该行内容
        points_content = ','.join(str(p) for p in pt3)
        line_content = f"{file_name_no_ext}: {points_content}\n"

        # 4. 判断是否新建文件
        if line_count >= max_lines:
            current_file.close()
            file_index += 1
            current_file_path = os.path.join(dirDst1, f"output_ball_{file_index}.txt")
            current_file = open(current_file_path, 'w', encoding='utf-8')
            line_count = 0

        # 5. 写入当前行
        current_file.write(line_content)
        line_count += 1

    # 关闭最后一个文件
    current_file.close()
    print(f"✅ XML转换完成！结果保存至: {dirDst1}")
    print(f"   生成文件数: {file_index}，最后一个文件行数: {line_count}")


def cvt_file(fSrc, fIntrinsic="./cfg/20250809_intrinsic.yml", fExtrinsic="./cfg/20250809_extrinsic.yml"):
    """处理单个文件（保持原有控制台打印逻辑）"""
    cam = CamCalib()
    cam.setIntrinsicFile(fIntrinsic)
    cam.setExtrinsicFile(fExtrinsic)

    # 提取无后缀的文件名
    file_name_no_ext = os.path.splitext(os.path.basename(fSrc))[0]

    if fSrc.endswith('.xml'):
        pt3 = _cvt_xml(fSrc, cam)
        print(f"{file_name_no_ext}: {','.join(str(p) for p in pt3)}")
    elif fSrc.endswith('.json'):
        pt3List = _cvt_json(fSrc, cam)
        points_content = ', '.join([','.join(str(p) for p in pt3) for pt3 in pt3List])
        print(f"{file_name_no_ext}: {points_content}")


def cvt_point(ptLeft, ptRight, fIntrinsic="./cfg/20250809_intrinsic.yml", fExtrinsic="./cfg/20250809_extrinsic.yml"):
    """转换单个点（保持原有逻辑）"""
    cam = CamCalib()
    cam.setIntrinsicFile(fIntrinsic)
    cam.setExtrinsicFile(fExtrinsic)
    pt3 = cam.cvtPoint(ptLeft, ptRight)
    print(pt3)


def _read_xml(xml_file):
    """读取XML文件（保持原有逻辑）"""
    vocs = {}
    root = lxml_.load_xml(xml_file)
    size = root.find('size')
    w = size.find('width').text
    h = size.find('height').text
    vocs['width'] = int(w)
    vocs['height'] = int(h)
    vocs['filename'] = root.find('filename').text
    obj_arr = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin, xmax, ymin, ymax = bndbox.find('xmin').text, bndbox.find('xmax').text, \
            bndbox.find('ymin').text, bndbox.find('ymax').text
        obj_arr.append({
            'name': obj.find('name').text,
            'xmin': float(xmin),
            'xmax': float(xmax),
            'ymin': float(ymin),
            'ymax': float(ymax),
        })
    vocs['object'] = obj_arr
    return vocs


def _read_json(json_file):
    """读取JSON文件（保持原有逻辑）"""
    import json
    with open(json_file) as f:
        jdata = json.load(f)
    return jdata


def _cvt_xml(fSrc, cam):
    """XML转3D坐标（保持原有逻辑）"""
    fLeft, fRight = None, None
    if 'left' in fSrc: fLeft = fSrc
    if 'right' in fSrc: fRight = fSrc
    if not fLeft: fLeft = fRight.replace('right', 'left')
    if not fRight: fRight = fLeft.replace('left', 'right')

    objLeft = _read_xml(fLeft)['object'][0]
    objRight = _read_xml(fRight)['object'][0]
    ptLeft = [(objLeft['xmin'] + objLeft['xmax']) / 2, (objLeft['ymin'] + objLeft['ymax']) / 2]
    ptRight = [(objRight['xmin'] + objRight['xmax']) / 2, (objRight['ymin'] + objRight['ymax']) / 2]
    pt3 = cam.cvtPoint(ptLeft, ptRight)
    return pt3


def _cvt_json(fSrc, cam):
    """JSON转3D坐标（保持原有逻辑）"""
    fLeft, fRight = None, None
    if 'left' in fSrc: fLeft = fSrc
    if 'right' in fSrc: fRight = fSrc
    if not fLeft: fLeft = fRight.replace('right', 'left')
    if not fRight: fRight = fLeft.replace('left', 'right')

    objLeft = _read_json(fLeft)['shapes'][0]['points']
    objRight = _read_json(fRight)['shapes'][0]['points']

    pt3List = []
    for ptLeft, ptRight in zip(objLeft, objRight):
        pt3 = cam.cvtPoint(ptLeft, ptRight)
        pt3List.append(pt3)
    return pt3List


if __name__ == '__main__':
    # import fire as _fire
    # _fire.Fire({
    #     'cvt_dir_json': cvt_dir_json,
    #     'cvt_dir_xml': cvt_dir_xml,
    #     'cvt_file':cvt_file,
    #     'cvt_point':cvt_point
    # })
    # ====================== 测试配置 ======================
    src_dir = "./Infered_2D_Points_Racket/20250809_143033"  # JSON文件所在的源文件夹
    dst_dir = "./Infered_3D_Points_Racket/test"  # 汇总结果保存文件夹
    max_lines_per_file = 105  # 单个文件最大行数

    # 调用JSON文件夹汇总转换函数
    cvt_dir_json(src_dir, dst_dir, max_lines=max_lines_per_file)

    # 如需处理XML文件夹，取消下面注释
    # src_xml_dir = "./data/ball_2d_to_3d"
    # cvt_dir_xml(src_xml_dir, dst_dir, max_lines=max_lines_per_file)

    # 方式2：处理单个JSON文件（如需使用，取消下面的注释并修改文件路径）
    # single_json_file = r".\data\bat_2d_to_3d\left\000680.json"
    # cvt_file(single_json_file)
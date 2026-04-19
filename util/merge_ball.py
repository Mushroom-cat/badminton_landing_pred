import os

def merge_coordinate_files(folder1, folder2, output_folder):
    """
    folder1: 原始文件夹 (包含 21 点/63 个数字的行)
    folder2: 待追加文件夹 (包含 1 点/3 个数字的行)
    output_folder: 结果保存路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files1 = [f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]

    for filename in files1:
        base_name = os.path.splitext(filename)[0]

        # 1. 预读文件 2 的坐标到字典
        coord_map = {}
        file2_candidates = [f for f in os.listdir(folder2) if os.path.splitext(f)[0] == base_name]
        if file2_candidates:
            file2_path = os.path.join(folder2, file2_candidates[0])
            with open(file2_path, 'r', encoding='utf-8') as f2:
                for line in f2:
                    if ':' in line:
                        parts = line.strip().split(':')
                        coord_map[parts[0].strip()] = parts[1].strip()

        # 2. 读取文件 1 并处理
        file1_path = os.path.join(folder1, filename)
        processed_lines = []

        with open(file1_path, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            total_lines = len(lines)

            for index, line in enumerate(lines):
                line = line.strip()
                if not line or ':' not in line:
                    continue

                # 判断是否为最后一行
                # index 是从 0 开始的，所以最后一行是 total_lines - 1
                if index == total_lines - 1:
                    # 最后一行特殊处理：直接原样加入，不追加任何东西
                    processed_lines.append(line)
                else:
                    # 非最后一行：执行合并逻辑
                    frame_id, data = line.split(':')
                    frame_id = frame_id.strip()
                    data = data.strip()

                    # 查找对应坐标，若无则补 0.0,0.0,0.0
                    extra_coords = coord_map.get(frame_id, "0.0,0.0,0.0")

                    # 合并
                    new_line = f"{frame_id}:{data},{extra_coords}"
                    processed_lines.append(new_line)

        # 3. 写入新文件
        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write('\n'.join(processed_lines) + '\n')

        print(f"处理完成: {filename}")


# --- 配置路径 ---
# 请修改为你的实际文件夹路径
dir1 = "./20260202_ds_3d_pose"  # 包含21个点的文件夹
dir2 = "./20260202_ds_3d_det_fix5"  # 包含1个点的文件夹
out_dir = "./20260202_all"  # 结果保存文件夹

merge_coordinate_files(dir1, dir2, out_dir)

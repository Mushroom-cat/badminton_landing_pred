import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def rename_ball_to_txt(folder_path):
    """
    将指定文件夹内所有.ball后缀的文件重命名为.txt后缀
    :param folder_path: 目标文件夹的路径
    """
    # 检查文件夹路径是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在，请检查路径是否正确。")
        return

    # 统计重命名成功的文件数
    rename_count = 0
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        file_full_path = os.path.join(folder_path, filename)
        # 跳过目录（只处理文件）
        if os.path.isdir(file_full_path):
            continue

        # 分割文件名和后缀（例如：test.ball 会被分成 ('test', '.ball')）
        file_name, file_ext = os.path.splitext(filename)

        # 判断后缀是否为.ball（忽略大小写）
        if file_ext.lower() == '.ball':
            # 构建新的文件名（原文件名 + .txt）
            new_filename = f"{file_name}.txt"
            new_file_full_path = os.path.join(folder_path, new_filename)

            try:
                # 执行重命名操作
                os.rename(file_full_path, new_file_full_path)
                print(f"成功重命名：{filename} -> {new_filename}")
                rename_count += 1
            except Exception as e:
                print(f"重命名失败：{filename}，原因：{str(e)}")

    print(f"\n重命名操作完成，共成功处理 {rename_count} 个.ball文件\n")


def merge_ball_data_to_label(label_folder, ball_folder, output_folder):
    """
    将label文件夹和ball文件夹中的对应txt文件合并内容，保存到输出文件夹
    :param label_folder: label文件夹路径
    :param ball_folder: 处理后的ball文件夹路径（已完成重命名）
    :param output_folder: 合并后文件的输出路径
    """
    # 检查原文件夹是否存在
    if not os.path.exists(label_folder):
        print(f"错误：文件夹 {label_folder} 不存在！")
        return
    if not os.path.exists(ball_folder):
        print(f"错误：文件夹 {ball_folder} 不存在！")
        return

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    print(f"输出文件夹已准备好：{output_folder}")

    # 获取label_folder中的所有txt文件
    txt_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]
    if not txt_files:
        print("提示：label文件夹中没有找到txt文件！")
        return

    # 统计处理成功的文件数
    merge_count = 0
    # 遍历每个txt文件
    for file_name in txt_files:
        try:
            # 拼接文件完整路径
            label_file_path = os.path.join(label_folder, file_name)
            ball_file_path = os.path.join(ball_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            # 检查ball文件夹中是否有对应文件
            if not os.path.exists(ball_file_path):
                print(f"警告：{ball_folder} 中未找到文件 {file_name}，跳过该文件！")
                continue

            # 读取ball文件数据，存储为字典
            ball_data = {}
            with open(ball_file_path, "r", encoding="utf-8") as f_ball:
                for line in f_ball:
                    line = line.strip()  # 去除换行符和空格
                    if not line:  # 跳过空行
                        continue
                    # 拆分帧序号和三维坐标（按冒号分割，只分割一次）
                    frame_part, coord_part = line.split(":", 1)
                    ball_data[frame_part.strip()] = coord_part.strip()

            # 读取并处理label文件数据
            with open(label_file_path, "r", encoding="utf-8") as f_label:
                lines = [line.strip() for line in f_label if line.strip()]  # 读取所有非空行

            if not lines:
                print(f"警告：{label_file_path} 为空文件，跳过！")
                continue

            processed_lines = []
            # 处理前n-1行（除最后一行外的所有行）
            for line in lines[:-1]:
                # 拆分帧序号和原有内容（按冒号分割，只分割一次）
                if ":" not in line:
                    print(f"警告：{label_file_path} 中行 '{line}' 格式错误，跳过该行！")
                    processed_lines.append(line)
                    continue
                frame_part, content_part = line.split(":", 1)
                frame_part = frame_part.strip()
                # 获取对应的三维坐标，若不存在则用0,0,0填充
                coord = ball_data.get(frame_part, "0,0,0")
                # 拼接新行：帧序号:原有内容,三维坐标
                new_line = f"{frame_part}:{content_part.strip()},{coord}"
                processed_lines.append(new_line)
            # 保留最后一行（落点坐标）
            processed_lines.append(lines[-1])
            # processed_lines.append("001151:92.789,-191.782,-3.46435")

            # 将处理后的数据写入新文件夹
            with open(output_file_path, "w", encoding="utf-8") as f_output:
                f_output.write("\n".join(processed_lines))

            print(f"成功处理并保存：{file_name} -> {output_folder}")
            merge_count += 1

        except Exception as e:
            print(f"处理文件 {file_name} 时出错：{str(e)}，跳过该文件！")

    print(f"\n文件合并操作完成，共成功处理 {merge_count} 个文件！")


# -------------------------- 统一配置区（只需修改这里） --------------------------
# 请根据实际路径修改以下配置
LABEL_FOLDER = "20260418_label_ds_3d_pose"  # label文件夹路径
BALL_FOLDER = "20260418_label_ds_3d_det"  # ball文件夹路径（需要重命名的文件夹）
OUTPUT_FOLDER = "./scene3"  # 合并后文件的输出路径
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # 第一步：执行.ball文件重命名为.txt
    print("===== 开始执行文件重命名操作 =====")
    rename_ball_to_txt(BALL_FOLDER)

    # 第二步：执行文件合并操作
    print("===== 开始执行文件合并操作 =====")
    merge_ball_data_to_label(LABEL_FOLDER, BALL_FOLDER, OUTPUT_FOLDER)

    print("\n===== 所有操作全部完成！=====")
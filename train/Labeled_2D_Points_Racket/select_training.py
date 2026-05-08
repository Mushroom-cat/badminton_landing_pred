import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import re


def extract_numeric_part(filename):
    """
    从文件名中提取数字部分（如000678.json提取出678）
    若文件名不是数字开头，返回None

    Args:
        filename: 文件名（如"000678.json"）

    Returns:
        int/None: 提取的数字，非数字命名返回None
    """
    # 匹配以数字开头的文件名（支持000678.json、678.json等格式）
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def find_json_with_non_17_points(folder_path, output_file):
    """
    查找指定文件夹下points数量不等于17的JSON文件，并输出到txt文件（标注实际点数）
    按路径+文件名排序，数字命名文件不连续时用空行分隔

    Args:
        folder_path: 要遍历的根文件夹路径
        output_file: 输出结果的txt文件路径
    """
    # 初始化结果列表，存储(文件路径, 文件名, 数字部分, 点数)元组
    result = []

    # 遍历文件夹及其子目录
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 只处理json文件
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    # 读取并解析JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 检查shapes数组中的每个shape的points长度
                    for shape in data.get('shapes', []):
                        points = shape.get('points', [])
                        points_count = len(points)

                        # 如果点数不等于17，记录文件信息
                        if points_count != 17:
                            full_file_path = os.path.abspath(file_path)
                            numeric_part = extract_numeric_part(file)
                            result.append((full_file_path, file, numeric_part, points_count))
                            # 一个文件只要有一个shape符合条件就记录，跳出循环
                            break

                except json.JSONDecodeError:
                    print(f"警告：文件 {file_path} 不是有效的JSON格式，已跳过")
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错：{str(e)}")

    # 按路径+文件名排序（先按文件路径，再按文件名）
    result.sort(key=lambda x: (x[0], x[1]))

    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 所有points数量不等于17的JSON文件 ===\n")
        f.write("格式：文件路径 | 实际点数\n")
        f.write("----------------------------------------\n")

        if result:
            prev_numeric = None  # 记录上一个文件的数字部分
            for idx, (file_path, filename, numeric_part, count) in enumerate(result):
                # 判断是否需要插入空行（仅针对数字命名的文件）
                if idx > 0 and numeric_part is not None and prev_numeric is not None:
                    # 数字不连续（差值大于1）时插入空行
                    if numeric_part - prev_numeric > 1:
                        f.write("\n")

                # 写入当前文件信息
                f.write(f"{file_path} | {count} 个点\n")

                # 更新上一个数字部分
                if numeric_part is not None:
                    prev_numeric = numeric_part
        else:
            f.write("无\n")

    print(f"处理完成！结果已保存到：{os.path.abspath(output_file)}")
    print(f"统计结果：")
    print(f"- 点数不等于17的文件总数：{len(result)}")
    # 统计各点数的分布
    if result:
        count_distribution = {}
        for _, _, _, cnt in result:
            count_distribution[cnt] = count_distribution.get(cnt, 0) + 1
        print("- 各点数分布：")
        for cnt, num_files in sorted(count_distribution.items()):
            print(f"  - {cnt}个点：{num_files}个文件")


# ==================== 配置参数 ====================
# 请修改为你的目标文件夹路径
TARGET_FOLDER = "../Labeled_2D_Points_Racket_All"
# 输出文件路径
OUTPUT_FILE = "non_17_points_result.txt"
# =================================================

# 执行主函数
if __name__ == "__main__":
    # 检查目标文件夹是否存在
    if not os.path.exists(TARGET_FOLDER):
        print(f"错误：目标文件夹 {TARGET_FOLDER} 不存在！")
    else:
        find_json_with_non_17_points(TARGET_FOLDER, OUTPUT_FILE)
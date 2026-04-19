import onnxruntime as ort
import numpy as np
import os

def inference_onnx(model_path, seq_data, mask_data):
    """
    model_path: .onnx 文件的路径
    seq_data: shape 为 (batch_size, seq_len, 63) 的 numpy 数组
    mask_data: shape 为 (batch_size, seq_len) 的 numpy 数组 (dtype=bool)
    """

    # 1. 创建推理会话
    session = ort.InferenceSession(model_path)

    # 2. 准备输入数据
    # 注意：ONNX 期待的数据类型必须与导出时一致
    # seq 通常是 float32，mask 是 bool
    input_seq = seq_data.astype(np.float32)
    input_mask = mask_data.astype(bool)

    # 3. 运行推理
    # input_names 必须与导出时的 ['seq', 'mask'] 一致
    onnx_inputs = {
        'seq': input_seq,
        'mask': input_mask
    }

    # run 的第一个参数为 None 表示获取所有输出，也可以指定 ['output']
    outputs = session.run(['output'], onnx_inputs)

    # outputs 是一个列表，获取第一个输出
    return outputs[0]


def batch_onnx_inference_and_eval(model_path, folder_path, mode='after', skip_n=0):
    """
    mode:
        'after'  -> 使用 (1,50,66) 输入
        'before' -> 使用 (1,50,63)，并跳过最后 skip_n 行

    skip_n:
        仅在 before 模式生效
    """

    # 1. 初始化 ONNX 推理会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else [
        'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        return f"模型加载失败: {e}"

    all_distances = []
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt') or '.' not in f]

    print(f"开始处理文件夹: {folder_path}，检测到 {len(file_list)} 个文件。")
    print(f"当前模式: {mode}, skip_n: {skip_n}")

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            # 最少需要 50 + skip_n + 1
            if len(lines) < 51 + skip_n:
                print(f"跳过文件 {filename}: 行数不足 {51 + skip_n}")
                continue

            # ========================
            # 1. Label（永远最后一行）
            # ========================
            label_line = lines[-1]
            label_coords = np.array(
                [float(x) for x in label_line.split(':')[1].split(',')],
                dtype=np.float32
            )

            # ========================
            # 2. 选择序列区间
            # ========================
            if mode == 'after':
                seq_lines = lines[-51:-1]  # 最后50帧

            elif mode == 'before':
                start = -(51 + skip_n)
                end = -(1 + skip_n)
                seq_lines = lines[start:end]

            else:
                raise ValueError(f"未知 mode: {mode}")

            # ========================
            # 3. 解析输入 (66维)
            # ========================
            seq_data = []
            for line in seq_lines:
                coords = [float(x) for x in line.split(':')[1].split(',')]
                seq_data.append(coords)

            input_seq = np.array(seq_data, dtype=np.float32).reshape(1, 50, 66)

            # ========================
            # 4. 根据模式处理输入
            # ========================
            if mode == 'before':
                input_seq = input_seq[:, :, :63]  # 裁剪到63维

            # mask 不变
            input_mask = np.ones((1, 50), dtype=bool)

            # ========================
            # 5. 推理
            # ========================
            onnx_inputs = {
                'seq': input_seq,
                'mask': input_mask
            }

            prediction = session.run(['xyz'], onnx_inputs)[0]
            pred_coords = prediction.flatten()

            # ========================
            # 6. 误差计算
            # ========================
            distance = np.linalg.norm(pred_coords - label_coords)
            all_distances.append(distance)
            #print(pred_coords)

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    if not all_distances:
        return "没有成功处理任何文件，无法计算平均距离。"

    avg_distance = np.mean(all_distances)
    return avg_distance


# --- 使用示例 ---

data_dir = "datasets/scene1+2"
# after 模式（直接66维）
model_path = "./models/ImprovedTransformerModel_20260420_004852.onnx"
result = batch_onnx_inference_and_eval(model_path, data_dir, mode='after')
print(f"所有文件的平均欧氏距离为: {result}")

# before 模式（用63维 + 往前偏移10帧）
model_path = "./models/ImprovedTransformerModel_20260420_004852.onnx"
result = batch_onnx_inference_and_eval(model_path, data_dir, mode='before', skip_n=5)
print(f"所有文件的平均欧氏距离为: {result}")


# 模拟输入数据（支持 dynamic_axes 定义的动态尺寸）
# batch_size = 1
# seq_len = 50
# dummy_seq_np = np.random.randn(batch_size, seq_len, 63).astype(np.float32)
# dummy_mask_np = np.ones((batch_size, seq_len), dtype=bool)
#
# # 执行
# result = inference_onnx(model_path, dummy_seq_np, dummy_mask_np)
#
# print(f"输出形状: {result.shape}")
# print(f"推理结果预览: \n{result}")
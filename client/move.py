import os
import shutil
import numpy as np

def process_nodes(source_root, target_root):
    # 创建目标目录及其子目录
    target_audio_dir = os.path.join(target_root, "audio")
    target_radar_dir = os.path.join(target_root, "radar")
    target_depth_dir = os.path.join(target_root, "depth")
    target_label_file = os.path.join(target_root, "label.npy")
    
    os.makedirs(target_audio_dir, exist_ok=True)
    os.makedirs(target_radar_dir, exist_ok=True)
    os.makedirs(target_depth_dir, exist_ok=True)

    # 初始化目标 label.npy，如果不存在
    if not os.path.exists(target_label_file):
        np.save(target_label_file, np.empty((0,), dtype=np.float32))

    # 加载目标目录的 label.npy
    target_label = np.load(target_label_file, allow_pickle=True)

    # 全局索引，用于重新编号
    global_index = 0

    # 处理每个 node 目录
    for node_index in range(16):
        node_dir = os.path.join(source_root, f"node_{node_index}")
        audio_dir = os.path.join(node_dir, "audio")
        radar_dir = os.path.join(node_dir, "radar")
        depth_dir = os.path.join(node_dir, "depth")
        label_file = os.path.join(node_dir, "label.npy")

        # 加载当前 node 的 label.npy
        if not os.path.exists(label_file):
            print(f"Warning: {label_file} does not exist. Skipping.")
            continue
        label = np.load(label_file, allow_pickle=True)

        # 获取当前目录中 npy 文件的总数
        n = len(label)
        if n < 30:
            print(f"Warning: Not enough files in {node_dir} to process 30 items. Skipping.")
            continue

        # 倒序取 30 个文件
        indices = range(n - 30, n)

        for index in indices:
            # 定义源文件路径
            audio_src = os.path.join(audio_dir, f"{index}.npy")
            radar_src = os.path.join(radar_dir, f"{index}.npy")
            depth_src = os.path.join(depth_dir, f"{index}.npy")

            # 定义目标文件路径，重新编号为 global_index
            audio_dest = os.path.join(target_audio_dir, f"{global_index}.npy")
            radar_dest = os.path.join(target_radar_dir, f"{global_index}.npy")
            depth_dest = os.path.join(target_depth_dir, f"{global_index}.npy")

            # 复制文件到目标目录
            shutil.copy2(audio_src, audio_dest)
            shutil.copy2(radar_src, radar_dest)
            shutil.copy2(depth_src, depth_dest)

            # 添加标签到目标 label.npy
            target_label = np.append(target_label, label[index])

            # 删除原文件
            os.remove(audio_src)
            os.remove(radar_src)
            os.remove(depth_src)

            # 更新全局索引
            global_index += 1

        # 删除原 label.npy 中的对应条目
        label = label[:n - 30]
        np.save(label_file, label)

    # 保存更新后的目标 label.npy
    np.save(target_label_file, target_label)

# 示例调用
source_root = "/home/chenxu/codes/AC/datasets"  # 替换为实际的源目录路径
target_root = "/home/chenxu/codes/ichibanFATE/server/test_datasets/AC"  # 替换为实际的目标目录路径
process_nodes(source_root, target_root)

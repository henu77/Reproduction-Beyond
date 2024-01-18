import os
import pickle
import numpy as np
import pandas as pd

path = "dataset/visual_echoes/mp3d_split_wise/val.pkl"
root_path = "dataset/mp3d"

index = 0
with open(path, 'rb') as f:

    data_dict = pickle.load(f)
    print(data_dict.keys())

for scene in data_dict.keys():
    print(scene)
    data = data_dict[scene]
    for data_key in data.keys():
        print(data_key[0], data_key[1])
        save_path = os.path.join(root_path, scene, str(data_key[1]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # print(save_path)
        data1 = data[data_key]
        rgb = data1['rgb']
        depth = data1['depth']
        print(rgb.shape, depth.shape)
        # 保存rgb和depth

        rgb_path = os.path.join(save_path, str(data_key[0]) + '.png')
        depth_path = os.path.join(save_path, str(data_key[0]) + '.npy')
        print(rgb_path, depth_path)
        # 保存rgb
        from PIL import Image

        img = Image.fromarray(rgb).convert('RGB')
        img.save(rgb_path)
        # 保存depth

        np.save(depth_path, depth)
        index += 1
print(f"总共保存了{index}个文件")

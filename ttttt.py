import numpy as np

def get_scene_list(path):
    with open(path) as f:
        scenes_test = f.readlines()
    scenes_test = [x.strip() for x in scenes_test]
    return scenes_test


print(get_scene_list('dataset/metadata/mp3d/mp3d_scenes_test.txt'))

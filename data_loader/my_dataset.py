import os.path
import re
import time
import librosa
import random
import math
import numpy as np
import glob
import torch
import pickle
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch.utils.data as data
from scipy.io.wavfile import read as read_wav
from scipy.signal import stft


def get_scene_list(path):
    with open(path) as f:
        scenes_test = f.readlines()
    scenes_test = [x.strip() for x in scenes_test]
    return scenes_test


def normalize(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples ** 2)))
    samples = samples * (desired_rms / rms)
    return samples


def generate_spectrogram(audioL, audioR, winl=32):
    channel_1_spec = librosa.stft(audioL, n_fft=512, win_length=winl)
    channel_2_spec = librosa.stft(audioR, n_fft=512, win_length=winl)

    spectro_two_channel = np.concatenate(
        (np.expand_dims(np.abs(channel_1_spec), axis=0), np.expand_dims(np.abs(channel_2_spec), axis=0)), axis=0)
    # print(spectro_two_channel.shape)
    return spectro_two_channel


def get_file_list(path):
    file_name_prefix = []
    for file in os.listdir(path):
        file_name_prefix.append(file.split('.')[0])
    file_name_prefix = np.unique(file_name_prefix)
    return file_name_prefix


def add_to_list(index_list, file_path, data):
    for index in index_list:
        rgb = os.path.join(file_path, index) + '.png'
        audio = os.path.join(file_path, index) + '.wav'
        depth = os.path.join(file_path, index) + '.npy'
        data.append([rgb, audio, depth])


class AudioVisualDataset(data.Dataset):
    def __init__(self, dataset, mode, config):
        super(AudioVisualDataset, self).__init__()
        self.train_data = []
        self.val_data = []
        self.test_data = []
        replica_dataset_path = config.replica_dataset_path
        mp3d_dataset_path = config.mp3d_dataset_path
        metadata_path = config.metadata_path
        if dataset == 'mp3d':
            self.win_length = 32
            self.audio_sampling_rate = 16000
            self.audio_length = 0.060
            # train,val,test scenes
            train_scenes_file = os.path.join(metadata_path, 'mp3d_scenes_train.txt')
            val_scenes_file = os.path.join(metadata_path, 'mp3d_scenes_val.txt')
            test_scenes_file = os.path.join(metadata_path, 'mp3d_scenes_test.txt')
            train_scenes = get_scene_list(train_scenes_file)
            val_scenes = get_scene_list(val_scenes_file)
            test_scenes = get_scene_list(test_scenes_file)
            for scene in os.listdir(mp3d_dataset_path):
                if scene in train_scenes:
                    for orn in os.listdir(os.path.join(mp3d_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(mp3d_dataset_path, scene, orn))
                        add_to_list(file_name_prefix, os.path.join(mp3d_dataset_path, scene, orn), self.train_data)
                elif scene in val_scenes:
                    for orn in os.listdir(os.path.join(mp3d_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(mp3d_dataset_path, scene, orn))
                        add_to_list(file_name_prefix, os.path.join(mp3d_dataset_path, scene, orn), self.val_data)
                elif scene in test_scenes:
                    for orn in os.listdir(os.path.join(mp3d_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(mp3d_dataset_path, scene, orn))
                        add_to_list(file_name_prefix, os.path.join(mp3d_dataset_path, scene, orn), self.test_data)
        if dataset == 'replica':
            self.win_length = 64
            self.audio_sampling_rate = 44100
            self.audio_length = 0.060
            # apartment 2, frl apartment 5, and office 4 are test scenes
            for scene in os.listdir(replica_dataset_path):
                if scene not in ['apartment_2', 'frl_apartment_5', 'office_4']:
                    # 训练集
                    for orn in os.listdir(os.path.join(replica_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(replica_dataset_path, scene, orn))
                        add_to_list(file_name_prefix, os.path.join(replica_dataset_path, scene, orn), self.train_data)
                else:
                    for orn in os.listdir(os.path.join(replica_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(replica_dataset_path, scene, orn))
                        val = file_name_prefix[:len(file_name_prefix) // 2]
                        test = file_name_prefix[len(file_name_prefix) // 2:]
                        add_to_list(val, os.path.join(replica_dataset_path, scene, orn), self.val_data)
                        add_to_list(test, os.path.join(replica_dataset_path, scene, orn), self.test_data)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.mode = mode
        self.dataset = dataset

    def __getitem__(self, index):
        # rgb, audio, depth
        if self.mode == 'train':
            data_ = self.train_data[index]
        elif self.mode == 'val':
            data_ = self.val_data[index]
        elif self.mode == 'test':
            data_ = self.test_data[index]
        # print(data_)
        rgb_path, audi_path, depth_path = data_[0], data_[1], data_[2]
        rgb = Image.open(rgb_path).convert('RGB')
        rgb = self.vision_transform(rgb)
        audio, audio_rate = librosa.load(audi_path, sr=self.audio_sampling_rate, mono=False, duration=self.audio_length)
        audio = normalize(audio)
        audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0, :], audio[1, :], self.win_length))
        depth = torch.FloatTensor(np.load(depth_path))
        depth = depth.unsqueeze(0)
        return {'img': rgb, 'audio': audio_spec_both, 'depth': depth}

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)
        elif self.mode == 'test':
            return len(self.test_data)


def get_data_loader(dataset_name, mode, shuffle, config):
    dataset = AudioVisualDataset(dataset_name, mode, config)
    return data.DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers)


class Config(object):
    def __init__(self):
        self.expr_dir = 'model_mp3d'
        self.lr_visual = 0.0001
        self.lr_audio = 0.0001
        self.lr_attention = 0.0001
        self.lr_material = 0.0001
        self.learning_rate_decrease_itr = -1
        self.decay_factor = 0.94
        self.optimizer = 'adam'
        self.weight_decay = 0.0001
        self.beta1 = 0.9
        self.batch_size = 100
        self.epochs = 50
        self.dataset = 'mp3d'
        self.checkpoints_dir = ''
        self.device = 'cuda'
        self.num_workers = 4
        self.init_material_weight = '/home/malong/Reproduction-Beyond/model_pth/material_pre_trained_minc.pth'
        self.replica_dataset_path = '/home/malong/Reproduction-Beyond/dataset/replica-dataset'
        self.mp3d_dataset_path = '/home/malong/Reproduction-Beyond/dataset/mp3d-dataset'
        self.metadata_path = '/home/malong/Reproduction-Beyond/dataset/metadata/mp3d'
        if self.dataset == 'replica':
            self.max_depth = 14.104
            self.audio_shape = [2, 257, 166]
        else:
            self.max_depth = 10.0
            self.audio_shape = [2, 257, 121]
        self.modo = 'train'
        self.display_freq = 1
        self.validation_freq = 1


# config = Config()
#
# a = AudioVisualDataset(config.dataset, config.modo, config)
# d = a[1]
# print(a[1])

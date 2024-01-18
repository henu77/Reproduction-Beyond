import os
import torch
import numpy as np
import torchvision.transforms as transforms
from ori_models.models import ModelBuilder
from ori_models.audioVisual_model import AudioVisualModel
from data_loader.my_dataset import get_data_loader
from util.util import compute_errors
from ori_models import criterion


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
        self.epochs = 100
        self.dataset = 'mp3d'
        self.checkpoints_dir = 'model_mp3d'
        self.device = 'cuda'
        self.num_workers = 2
        self.init_material_weight = '/home/malong/Reproduction-Beyond/model_pth/material_pre_trained_minc.pth'
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

        self.mode = 'test'
        self.display_freq = 100
        self.validation_freq = 100


opt = Config()

loss_criterion = criterion.LogDepthLoss()
opt.device = torch.device("cuda")

builder = ModelBuilder()
net_audiodepth = builder.build_audiodepth(opt.audio_shape,
                                          weights=os.path.join(opt.checkpoints_dir,
                                                               'audiodepth_' + opt.dataset + '.pth'))
net_rgbdepth = builder.build_rgbdepth(
    weights=os.path.join(opt.checkpoints_dir, 'rgbdepth_' + opt.dataset + '.pth'))
net_attention = builder.build_attention(
    weights=os.path.join(opt.checkpoints_dir, 'attention_' + opt.dataset + '.pth'))
net_material = builder.build_material_property(
    weights=os.path.join(opt.checkpoints_dir, 'material_' + opt.dataset + '.pth'))
nets = (net_rgbdepth, net_audiodepth, net_attention, net_material)

# construct our audio-visual model
model = AudioVisualModel(nets, opt)
# model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)
model.eval()

opt.mode = 'test'
dataloader_val = get_data_loader(opt.dataset, opt.mode, opt.batch_size, False, opt.num_workers,opt)
dataset_size_val = len(dataloader_val)
print('#validation clips = %d' % dataset_size_val)

losses, errs = [], []
with torch.no_grad():
    for i, val_data in enumerate(dataloader_val):
        val_data['audio'] = val_data['audio'].to(opt.device)
        val_data['img'] = val_data['img'].to(opt.device)
        val_data['depth'] = val_data['depth'].to(opt.device)
        output = model.forward(val_data)
        depth_predicted = output['depth_predicted']
        depth_gt = output['depth_gt']
        img_depth = output['img_depth']
        audio_depth = output['audio_depth']
        attention = output['attention']
        loss = loss_criterion(depth_predicted[depth_gt != 0], depth_gt[depth_gt != 0])
        losses.append(loss.item())

        for idx in range(depth_gt.shape[0]):
            errs.append(compute_errors(depth_gt[idx],
                                       depth_predicted[idx]))

mean_loss = sum(losses) / len(losses)
mean_errs = np.array(errs).mean(0)

print('Loss: {:.3f}, RMSE: {:.3f}'.format(mean_loss, mean_errs[1]))

errors = {}
errors['ABS_REL'], errors['RMSE'], errors['LOG10'] = mean_errs[0], mean_errs[1], mean_errs[5]
errors['DELTA1'], errors['DELTA2'], errors['DELTA3'] = mean_errs[2], mean_errs[3], mean_errs[4]
errors['MAE'] = mean_errs[6]

print('ABS_REL:{:.3f}, LOG10:{:.3f}, MAE:{:.3f}'.format(errors['ABS_REL'], errors['LOG10'], errors['MAE']))
print('DELTA1:{:.3f}, DELTA2:{:.3f}, DELTA3:{:.3f}'.format(errors['DELTA1'], errors['DELTA2'], errors['DELTA3']))
print('===' * 25)

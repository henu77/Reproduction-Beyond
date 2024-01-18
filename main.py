import os
import time
import torch

from ori_models.models import ModelBuilder
from ori_models.audioVisual_model import AudioVisualModel
from data_loader.my_dataset import get_data_loader
from util.util import TextWrite, compute_errors, mkdirs,compute_errors_batch
import numpy as np
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
        self.batch_size = 50
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
        self.display_freq = 50
        self.validation_freq = 50


def decrease_learning_rate(optimizer, decay_factor=0.94):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor


def create_optimizer(nets, opt):
    (net_visualdepth, net_audiodepth, net_attention, net_material) = nets
    param_groups = [{'params': net_rgbdepth.parameters(), 'lr': opt.lr_visual},
                    {'params': net_audiodepth.parameters(), 'lr': opt.lr_audio},
                    {'params': net_attention.parameters(), 'lr': opt.lr_attention},
                    {'params': net_material.parameters(), 'lr': opt.lr_material}
                    ]
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)


def evaluate(model, loss_criterion, dataset_val):
    losses = []
    errors = []
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            val_data['audio'] = val_data['audio'].to(config.device)
            val_data['img'] = val_data['img'].to(config.device)
            val_data['depth'] = val_data['depth'].to(config.device)
            output = model.forward(val_data)
            depth_predicted = output['depth_predicted']
            depth_gt = output['depth_gt']
            loss = loss_criterion(depth_predicted[depth_gt != 0], depth_gt[depth_gt != 0])
            losses.append(loss.item())
            errors.append(compute_errors_batch(depth_gt,
                                             depth_predicted))

    mean_loss = sum(losses) / len(losses)
    mean_errors = np.array(errors).mean(0)
    val_errors = {}
    val_errors['ABS_REL'], val_errors['RMSE'] = mean_errors[0], mean_errors[1]
    val_errors['DELTA1'] = mean_errors[2]
    val_errors['DELTA2'] = mean_errors[3]
    val_errors['DELTA3'] = mean_errors[4]
    return mean_loss, val_errors


config = Config()

mkdirs(config.expr_dir)

train_output_file = TextWrite(os.path.join(config.expr_dir, 'train_output.csv'))
val_output_file = TextWrite(os.path.join(config.expr_dir, 'val_output.csv'))
test_output_file = TextWrite(os.path.join(config.expr_dir, 'test_output.csv'))
# network builders
builder = ModelBuilder()
net_audiodepth = builder.build_audiodepth(audio_shape=config.audio_shape)
net_rgbdepth = builder.build_rgbdepth()
net_attention = builder.build_attention()
net_material = builder.build_material_property(init_weights=config.init_material_weight)
# exit()
nets = (net_rgbdepth, net_audiodepth, net_attention, net_material)

# construct our audio-visual model
model = AudioVisualModel(nets, config)
# print(model)
# model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(config.device)

# 数据集
train_dataloader = get_data_loader(config.dataset, "train", True, config)
val_data_loader = get_data_loader(config.dataset, "val", False, config)

optimizer = create_optimizer(nets, config)

# initialization
batch_loss = []
best_rmse = float("inf")
best_loss = float("inf")

loss_criterion = criterion.LogDepthLoss()

for epoch in range(1, config.epochs + 1):
    batch_loss = []
    for i, data in enumerate(train_dataloader):
        # print(data['audio'].shape)
        data['audio'] = data['audio'].to(config.device)
        data['img'] = data['img'].to(config.device)
        data['depth'] = data['depth'].to(config.device)

        model.zero_grad()
        output = model.forward(data)

        depth_predicted = output['depth_predicted']
        depth_gt = output['depth_gt']
        loss = loss_criterion(depth_predicted[depth_gt != 0], depth_gt[depth_gt != 0])
        batch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % config.display_freq == 0:
            avg_loss = sum(batch_loss) / len(batch_loss)
            print(f"train epoch: {epoch}, batch: {i+1}, loss: {avg_loss}")
            train_output_file.write_line([epoch, i+1, avg_loss])

        if (i + 1) % config.validation_freq == 0:
            model.eval()
            config.mode = 'val'
            val_loss, val_err = evaluate(model, loss_criterion, val_data_loader)
            print(f'valid epoch:{epoch}, batch:{i+1}, loss:{val_loss}, RMSE:{val_err["RMSE"]}')

            val_output_file.write_line(
                [epoch, i+1, val_loss,
                 val_err['RMSE'], val_err['ABS_REL'], val_err['DELTA1'],
                 val_err['DELTA2'], val_err['DELTA3']])
            model.train()
            config.mode = 'train'
            # save the model that achieves the smallest validation error
            if val_err['RMSE'] < best_rmse:
                best_rmse = val_err['RMSE']
                print('saving the best model (epoch %d) with validation RMSE %.5f\n' % (epoch, val_err['RMSE']))
                torch.save(net_rgbdepth.state_dict(),
                           os.path.join(config.expr_dir, 'rgbdepth_' + config.dataset + '.pth'))
                torch.save(net_audiodepth.state_dict(),
                           os.path.join(config.expr_dir, 'audiodepth_' + config.dataset + '.pth'))
                torch.save(net_attention.state_dict(),
                           os.path.join(config.expr_dir, 'attention_' + config.dataset + '.pth'))
                torch.save(net_material.state_dict(),
                           os.path.join(config.expr_dir, 'material_' + config.dataset + '.pth'))

    if (config.learning_rate_decrease_itr > 0 and epoch % config.learning_rate_decrease_itr == 0):
        decrease_learning_rate(optimizer, config.decay_factor)
        print('decreased learning rate by ', config.decay_factor)

import os
import shutil
from util.util import mkdirs,mkdir

old_path = "dataset/visual_echoes/echoes_navigable"
save_path='dataset/mp3d'

for sence in os.listdir(old_path):
    sence_path = os.path.join(old_path,sence)
    mkdirs(os.path.join(save_path,sence))
    for ori in os.listdir(sence_path+'/3ms_sweep_16khz'):
        mkdirs(os.path.join(save_path,sence,ori))
        for i in os.listdir(sence_path+'/3ms_sweep_16khz/'+ori):
            # pass
            # print(os.path.join(save_path,ori,i))
            shutil.copy(sence_path+'/3ms_sweep_16khz/'+ori+'/'+i,os.path.join(save_path,sence,ori,i))

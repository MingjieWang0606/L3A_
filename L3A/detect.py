import numpy as np
import os
import torch
import sys
import importlib
from visualizer.show3d_points import showpoints

cls_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
             'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
             'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
             'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

normal_channel = True
visible = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
experiment_dir = 'log/classification/pointnet2_cls_msg'
data_dir = 'test/data'
data_list = os.listdir(data_dir)
data_num = len(data_list)

model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
MODEL = importlib.import_module(model_name)
model = MODEL.get_model(40, normal_channel=True).cuda()
checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
criterion = MODEL.get_loss().cuda()

for i in range(data_num):
    data = np.loadtxt(os.path.join(data_dir, data_list[i]), delimiter=',').astype(np.float32)
    if not normal_channel:
        point_set = data[:, 0:3]
    else:
        point_set = data[:, 0:6]
    classifier = model.eval()
    inp = torch.from_numpy(point_set).unsqueeze(0).transpose(2, 1).cuda()
    pred, _ = classifier(inp)
    pred_choice = pred.data.max(1)[1]
    cls_idx = pred_choice.cpu().numpy()[0]
    print('The class of %s is：' % data_list[i] + cls_names[cls_idx])
    if visible:
        if normal_channel:
            point_set = point_set[:, 0:3]
        showpoints(point_set, ballradius=10)
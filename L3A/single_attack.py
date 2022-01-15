from adv_utils import *
from visualizer.show3d_points import showpoints
normal_channel = True

base_para = {
    'model': 'pointnet',
    'split_name': 'test',
    'step_num': 150,
    'lmd': 0.5,
    'dloss': 'L2',
    'is_sample': True,
    'n_points': 50,
    'n_samples': 50,
    'radius': 0.25,
    'back_thr': 0.1,
    'is_specific': False,
    'adv_target_idx': None,
    'save_pn_file': False,
    'save_as_dataset': False,
    'is_pwa': False,
    'is_lcons': False,
    'count_rate': 0.4
}
para = base_para

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
data_dir = 'test/data'
data_list = os.listdir(data_dir)
data_num = len(data_list)

experiment_dir = 'log/classification/' + para['model']
model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
MODEL = importlib.import_module(model_name)
model = MODEL.get_model(40, normal_channel=normal_channel).cuda()
classifier = model.eval()
checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
criterion = MODEL.get_loss().cuda()

for i in tqdm(range(data_num), total=data_num):
    data = np.loadtxt(os.path.join(data_dir, data_list[i]), delimiter=',').astype(np.float32)
    if not normal_channel:
        point_set = data[:, 0:3]
    else:
        point_set = data[:, 0:6]

    points = torch.from_numpy(point_set).unsqueeze(0).cuda()
    target_name = data_list[i].split('_')[0]
    target = torch.tensor(0.0).unsqueeze(0).cuda()
    for j, name in enumerate(cls_names):
        if target_name == name:
            target = torch.tensor(j, dtype=torch.float).unsqueeze(0).cuda()

    adv_points, adv_target, pred_class, idx = adv_propagation(model.eval(), criterion, points, target, para)

    inp = adv_points.detach().transpose(2, 1)
    pred, _ = classifier(inp)
    pred_choice = pred.data.max(1)[1]
    cls_idx = pred_choice.cpu().numpy()[0]
    print('%s类别为：' % data_list[i] + cls_names[cls_idx])

    if para['is_sample']:
        color_map = idx * 255
        color_map = np.append(color_map, np.zeros((idx.shape[0], 2)), axis=1)
    else:
        color_map = np.zeros((adv_points.shape[1], 3))

    with torch.no_grad():
        adv_points_set = adv_points.detach().squeeze().cpu().numpy()
        if normal_channel:
            adv_points_set = adv_points_set[:, 0:3]
        showpoints(adv_points_set, c_gt=color_map)

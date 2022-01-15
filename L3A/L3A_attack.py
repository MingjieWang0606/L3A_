"""
Author: Sun
Date: April 2021
"""
from adv_utils import *
from tensorboardX import SummaryWriter
from shutil import copyfile

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.enabled = False


def test_attack(model, loader, criterion, para):
    adv_t = para['adv_target_idx']
    save_pn_file = para['save_pn_file']
    file_idx = np.ones(40, dtype=int)
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    mean_correct_a = []
    class_acc_a = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        pred_names = []
        adv_names = []
        print("BATCH: %d" % j)
        points, target = data
        target = target[:, 0]
        # points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        with torch.no_grad():
            pred_choice, _ = inference(model, points)
            ori_names = []
            for i in range(points.shape[0]):
                 ori_names.append(cls_names[target[i]])

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1
            correct = pred_choice.eq(target.long().data).cpu().sum()
            if adv_t is not None:
                misjudge = pred_choice.eq(adv_target.long().data).cpu().sum()
                print('before ori accuracy:%f, adv accuracy:%f' 
                      % ((correct.item() / float(points.size()[0])), (misjudge.item() / float(points.size()[0]))))
            else:
                print('before ori accuracy:%f' % (correct.item() / float(points.size()[0])))
            mean_correct.append(correct.item() / float(points.size()[0]))

        adv_points, adv_target, pred_class, _ = adv_propagation(model.eval(), criterion, points, target, para)

        if save_pn_file:
            save_path = para['model'] + '_' + str(para['lmd']) + '_' + para['dloss']
            if not para['is_sample']:
                save_path = save_path + '_' + 'globel'
            save_path = os.path.join("test/result", save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i in range(adv_points.shape[0]):
                file_name = cls_names[target[i]]
                result = adv_points[i].detach().cpu().squeeze().numpy()
                adv_file_name = file_name + '_{:0>4d}'.format(file_idx[target[i]]) + '.txt'
                file_idx[target[i]] = file_idx[target[i]] + 1
                if para['save_as_dataset']:
                    split_file = 'modelnet40_' + para['split_name'] + '.txt'
                    sf = open(os.path.join(save_path, split_file), 'a')
                    sf.write(adv_file_name.split('.')[0] + '\n')
                    sf.close()
                    file_path = os.path.join(save_path, file_name)
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    np.savetxt(os.path.join(file_path, adv_file_name), result, fmt='%f', delimiter=',')
                else:
                    np.savetxt(os.path.join(save_path, adv_file_name), result, fmt='%f', delimiter=',')
            if para['save_as_dataset']:
                copyfile("data/modelnet40_normal_resampled/modelnet40_shape_names.txt",
                         os.path.join(save_path, 'modelnet40_shape_names.txt'))

        for k in range(points.shape[0]):
            pred_names.append(cls_names[pred_class[k]])
            adv_names.append(cls_names[adv_target[k]])
        print('ORI_TARGET:{}\nADV_TARGET:{}\nPRED_RESULT:{}'.format(ori_names, adv_names, pred_names))
        with torch.no_grad():
            pred_choice_a, _ = inference(model.eval(), adv_points)
            for cat in np.unique(target.cpu()):
                classacc_a = pred_choice_a[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc_a[cat, 0] += classacc_a.item() / float(adv_points[target == cat].size()[0])
                class_acc_a[cat, 1] += 1
            correct_a = pred_choice_a.eq(target.long().data).cpu().sum()
            misjudge_a = pred_choice_a.eq(adv_target.long().data).cpu().sum()
            print('after ori accuracy:%f, adv accuracy:%f'
                  % ((correct_a.item() / float(points.size()[0])), (misjudge_a.item() / float(points.size()[0]))))
            mean_correct_a.append(correct_a.item() / float(adv_points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    class_acc_a[:, 2] = class_acc_a[:, 0] / class_acc_a[:, 1]
    class_acc_a = np.mean(class_acc_a[:, 2])
    instance_acc_a = np.mean(mean_correct_a)

    return class_acc, instance_acc, class_acc_a, instance_acc_a


if __name__ == '__main__':
    base_para = {
        'model' : 'pointnet',
        'split_name' : 'test',
        'step_num' : 200,
        'lmd' : 0.2,
        'dloss' : 'L2',
        'is_sample' : True,
        'n_points' : 50,
        'n_samples' : 50,
        'radius' : 0.25,
        'back_thr' : 0.1,
        'is_specific' : True,
        'adv_target_idx' : None,
        'save_pn_file' : True,
        'save_as_dataset' : True,
        'is_pwa' : False,
        'is_lcons' : False
    }

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'models'))

    para = base_para

    experiment_dir = 'log/classification/' + para['model']

    num_class = 40

    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)

    model = MODEL.get_model(num_class, normal_channel=True).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    floss = MODEL.get_loss().cuda()

    DATA_PATH = 'data/modelnet40_normal_resampled/'
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split=para['split_name'], normal_channel=True)
    data_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=12, shuffle=False, num_workers=4)

    if para['is_sample']:
        print(para['model'] + ' lmd:' + str(para['lmd']) + ' ' + para['dloss'])
    else:
        print(para['model'] + ' lmd:' + str(para['lmd']) + ' ' + para['dloss'] + ' globel')

    print("N_POINTS = {}, N_SAMPLES = {}, RADIUS = {}, BACK_THR = {}\n"
        .format(para['n_points'], para['n_samples'], para['radius'], para['back_thr']))

    class_acc, instance_acc, class_acc_a, instance_acc_a = test_attack(model, data_loader, floss, para)

    print('before Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
    print('after Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc_a, class_acc_a))


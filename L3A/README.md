# Local Aggressive Adversarial Attack of 3D point Cloud 

**Code of "[Local Aggressive Adversarial Attack of 3D point Cloud](https://arxiv.org/abs/2105.09090)" and Local Aggressive And Physically Realizable Adversarial Attacks on Streaming 3D Point Cloud**

Local Aggressive Adversarial Attack of 3D point Cloud is accepted by ACML 2021 oral

Local Aggressive And Physically Realizable Adversarial Attacks on Streaming 3D Point Cloud is still under review


## L3A
### Data Preparation

Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

### Train Model
```
## Check model in ./models 
## E.g. pointnet2_msg
python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg
python test_cls.py --normal --log_dir pointnet2_cls_msg
```

### Run Attack
```
python L3A_attack.py
```

### Visualization
```
# Store your pc data in ./test/data

# Attack and visualize
python single_attack.py

# visualize only
python detect.py
```

### Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3) <br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) <br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br> 
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)

### Environments
Ubuntu 16.04 <br>
Python 3.6.12 <br>
Pytorch 1.1.0

## Physically Realizable Adversarial Attacks

### Installation

Install the required environments with the requirements.txt file using ANACONDA

```
conda env create -f requirements.yml

```

### Generating the adversarial object
```
python attack.py [-obj] [-obj_save] [-lidar] 

-obj Initial benign 3D object path

-obj_save Adversarial 3D object saving dir

-lidar LiDAR point cloud data path
```


### Download the target model



You can find the model through the [official Baidu Apollo GitHub]（https://github.com/ApolloAuto/apollo），Or you can download the model [here]（https://drive.google.com/file/d/17Eg1ySmucr1UQfye5wxAgGpv6VN5R0FP/view?usp=sharing） and then unzip it to `./data` folder. 



### Example for generating the adversarial object
```
python new_attack.py -obj ./object/object.ply -obj_save ./object/obj_save -lidar ./data/lidar.bin 
```


### Or you can modify it in new_attack.py as follows:

```

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-obj', '--obj', dest='object', default="./object/cube.ply")

    parser.add_argument('-obj_save' ,'--obj_save', dest='object_save', default="./object/obj_save")

    parser.add_argument('-lidar', '--lidar', dest='lidar', default='./data/lidar.bin')

    parser.add_argument('-cam', '--cam', dest='cam', default='./data/cam.png')

    parser.add_argument('-cali', '--cali', dest='cali', default='./data/cali.txt')

    parser.add_argument('-o', '--opt', dest='opt', default="Adam")  # pgd

    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.2)

    parser.add_argument('-it', '--iteration', dest='iteration', type=int, default=500)

    args = parser.parse_args()
```

### Acknowledgments

Our code is inspired by Invisible for both Camera and LiDAR: [Security of Multi-Sensor Fusion based Perception in Autonomous Driving Under Physical-World Attacks](https://sites.google.com/view/cav-sec/msf-adv) and [Towards Robust LiDAR-based Perception in Autonomous Driving](https://www.usenix.org/system/files/sec20_slides_sun.pdf)


### Citation

Please kindly cite the following paper in your publications if it helps your research:
```
@article{sun2021local,
  title={Local Aggressive Adversarial Attacks on 3D Point Cloud},
  author={Sun, Yiming and Chen, Feng and Chen, Zhiyu and Wang, Mingjie},
  journal={arXiv preprint arXiv:2105.09090},
  year={2021}
}
```

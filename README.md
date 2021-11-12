# PVE-MCC_for_unsignalized_intersection
Aiming at the problem of the traffic efficiency of intelligent networked vehicles passing through unsignalized-intersection in the future smart cities, this paper proposed a Progressive Value-expectation Estimation Multi-agent Cooperative Control (PVE-MCC) algorithm based on reinforcement learning. The algorithm takes the intelligent networked vehicles as the research object and designed the reward function for the optimization objective from the three aspects of traffic efficiency, safety, and comfort.

![visible](https://github.com/Mingtzge/PVE-MCC_for_unsignalized_intersection/blob/main/show_demo/demo.gif)

## Prerequisites
- Linux or macOS
- Python 3
- matlab 2017b
- CPU or NVIDIA GPU + CUDA CuDNN
### python modules
- numpy==1.16.2
- opencv-contrib-python==3.4.2.16
- opencv-python==4.2.0.32
- tensorflow==1.12.0
- matplotlib==3.0.2
- scipy==1.2.1 

## Getting Started
### Installation
- Clone this repo:
```bash
git clone git@github.com:Mingtzge/PVE-MCC_for_unsignalized_intersection.git
cd PVE-MCC_for_unsignalized_intersection
```

### Test the pre-trained model
```
python main.py --exp_name baseline --mat_path arvTimeNewVeh_new_1000_12.mat  --type test  --visible --video_name test
```

### train/test (on the "main" branch)
- Train a model:
```
python main.py --mat_path arvTimeNewVeh_for_train.mat --type train --exp_name train_demo
```
To see more intermediate results, run 
```
tensorboard --logdir ./model_data/train_demo/log
```
- Test the model:
```
python main.py --exp_name baseline --mat_path arvTimeNewVeh_new_1000_12.mat  --type train_demo  --visible --video_name test```
Note:the visual prarameters "--visible" and "--video_name" is optional. If use the "--visible", there will be a simulation interface to show the running interface of the vehicle in real time. the "--video_name test" is used to generate a video ,named "test.avi", saved in "./result_imgs/".

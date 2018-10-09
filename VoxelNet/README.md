# README for VoxelNet
This part of code is for VoxelNet section in our paper "Focal Loss in 3D Object Detection".
## Dependencies
- python 3.5.2
- tensorflow-gpu 1.5
- cv2
- numpy
- easydict
## Organization
```
.
├── config.py       // config file
├── kitti_eval      // for kitti offiline evaluation
├── log             // storing log files
├── model           // codes for network 
├── predicts        // storing test results (single one checkpoint)
├── predicts-all    // storing test results (all available checkpoints)
├── README.md
├── save_model      // storing weights
├── setup.py
├── setup.sh        // scipt for helping you setup (run this before start)
├── test_all.py     // code for evaluate all checkpoints
├── test.py         // code for evaluate single one checkpoint
├── train.py        // code for training
└── utils           // codes for basic operations
```
## Usage
### Compile Cython and KITTI Evaluation Codes
```
cd VoxelNet/
./setup.sh
```
### Configuration
Parameters can be set in the **config.py**.
The information for each parameters are detailed in this file.
### Train
After sucessfully setting your **config.py**, you can train your VoxelNet with
```
cd VoxelNet/
python3 train.py
```
### Evaluate
After sucessfully setting your **config.py**, you can test your VoxelNet with
```
cd VoxelNet/
python3 test.py         # test single checkpoint
# python3 test_all.py     # test all checkpoints
```
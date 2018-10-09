# README for 3D-FCN
This part of code is for 3D-FCN section in our paper "Focal Loss in 3D Object Detection".
## Dependencies
- python 3.5.2
- tensorflow-gpu 1.7
- cv2
- numpy
- easydict
- Cython
- shapely 
- numba 
- matplotlib
## Organization
```
.
├── config.py           // config file
├── input_velodyne.py
├── kitti_eval          // for kitti offiline evaluation
├── log                 // storing log files
├── model               // codes for network 
├── multiview_2d.py
├── parse_xml.py
├── predicts            // storing test results (single one checkpoint)
├── predicts-all        // storing test results (all available checkpoints)
├── README.md
├── save_model          // storing weights
├── test_all.py         // code for evaluate all checkpoints
├── test.py             // code for evaluate single one checkpoint
├── train.py            // code for training
└── utils               // codes for basic operations
```
## Usage
### Compile Cython and KITTI Evaluation Codes
```
cd 3D-FCN/
./setup.sh
```
### Configuration
Parameters can be set in the **config.py**.
The information for each parameters are detailed in this file.
### Train
After sucessfully setting your **config.py**, you can train your 3D-FCN with
```
cd 3D-FCN/
python3 train.py
```
### Evaluate
After sucessfully setting your **config.py**, you can test your 3D-FCN with
```
cd 3D-FCN/
python3 test.py         # test single checkpoint
# python3 test_all.py     # test all checkpoints
```
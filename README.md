# 2018.12视频通信大作业

*A Pytorch implement of NTIRE2018 No.1 network WDSR https://arxiv.org/abs/1808.08718v1* \
*Dataset: DIV2K 2017 https://data.vision.ee.ethz.ch/cvl/DIV2K/* 
```
DATA 
├── HR  
└── LR
```
*Training data is augmented with random horizontal filp and rotations, check utility.py and rewrite class SRdataset!*


## How to train
**Delete & make new**
```
vim ./loss.log
mkdir ./samples
mkdir ./checkpoint
```

**GPUs are needed for training**
```
python main.py --cuda
```

## How to test
**Test method**

700x700 HR image and its LR counterpart are randomly cropped from every image in DIV2K Validset  \
Calculate the mean PSNR of HR image and Image Restored by network

**make correspond empty folder to store samples before test**
```
mkdir ./foldername/
```

**change samples save_path and model to restore in psnr.py**
```
python psnr.py
```

***Specific description of given samples, checkpoint as well as test results can be found in .numbers file ^_^***

## Result
<img src="https://raw.githubusercontent.com/SJHNJU/WDSR/master/samples/1.png" width=300 alt='Truth'>

<img src="https://raw.githubusercontent.com/SJHNJU/WDSR/master/samples/2.png" width=300 alt='LR'>

<img src="https://raw.githubusercontent.com/SJHNJU/WDSR/master/samples/3.png" width=300 alt='Output'>

# 2018.12视频通信大作业

*A Pytorch implement of NTIRE2018 No.1 network WDSR https://arxiv.org/abs/1808.08718v1* \
*Dataset: DIV2K 2017 https://data.vision.ee.ethz.ch/cvl/DIV2K/* \
*Training data is augmented with horizontal filp and rotations, check utility.py!*

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

700x700 HR image and its LR counterpart are random cropped from every image in DIV2K Validset  \
Calculate the mean PSNR of HR image and Image Restored by network

**make correspond empty folder to store samples before test**
```
mkdir ./foldername/
```

**change samples save_path in psnr.py**
```
python psnr.py
```

***Specific description of given samples, checkpoint as well as test results can be found in .numbers file ^_^***

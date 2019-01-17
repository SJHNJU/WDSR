# 2018.12视频通信大作业
*A Pytorch implement of NTIRE2018 No.1 network WDSR https://arxiv.org/abs/1808.08718v1*

## How to train
**delete & make new**

./loss.log \
./samples/ \
./checkpoint/

**GPUs are needed for training** \
```python main.py --cuda```

## How to test
**test method**

700x700 HR image and its LR counterpart are random cropped from every image in the DIV2K validset  \
Calculate the mean PSNR of HR image and the image restored by network

**make correspond empty folder to store samples before test** \
```mkdir ./foldername/``` \
```python psnr.py```

***Specific description of given samples and checkpoint as well as test results are in the .numbers file***

## 2018.12视频通信大作业
*A Pytorch implement of NTIRE2018 No.1 network WDSR* \
https://arxiv.org/abs/1808.08718v1

### How to train
**delete & make new**

loss.log \
samples \
checkpoint

Use GPU for training \
*python main.py --cuda* 

look up .numbers to find specific description of given samples and checkpoint


### How to test
**test method**

Random crop 700x700 HR image from every image in the DIV2K validset and its LR counterpart \
calculate the mean PSNR of HR and the network's output

**make correspond empty folder to store samples before test** \
use *python psnr.py* begin test


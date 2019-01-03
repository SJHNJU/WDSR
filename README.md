## 正确训练网络的方式
每次都先删掉已有的：

1、 loss.log \
2、 samples \
3、 checkpoint

再新建samples checkpoint 两个文件夹和空的loss.log
然后修改参数

python main.py --cuda 开始训练
有关samples 和 checkpoint模型说明请按侯后面的的序号查看电子表格

## 如何测试
测试方法：

在DIV2K validset上每张HR图随机截700x700图块与对应输出图像求psnr，最后求平均值
请先建立对应的空文件夹存放测试采样图片

python psnr.py 开始测试

### 所有训练测试都需要在服务器上进行

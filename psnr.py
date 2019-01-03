import math
import scipy.misc
from wdsr_b import *
from args import *
from torch.autograd import Variable
from utility import *


'''
    test the PNSR 

'''


def psnr(lr_path, hr_path):
    lr = io.imread(lr_path).astype(np.float32) / 255
    hr = io.imread(hr_path).astype(np.float32) / 255

    m, n, c = lr.shape[0], hr.shape[1], hr.shape[2]
    mse = np.sum((lr - hr) ** 2) / (m * n * c)

    return 10 * math.log10(1 / mse)


def LR2HR(lr_path, hr_path, model_path, args):
    lr = io.imread(lr_path).astype(np.float32).transpose((2, 0, 1))
    hr = io.imread(hr_path).astype(np.float32).transpose((2, 0, 1))

    hr, lr = torch.from_numpy(hr), torch.from_numpy(lr)

    lr, hr = Variable(lr, requires_grad=False).cuda(), Variable(hr, requires_grad=False).cuda()

    model = MODEL(args).cuda()

    print('===> Load model')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"].state_dict())
    
    print('===> Begin SR')
    output = model(lr)

    output = output.cpu().data.numpy()
    print('Net output size is {}'.format(output.shape))
    output = np.squeeze(output)
    print('squeezed size is {}'.format(output.shape))

    output = output.transpose((1, 2, 0)) / 255 
    
    print('output size is {}'.format(output.shape))
    scipy.misc.toimage(output, cmin=0.0, cmax=1.0).save('./figs/out.png')


'''
    Random crop 400x400, 800x800 from LR, HR image
    Calculate the PSNR on the validation set
'''


def test(model_path, args, hr_path='./DIV2K/DIV2K_valid_HR', lr_path='./DIV2K/DIV2K_valid_LR_bicubic/X2', ):
    model = MODEL(args).cuda()
    version = 6
    print('===> Load model')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"].state_dict())

    PSNR = []
    for i in range(801, 901):
        hr_img_path = hr_path + '/0{}.png'.format(i)
        lr_img_path = lr_path + '/0{}x2.png'.format(i)
        lr = io.imread(lr_img_path).astype(np.float32)
        hr = io.imread(hr_img_path).astype(np.float32)
        # Random crop image
        m = random.randint(0, lr.shape[0] - 352)
        n = random.randint(0, lr.shape[1] - 352)

        hr = hr[2*m:2*m+700, 2*n:2*n+700, :]
        hr = hr / 255
        scipy.misc.toimage(hr, cmin=0.0, cmax=1.0).save('./{}/hr{}.png'.format(version, i))
        lr = lr[m:m + 350, n:n + 350, :]
        lr = lr.transpose((2, 0, 1))

        lr = torch.from_numpy(lr)

        lr = Variable(lr, requires_grad=False).cuda()
        print('===> Begin SR {}.png'.format(i))
        output = model(lr)
        print('===> Save image{}'.format(i))
        output = output.cpu().data.numpy()
        output = np.squeeze(output)
        output = output.transpose((1, 2, 0)) / 255
        scipy.misc.toimage(output, cmin=0.0, cmax=1.0).save('./{}/out{}.png'.format(version, i))
        psnr_data = psnr('./{}/out{}.png'.format(version, i), './{}/hr{}.png'.format(version, i))
        print('image{} psnr: {}'.format(i, psnr_data))
        PSNR.append(psnr_data)

    return PSNR


if __name__ == '__main__':
    args = get_args()
    model_path = './checkpoint6/model_epoch133_step1.pth'
    PSNR = test(model_path, args, hr_path='./DIV2K/DIV2K_valid_HR', lr_path='./DIV2K/DIV2K_valid_LR_bicubic/X2')
    
    PSNR = np.array(PSNR)
    print(PSNR.mean())






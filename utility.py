import random
import os
from skimage import io
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import matplotlib.pyplot as plt
import numpy as np

def check():
    hr_root = './DATA_augment/HR/'
    lr_root = './DATA_augment/LR/'
    for i in range(42000):
        hr_path = hr_root + str(i) + '.png'
        lr_path = lr_root + str(i) + 'x2.png'

        if os.path.exists(hr_path):
            continue
        else:
            print(i)

        if os.path.exists(lr_path):
            continue
        else:
            print(i)            

def random_crop():
    hr_root = './DATA/HR/'
    lr_root = './DATA/LR/'
    for i in range(14, 10414):
        hr_path = hr_root + str(i) + '.png'
        lr_path = lr_root + str(i) + 'x2.png'

        hr_img = cv2.imread(hr_path)
        lr_img = cv2.imread(lr_path)

        hr_img_flip_1 = cv2.flip(hr_img,1)
        lr_img_flip_1 = cv2.flip(lr_img,1)

        hr_img_flip_0 = cv2.flip(hr_img,0)
        lr_img_flip_0 = cv2.flip(lr_img,0)

        hr_img_flip_m1 = cv2.flip(hr_img,-1)
        lr_img_flip_m1 = cv2.flip(lr_img,-1)

        idx = i - 13
        
        # print(idx)
        cv2.imwrite('./DATA_augment/HR/'+str(idx)+'.png', hr_img)
        cv2.imwrite('./DATA_augment/LR/'+str(idx)+'x2.png', lr_img)
        cv2.imwrite('./DATA_augment/HR/'+str(idx+10400)+'.png', hr_img_flip_1)
        cv2.imwrite('./DATA_augment/LR/'+str(idx+10400)+'x2.png', lr_img_flip_1)
        cv2.imwrite('./DATA_augment/HR/'+str(idx+20800)+'.png', hr_img_flip_0)
        cv2.imwrite('./DATA_augment/LR/'+str(idx+20800)+'x2.png', lr_img_flip_0)
        cv2.imwrite('./DATA_augment/HR/'+str(idx+31200)+'.png', hr_img_flip_m1)
        cv2.imwrite('./DATA_augment/LR/'+str(idx+31200)+'x2.png', lr_img_flip_m1)

class SRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        hr_dir = self.root_dir + '/HR'
        return len([name for name in os.listdir(hr_dir)
                    if os.path.isfile(os.path.join(hr_dir, name))])-1

    def __getitem__(self, idx):
        if idx > 41599:
            idx = idx - 200
        hr_name = self.root_dir + '/HR/' + str(idx+2) + '.png'
        lr_name = self.root_dir + '/LR/' + str(idx+2) + 'x2.png'
        hr = io.imread(hr_name)
        lr = io.imread(lr_name)
        hr = hr.astype('float32')
        lr = lr.astype('float32')

        data_dict = {'hr': hr, 'lr': lr}

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        hr, lr = sample['hr'], sample['lr']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        hr = hr.transpose((2, 0, 1))
        lr = lr.transpose((2, 0, 1))
        return {'hr': torch.from_numpy(hr),
                'lr': torch.from_numpy(lr)}


if __name__ == '__main__':
    check()
    # dataset = SRDataset(root_dir='./DATA',
    #                     transform=transforms.Compose([ToTensor()])
    #                     )

    # data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1)

    # for batch_idx, data_batch in enumerate(data_loader, start=1):
    #     for i in range(16):
    #         hr = data_batch['hr'].numpy()[i]
    #         lr = data_batch['lr'].numpy()[i]

    #         hr = hr / 255
    #         lr = lr / 255

    #         plt.subplot(121), plt.imshow(np.transpose(hr, (1, 2, 0)))
    #         plt.subplot(122), plt.imshow(np.transpose(lr, (1, 2, 0)))
    #         plt.show()

    #         if i == 7:
    #             break

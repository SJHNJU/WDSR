from skimage import io
from torch.autograd import Variable
import os
import torch
import scipy.misc


def save_checkpoint(model, epoch, step):
    model_folder = "./checkpoint/"
    model_out_path = model_folder + "model_epoch{}_step{}.pth".format(epoch, step)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def adjust_learning_rate(args, epoch):
    """Sets the learning rate to the initial LR decayed by 0.5 every step epoch"""
    lr = args.lr * (0.5 ** (epoch // args.step))
    return lr


def train(data_loader, optimizer, model, criterion, epoch, args):
    lr = adjust_learning_rate(args, epoch)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    step = 0
    for iteration, batch in enumerate(data_loader, 1):
        step += 1
        lr, hr = Variable(batch['lr']), Variable(batch['hr'], requires_grad=False)

        if args.cuda:
            hr = hr.cuda()
            lr = lr.cuda()
        output = model(lr)
        loss = criterion(output, hr)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

#       Save a batch image every 200 batches
        if iteration % 260 == 0:
            hr_batch = hr.cpu().data.numpy().transpose(0, 2, 3, 1)
            out_batch = output.cpu().data.numpy().transpose(0, 2, 3, 1)
            lr_batch = lr.cpu().data.numpy().transpose(0, 2, 3, 1)

            for i in range(5):
                hr_img = hr_batch[i] / 255
                out_img = out_batch[i] / 255
                lr_img = lr_batch[i] / 255

                scipy.misc.toimage(out_img, cmin=0.0, cmax=1.0).save(
                                    './samples/epoch{}_step{}_out{}.png'.format(epoch, step, i))
                scipy.misc.toimage(hr_img, cmin=0.0, cmax=1.0).save(
                                    './samples/epoch{}_step{}_hr{}.png'.format(epoch, step, i))
                scipy.misc.toimage(lr_img, cmin=0.0, cmax=1.0).save(
                                    './samples/epoch{}_step{}_lr{}.png'.format(epoch, step, i))

            loss_str = "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(data_loader),
                                                                loss.data[0])
            print(loss_str)
            with open('./loss.log', 'a+') as f:
                f.write(loss_str+'\n')



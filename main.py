
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utility import *
from args import *
from wdsr_b import *
from train import *

os.environ["CUDA_VISIBLE_DEVICES"] = "9"


if __name__ == '__main__':
    args = get_args()
    print(args)

    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    seed = 1000
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True

    print("===> Loading dataset")
    dataset = SRDataset(root_dir='./DATA_augment',
                        transform=transforms.Compose([ToTensor()])
                        )

    data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=1)

    print("===> Building model")
    model = MODEL(args)
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                           betas=(0.9, 0.99), eps=1e-08)

    print("===> Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        train(data_loader, optimizer, model, criterion, epoch, args)
        save_checkpoint(model, epoch, 1)

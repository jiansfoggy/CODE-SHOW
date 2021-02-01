import os
import yaml,argparse
import torch
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
#from src.datasets import create_dataloader
from src.model import EfficientDet
from tensorboardX import SummaryWriter
import shutil
import numpy as np
from tqdm.autonotebook import tqdm
 
def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=8, help="The number of images per batch")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--data_path", type=str, default="data/COCO", help="the root folder of dataset")
    parser.add_argument("--log_path", type=str, default="tensorboard/signatrix_efficientdet_coco")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument('--image_weights', action='store_true', help='use weighted image selection for training')

    args = parser.parse_args()
    return args

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'Efficient-Det torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    return torch.device('cuda:0' if cuda else 'cpu')
 
def train(opt):
    num_gpus = 1
    #device = select_device(opt.device, batch_size=opt.batch_size)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    #torch.manual_seed(123)
    training_params = {"batch_size": opt.batch_size * num_gpus,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": collater,
                       "num_workers": 12}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": collater,
                   "num_workers": 12}

    rank = opt.global_rank
    training_set = CocoDataset(root_dir=opt.data_path, set="train.txt",
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    training_generator = DataLoader(training_set, **training_params)
    test_set = CocoDataset(root_dir=opt.data_path, set="valid.txt",
                           transform=transforms.Compose([Normalizer(), Resizer()]))
    test_generator = DataLoader(test_set, **test_params)

    nb = len(training_generator)
    #nb_t = len(test_generator)
    model = EfficientDet(num_classes=1)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    writer = SummaryWriter(opt.log_path)

    if torch.cuda.is_available():
        model = model.cuda()
        #model = model.to(device)
        model = nn.DataParallel(model)
    
    #model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    best_loss = 1e5
    best_epoch = 0
    model.train()

    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epochs):
        model.train()
        # if torch.cuda.is_available():
        #     model.module.freeze_bn()
        # else:
        #     model.freeze_bn()
        epoch_loss = []

        progress_bar = tqdm(training_generator)
        for iter, data in enumerate(progress_bar):
            try:
                optimizer.zero_grad()
                
                if torch.cuda.is_available():
                    #cls_loss, reg_loss = model([data['img'].to(device,non_blocking=True).float(),data['annot'].to(device)])
                    cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                else:
                    cls_loss, reg_loss = model([data['img'].float(), data['annot']])
                
                #cls_loss, reg_loss = model([data['img'].float(), data['annot']])
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                loss = cls_loss + reg_loss
                if loss == 0:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                epoch_loss.append(float(loss))
                total_loss = np.mean(epoch_loss)

                progress_bar.set_description(
                    'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Batch loss: {:.5f} Total loss: {:.5f}'.format(
                        epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss, reg_loss, loss,
                        total_loss))
                writer.add_scalar('Train/Total_loss', total_loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Regression_loss', reg_loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Classfication_loss (focal loss)', cls_loss, epoch * num_iter_per_epoch + iter)

            except Exception as e:
                print(e)
                continue
        scheduler.step(np.mean(epoch_loss))

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_regression_ls = []
            loss_classification_ls = []
            for iter, data in enumerate(test_generator):
                with torch.no_grad():
                
                    if torch.cuda.is_available():
                        #cls_loss, reg_loss = model([data['img'].to(device,non_blocking=True).float(),data['annot'].to(device)])
                        cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                    else:
                        cls_loss, reg_loss = model([data['img'].float(), data['annot']])
         
                    #cls_loss, reg_loss = model([data['img'].float(), data['annot']])
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss_classification_ls.append(float(cls_loss))
                    loss_regression_ls.append(float(reg_loss))
            cls_loss = np.mean(loss_classification_ls)
            reg_loss = np.mean(loss_regression_ls)
            loss = cls_loss + reg_loss

            print(
                'Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                    epoch + 1, opt.num_epochs, cls_loss, reg_loss,
                    np.mean(loss)))
            writer.add_scalar('Test/Total_loss', loss, epoch)
            writer.add_scalar('Test/Regression_loss', reg_loss, epoch)
            writer.add_scalar('Test/Classfication_loss (focal loss)', cls_loss, epoch)

            if loss + opt.es_min_delta < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(model, os.path.join(opt.saved_path, "signatrix_efficientdet_coco.pth"))

                dummy_input = torch.rand(opt.batch_size, 3, 512, 512)
                 
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                    #dummy_input = dummy_input.to(device)
                
                if isinstance(model, nn.DataParallel):
                    model.module.backbone_net.model.set_swish(memory_efficient=False)
                    print("before onnx, go into model file")
                    torch.onnx.export(model.module, dummy_input,
                                      os.path.join(opt.saved_path, "signatrix_efficientdet_coco.onnx"),
                                      verbose=False)
                    print("after onnx")
                    model.module.backbone_net.model.set_swish(memory_efficient=True)
                else:
                    model.backbone_net.model.set_swish(memory_efficient=False)

                    torch.onnx.export(model, dummy_input,
                                      os.path.join(opt.saved_path, "signatrix_efficientdet_coco.onnx"),
                                      verbose=False)
                    model.backbone_net.model.set_swish(memory_efficient=True)

            print("go out from onnx")
            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss))
                break
    writer.close()
    

if __name__ == "__main__":
    opt = get_args()
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    opt.quad, opt.cache_images, opt.rect = True, True, True
    opt.image_weights = True

    train(opt)

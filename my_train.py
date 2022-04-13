import argparse
import logging
import sys
from pathlib import Path
from numpy import save

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.dice_score import dice_loss, calc_loss
from evaluate import evaluate
from unet import UNet
from dataloader import UKB_SAX_EDES, UKB_SAX
from utils.dice_score import HD, dice_loss, area_coverage

import matplotlib.pyplot as plt 
torch.multiprocessing.set_sharing_strategy('file_system')
def save_seg(img,root:str,id:str,cmap='tab10', nc:int = 4):
    # print(img.shape)
    B,C,H,W = img.shape
    # img = img[:,:,0:3,...]
    if type(img).__module__ != 'numpy':
        img_print = img.cpu().detach()
    else:
        img_print = img
    img_print = torch.argmax(img_print, dim=1)
    seg_print = img_print[0,...].numpy()
    fig = plt.figure()
    plt.imshow(seg_print,cmap=plt.get_cmap(cmap))
    plt.axis('off')
    fig.savefig('{}/{}.png'.format(root,id),bbox_inches='tight')
    plt.close(fig)
    # plt.show()

data_dir = '/usr/not-backed-up/scnb/data/masks_sax_5k/'
data_dicom = '/usr/not-backed-up/scnb/data/dicom_lsax_5k/'
dir_checkpoint = '/usr/not-backed-up/scnb/motion_model'
save_root = '/usr/not-backed-up/scnb/unet_results'
def test_net_ngt(net,
             device,
             save_img: bool = False,
             ):
    # *1. Create dataset
    try:
        dataset = UKB_SAX(root_gt=data_dir, root_dicom=data_dicom, mode='test',num_of_frames=10)
    except (AssertionError, RuntimeError):
        print('Error in initializing Dataset.')


    # *2. Create data loaders
    n_test = int(len(dataset))
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)

    
    # *3. Begin training
    net.eval()
    num_val_batches = len(test_loader)
    dice_score = 0

    # iterate over the validation set
    for ind, batch in enumerate(test_loader):
        if ind == 2:
            break
        for t in range(batch[0].size(1)):
            image = batch[0][:,t:t+1,...]
        # *move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                # predict the mask
                mask_pred = net(image)
                if save_img:
                    save_seg(mask_pred,root=save_root,id='pre_{}_t{}'.format(ind,t))
           
def test_net(net,
             device,
             save_img: bool = False,
             ):
    # *1. Create dataset
    try:
        dataset = UKB_SAX(root_gt=data_dir, root_dicom=data_dicom, mode='test', num_of_frames=10)
    except (AssertionError, RuntimeError):
        print('Error in initializing Dataset.')


    # *2. Create data loaders
    n_test = int(len(dataset))
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)

    
    # *3. Begin training
    net.eval()
    num_val_batches = len(test_loader)
    dice_score = torch.empty(2,3,n_test,dtype=torch.float32)
    hds = torch.empty(2,3,n_test)
    gt_ac = torch.empty(3,2,n_test)
    ac = torch.empty(3,9,n_test)
    # iterate over the validation set
    for ind, batch in enumerate(test_loader):
        image = batch[0]
        mask_true = batch[1]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        # if save_img:
        #     save_seg(mask_true,root=save_root,id='gt_{}'.format(ind))
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            gt_ac[:,0,ind] = area_coverage(seg=mask_true[0,0,...],is_gt=True)
            gt_ac[:,1,ind] = area_coverage(seg=mask_true[0,1,...],is_gt=True)
            # predict the mask
            # * for each time point 
            for i in range(image.size(1)):
                mask_pred = net(image[0:1,i:i+1,...]) # 1*4*128*128
                mask_logits = torch.softmax(mask_pred,dim=1)
                # * calculate area coverage 
                if i > 0:
                    ac[:,i-1,ind] = area_coverage(seg=mask_pred[0,...])
                # * calculate DICE on ed and es time point 
                if i == 0:
                    to_hd = mask_logits[0]
                    to_hd[-1,...] = to_hd[-1,...] + to_hd[-1,...]
                    for k in range(3):
                        dice_score[0,k,ind] = 1. - dice_loss(prediction=mask_logits[0,k,...],target=mask_true[0,0,k,...]).item()
                        hds[0,k,ind] = HD(pred=to_hd[k,...],ref=mask_true[0,0,k,...]).item()
                if i == 4:
                    for k in range(3):
                        dice_score[1,k,ind] = 1. - dice_loss(prediction=mask_logits[0,k,...],target=mask_true[0,1,k,...]).item()
                        hds[1,k,ind] = HD(pred=to_hd[k,...],ref=mask_true[0,1,k,...]).item()


    logging.info(f'''Test INFO: Unet
    Number of test case: {n_test}
    DICE at [ED, ES]: mean = {torch.mean(dice_score,dim=-1)} std = {torch.std(dice_score,dim=-1)}
    HD at [ED, ES]: mean = {torch.mean(hds,dim=-1)} std = {torch.std(hds,dim=-1)}
    AC GT: mean = {torch.mean(gt_ac,dim=-1)} std = {torch.std(gt_ac,dim=-1)}
    AC Prediction: mean = {torch.mean(ac,dim=-1)} std = {torch.std(ac,dim=-1)}
    ''')
                
                

    
def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 10,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1,
              amp: bool = False):
    # *1. Create dataset
    try:
        dataset = UKB_SAX_EDES(root_gt=data_dir, root_dicom=data_dicom,mode='train',num_of_frames=10)
    except (AssertionError, RuntimeError):
        print('Error in initializing Dataset.')

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    print('Num of TRAIN: {} Num of VALIDATION: {}'.format(n_train, n_val))
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.8, min_lr=1e-8)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for ind, batch in enumerate(train_loader):
                images = batch[0][:,1:2,...]
                true_masks = batch[1][:,-1,...]

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = calc_loss(masks_pred, true_masks, bce_weight = 1) 

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), '{}/128by128_sax_unet_0303_checkpoint_epoch{}.pth'.format(dir_checkpoint,epoch + 1))
            logging.info(f'Checkpoint {epoch + 1} saved!')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--mode', default='train', help='mode train or test')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=1, n_classes=4, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    if args.mode == 'test':
        test_net(net=net, device=device, save_img=False)
    if args.mode == 'test_ngt':
        test_net_ngt(net=net, device=device, save_img=True)
    if args.mode == 'train':
        try:
            train_net(net=net,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    device=device,
                    img_scale=args.scale,
                    val_percent=args.val / 100,
                    amp=args.amp)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            sys.exit(0)
# device = 'cuda:0'
# train_loader = get_loader(root=data_dir,axis='sax',mode='train')
# n_train = len(train_loader)
# net = UNet(n_channels=1, n_classes=4, bilinear=False)
# lr = 1e-3
# optimizer = optim.Adam(list(net.parameters()), lr=lr)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
# criterion = nn.CrossEntropyLoss()
# epochs = 50
# net.to(device)

# for epoch in range(epochs):
#     with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
#         for ind,item in enumerate(train_loader):
#             with torch.autograd.set_detect_anomaly(True):
#                 imgs = item[0][:,1:2,...].to(device)
#                 segs = item[1][:,-1,...].to(device)
#                 preds = net(imgs)
#                 preds = F.softmax(preds, dim=1)
#                 loss = calc_loss(preds, segs, bce_weight = 1.)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

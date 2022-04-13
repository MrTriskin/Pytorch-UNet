import os
import re
import cv2
import pydicom as dicom 
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils import data
from torchvision import transforms as T
import torch.nn.functional as F
import imageio
from utils import *
from config import *
def get_indexs(seg_tps:list,tg:int):
    ed = int(re.findall(r'\d+',seg_tps[0])[0])
    es = int(re.findall(r'\d+',seg_tps[1])[0])
    tp = int(tg / 2 )
    gap = int((es - ed) // tp) 
    indexs = [ed,es]
    for i in range(1,tp):
        indexs.append(int(ed + i*gap))
        indexs.append(int(min(es + i*gap,50 - 2 * (tp - i))))
    indexs.sort()
    return indexs

def readable_dicom(dicom_dict):
    tmp_img_list = ['{}time{}/SAX/0005'.format(dicom_dict['img_dir'],"%03d" %(j)) for j in dicom_dict['indexs']]     
    for i in tmp_img_list:
        try:
            x = dicom.read_file(i).pixel_array
        except:
            print('CANT READ: {}'.format(i))
            return False
    return True

def readable_png(png_dict):
    tmp_seg_list = ['{}{}/masks/{}/0005.png'.format(png_dict['gt_dir'],j, k) for j in [png_dict['seg_tps'][0],png_dict['seg_tps'][-1]] for k in ['RVendo','LVendo','LVepi'] ]
    for i in tmp_seg_list:
        try:
            x = imageio.imread(i)
        except:
            print('CANT READ: {}'.format(i))
            return False
    return True

def getROI(rvendo,lvepi):
    x = imageio.imread(rvendo)
    
    tmp = np.where(x == np.max(x))
    r_max = np.max(tmp[0])
    c_min = np.min(tmp[1])
    x = imageio.imread(lvepi)
    tmp = np.where(x == np.max(x))
    c_max = np.max(tmp[1])
    r_min = np.min(tmp[0])
    
    return(r_min, r_max, c_min, c_max)

def fill_seg(seg):
    cnts = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(seg, pts =cnts[0], color=(255,255,255))
    return seg

class Norm_intensity(object):
    def __call__(self, pic, thres_roi=10.0):
        """ Normalise the image intensity by the mean and standard deviation """
        val_l = np.percentile(pic, thres_roi)
        roi = (pic >= val_l)
        mu, sigma = torch.mean(pic[roi]), torch.std(pic[roi])
        eps = 1e-6
        image2 = (pic - mu) / (sigma + eps)
        image2 = (image2 - torch.min(image2)) / (torch.max(image2) - torch.min(image2) + eps) 
        return image2

    def __repr__(self):
        return(self.__class__.__name__ + '()')
class Rescale_Intensity(object):
    def __call__(self, image, thres=(1.0, 99.0)):
        """ Rescale the image intensity to the range of [0, 1] """
        image = np.array(image)
        val_l, val_h = np.percentile(image, thres)
        image2 = image
        image2[image < val_l] = val_l
        image2[image > val_h] = val_h
        image2 = ((image2.astype(np.float32) - val_l) / (val_h - val_l)) 
        return image2

    def __repr__(self):
        return(self.__class__.__name__ + '()')
class UKB_SAX_EDES(data.Dataset):
    def __init__(self,root_gt:str,root_dicom:str, mode='train',num_of_frames:int=6):
        super(UKB_SAX_EDES,self).__init__()
        self.root_gt = root_gt
        self.root_dicom = root_dicom
        self.num_of_frames = num_of_frames
        self.resize_to = TRAIN_RESIZE
        self.no_seg = []
        self.with_seg = []
        reload_flg = False
        # Check for pre-saved id list 
        if os.path.isfile('./patient_ids.npy'):
            patient_ids = np.load('./patient_ids.npy')
        # Set reload flag as TRUE 
            reload_flg = True 
        else:
            # get patients' ID from root (/dataset)
            patient_ids = os.listdir(self.root_gt)
        print("Total number of patients is: {}...".format(len(patient_ids)))
        # walk through all patient's folder, remove sample if it dose not contain gt segmentation.
        no_seg = []
        with_seg= []
        patient_ids_new = []
        for i in patient_ids:
            if reload_flg:
                # Get gt frame number
                # REF gt_dir/image&mask/time/masks/RVendo/LVendo/LVepi
                gt_dir = "{}{}/image&mask/".format(self.root_gt,i)
                seg_tps = os.listdir(gt_dir)
                # under cine, image is DICOM 
                # REF img_dir/id/time00x/SAX/0005
                img_dir = "{}{}/".format(self.root_dicom,i)
                patient_id = i 
                seg = True 
                indexs = get_indexs(seg_tps,self.num_of_frames)
                instance_dict = {'id':patient_id, 'has_seg':seg, 'img_dir':img_dir, 'gt_dir':gt_dir, 'seg_tps':seg_tps, 'indexs':indexs}
                self.with_seg.append(instance_dict)
                patient_ids_new.append(i)
            else:
                # Check if both dicom and segs are exist
                has_dicom = os.path.isdir('{}{}'.format(self.root_dicom,i))
                if  has_dicom:
                    # Get gt frame number
                    # REF gt_dir/image&mask/time/masks/RVendo/LVendo/LVepi
                    gt_dir = "{}{}/image&mask/".format(self.root_gt,i)
                    seg_tps = os.listdir(gt_dir)
                    # under cine, image is DICOM 
                    # REF img_dir/id/time00x/SAX/0005
                    img_dir = "{}{}/".format(self.root_dicom,i)
                    patient_id = i 
                    if len(seg_tps) == 2 and len(os.listdir(img_dir)) >= 50:
                        # Assume all patient has 50 frames of MRI scans
                        seg = True 
                        indexs = get_indexs(seg_tps,self.num_of_frames)
                        instance_dict = {'id':patient_id, 'has_seg':seg, 'img_dir':img_dir, 'gt_dir':gt_dir, 'seg_tps':seg_tps, 'indexs':indexs}
                        if readable_dicom(instance_dict) and readable_png(instance_dict):
                            self.with_seg.append(instance_dict)
                            patient_ids_new.append(i)
                else:
                    print('For case {}, root_dicom: {}...'.format(i, has_dicom))
        print('The number of no-seg samples is: {}...'.format(len(self.no_seg)))
        print('The total number of seg samples is: {}...'.format(len(self.with_seg)))
        if not os.path.isfile('./patient_ids.py'):
            print('Saving clean patient id list...')
            np.save('./patient_ids.npy',patient_ids_new)
        if mode == 'train':
            self.with_seg = self.with_seg[0:4600]
        if mode == 'test':
            self.with_seg = self.with_seg[4600:4690]
            
    def make_seg_gt(self,segs):
        # INPUT: 6*H*W 
        # OUTPUT: 2*4*H*W 
        _,H,W = segs.shape
        # *background = ones - rvendo - lvepi
        br_ed = torch.ones(1,H,W,dtype=torch.long) - segs[0,...] - segs[2,...]
        br_es = torch.ones(1,H,W,dtype=torch.long) - segs[3,...] - segs[5,...]
        # *subtract lvendo from lvepi to form lvepi segs 
        segs[2,...] = segs[2,...] - segs[1,...]
        segs[-1,...] = segs[-1,...] - segs[-2,...]
        # *saperate segs for ed and es phases
        ed_seg = segs[0:3,...]
        ed_seg = torch.cat([segs[0:3,...],br_ed],dim=0)
        es_seg = segs[3:6,...]
        es_seg = torch.cat([segs[3:6,...],br_es],dim=0)
        out = torch.stack([ed_seg,es_seg],dim=0)
        out = torch.clamp(out,min=0.,max=1.)
        return out
        # *make label maps for cross entropy. dtype should be int64
        # *background -> 3, rvendo -> 0, lvendo -> 1, lvepi -> 2
    def __getitem__(self,index):
        # init Transform 
        Transform = {
            'train': T.Compose([
                T.ToPILImage(),
                T.Resize(self.resize_to),
                Rescale_Intensity(),
                T.ToTensor(),
            ]),
            'seg': T.Compose([
                T.ToPILImage(),
                T.Resize(self.resize_to,T.InterpolationMode.BILINEAR),
                T.ToTensor(),
            ]),
            'vali': T.Compose([
                T.ToPILImage(),
                T.Resize(self.resize_to),
                Rescale_Intensity(),
                T.ToTensor(),
            ]),
        }
        Transform_train = Transform['train']
        Transform_seg = Transform['seg']
        assert self.with_seg[index]['has_seg'], "Oh no! This sample dose not have segmentations!"
        tmp_img_list = ['{}{}/SAX/0005'.format(self.with_seg[index]['img_dir'],j) for j in [self.with_seg[index]['seg_tps'][0],self.with_seg[index]['seg_tps'][-1]]]
        tmp_seg_list = ['{}{}/masks/{}/0005.png'.format(self.with_seg[index]['gt_dir'],j, k)  for j in [self.with_seg[index]['seg_tps'][0],self.with_seg[index]['seg_tps'][-1]] for k in ['RVendo','LVendo','LVepi'] ]
        r_min, r_max, c_min, c_max = getROI(tmp_seg_list[3],tmp_seg_list[5])
        img = torch.cat([Transform_train(np.float32(dicom.read_file(i).pixel_array[r_min-10:r_max+10,c_min-10:c_max+10])) for i in tmp_img_list],0)
        seg = torch.cat([Transform_seg(np.float32(fill_seg(imageio.imread(i))[r_min-10:r_max+10,c_min-10:c_max+10]/255.)) for i in tmp_seg_list],0)
        seg = self.make_seg_gt(seg)
        info = self.with_seg[index]
        return img, seg, info

    def __len__(self):
        return len(self.with_seg)

class UKB_SAX(data.Dataset):
    def __init__(self,root_gt:str,root_dicom:str,mode='train',num_of_frames:int=6):
        super(UKB_SAX,self).__init__()
        self.root_gt = root_gt
        self.root_dicom = root_dicom
        self.num_of_frames = num_of_frames
        self.resize_to = TRAIN_RESIZE
        self.no_seg = []
        self.with_seg = []
        
        # Check for pre-saved id list 
        if os.path.isfile('./patient_ids.npy'):
            patient_ids = np.load('./patient_ids.npy')
        # Set reload flag as TRUE 
            reload_flg = True 
        else:
            # get patients' ID from root (/dataset)
            patient_ids = os.listdir(self.root_gt)
        print("Total number of patients is: {}...".format(len(patient_ids)))
        # walk through all patient's folder, remove sample if it dose not contain gt segmentation.
        no_seg = []
        with_seg= []
        patient_ids_new = []
        for i in patient_ids:
            if reload_flg:
                # Get gt frame number
                # REF gt_dir/image&mask/time/masks/RVendo/LVendo/LVepi
                gt_dir = "{}{}/image&mask/".format(self.root_gt,i)
                seg_tps = os.listdir(gt_dir)
                # under cine, image is DICOM 
                # REF img_dir/id/time00x/SAX/0005
                img_dir = "{}{}/".format(self.root_dicom,i)
                patient_id = i 
                seg = True 
                indexs = get_indexs(seg_tps,self.num_of_frames)
                instance_dict = {'id':patient_id, 'has_seg':seg, 'img_dir':img_dir, 'gt_dir':gt_dir, 'seg_tps':seg_tps, 'indexs':indexs}
                self.with_seg.append(instance_dict)
                patient_ids_new.append(i)
            else:
                # Check if both dicom and segs are exist
                has_dicom = os.path.isdir('{}{}'.format(self.root_dicom,i))
                if  has_dicom:
                    # Get gt frame number
                    # REF gt_dir/image&mask/time/masks/RVendo/LVendo/LVepi
                    gt_dir = "{}{}/image&mask/".format(self.root_gt,i)
                    seg_tps = os.listdir(gt_dir)
                    # under cine, image is DICOM 
                    # REF img_dir/id/time00x/SAX/0005
                    img_dir = "{}{}/".format(self.root_dicom,i)
                    patient_id = i 
                    if len(seg_tps) == 2 and len(os.listdir(img_dir)) >= 50:
                        # Assume all patient has 50 frames of MRI scans
                        seg = True 
                        indexs = get_indexs(seg_tps,self.num_of_frames)
                        instance_dict = {'id':patient_id, 'has_seg':seg, 'img_dir':img_dir, 'gt_dir':gt_dir, 'seg_tps':seg_tps, 'indexs':indexs}
                        if readable_dicom(instance_dict) and readable_png(instance_dict):
                            self.with_seg.append(instance_dict)
                            patient_ids_new.append(i)
                else:
                    print('For case {}, root_dicom: {}...'.format(i, has_dicom))
        print('The number of no-seg samples is: {}...'.format(len(self.no_seg)))
        print('The total number of seg samples is: {}...'.format(len(self.with_seg)))
        if not os.path.isfile('./patient_ids.py'):
            print('Saving clean patient id list...')
            np.save('./patient_ids.npy',patient_ids_new)
        if mode == 'train':
            self.with_seg = self.with_seg[0:4600]
        if mode == 'test':
            self.with_seg = self.with_seg[4600:4690]
            
    def make_seg_gt(self,segs):
        # INPUT: 6*H*W 
        # OUTPUT: 2*4*H*W 
        _,H,W = segs.shape
        # background = ones - rvendo - lvepi
        br_ed = torch.ones(1,H,W,dtype=torch.long) - segs[0,...] - segs[2,...]
        br_es = torch.ones(1,H,W,dtype=torch.long) - segs[3,...] - segs[5,...]
        # subtract lvendo from lvepi to form lvepi segs 
        segs[2,...] = segs[2,...] - segs[1,...]
        segs[-1,...] = segs[-1,...] - segs[-2,...]
        # saperate segs for ed and es phases
        ed_seg = segs[0:3,...]
        ed_seg = torch.cat([segs[0:3,...],br_ed],dim=0)
        es_seg = segs[3:6,...]
        es_seg = torch.cat([segs[3:6,...],br_es],dim=0)
        out = torch.stack([ed_seg,es_seg],dim=0)
        out = torch.clamp(out,min=0.,max=1.)
        return out
        # make label maps for cross entropy. dtype should be int64
        # background -> 3, rvendo -> 0, lvendo -> 1, lvepi -> 2
    def __getitem__(self,index):
        # init Transform 
        Transform = {
            'train': T.Compose([
                T.ToPILImage(),
                T.Resize(self.resize_to),
                Rescale_Intensity(),
                T.ToTensor(),
            ]),
            'seg': T.Compose([
                T.ToPILImage(),
                T.Resize(self.resize_to),
                T.ToTensor(),
            ]),
            'vali': T.Compose([
                T.ToPILImage(),
                T.Resize(self.resize_to),
                Rescale_Intensity(),
                T.ToTensor(),
            ]),
        }
        Transform_train = Transform['train']
        Transform_seg = Transform['seg']
        assert self.with_seg[index]['has_seg'], "Oh no! This sample dose not have segmentations!"
        tmp_img_list = ['{}time{}/SAX/0005'.format(self.with_seg[index]['img_dir'],"%03d" %(j)) for j in self.with_seg[index]['indexs']]
        tmp_seg_list = ['{}{}/masks/{}/0005.png'.format(self.with_seg[index]['gt_dir'],j, k)  for j in [self.with_seg[index]['seg_tps'][0],self.with_seg[index]['seg_tps'][-1]] for k in ['RVendo','LVendo','LVepi'] ]
        r_min, r_max, c_min, c_max = getROI(tmp_seg_list[3],tmp_seg_list[5])
        img = torch.cat([Transform_train(np.float32(dicom.read_file(i).pixel_array[r_min-10:r_max+10,c_min-10:c_max+10])) for i in tmp_img_list],0)
        seg = torch.cat([Transform_seg(np.float32(fill_seg(imageio.imread(i))[r_min-10:r_max+10,c_min-10:c_max+10]/255.)) for i in tmp_seg_list],0)
        seg = self.make_seg_gt(seg)
        info = self.with_seg[index]
        return img, seg, info

    def __len__(self):
        return len(self.with_seg)
        

def get_loader(root_gt, root_dicom, axis='lax',mode='train'):
    if axis == 'sax':
        dataset = UKB_SAX_EDES(root_gt=root_gt, root_dicom=root_dicom,mode=mode,num_of_frames=10)
        print('Len(dataset): {}'.format(len(dataset)))
    if mode == 'train':
        data_loader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    else:
        data_loader = data.DataLoader(
            dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return data_loader

# data_dir = '/usr/not-backed-up/scnb/data/masks_sax_5k/'
# data_dicom = '/usr/not-backed-up/scnb/data/dicom_lsax_5k/'
# train_loader = get_loader(root_gt=data_dir,root_dicom=data_dicom, axis='sax',mode='train')
# for ind, item in enumerate(train_loader):
#     print(item[0].shape)
#     print(torch.min(item[1][0:1,0,3,...]))
#     print(torch.max(item[1][0:1,0,3,...]))
#     # save_segs(img=item[1][0:1,...],id='gt_x_{}'.format(ind), cmap='coolwarm')
#     break

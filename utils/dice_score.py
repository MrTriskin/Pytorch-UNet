import torch 
import torch.nn.functional as F 
import numpy as np 
from sklearn import metrics
from medpy import metric
def area_coverage(seg, is_gt:bool = False):
    seg = seg.cpu().detach()
    ac = torch.empty(3,dtype=torch.float32)
    assert len(seg.shape) == 3 and seg.size(0) == 4, 'Make sure input with size C,H,W'
    if not is_gt:
        seg = torch.argmax(torch.softmax(seg,dim=0),dim=0,keepdim=True)
        for i in range(3):
            ac[i] = torch.sum(seg == i)
            # print(ac[i])
    else:
        seg = seg > 0.
        ac = torch.sum(seg,dim=(1,2))[0:3]
    return(ac)

def HD(pred,ref,connectivity:int=3):
    assert ref.shape == pred.shape, 'Make sure two inputs are in same shapes bro :)' 
    pred_np = (255*pred).cpu().detach().numpy().astype(np.uint8)
    ref_np = (255*ref.cpu()).detach().numpy().astype(np.uint8)
    hd_val = metric.binary.hd(ref_np,pred_np,connectivity=connectivity)
    return(hd_val)
# def HD(pred,ref):
#     assert ref.shape == pred.shape, 'Make sure two inputs are in same shapes bro :)' 
#     pred_np = (pred*255).cpu().detach().numpy().astype(np.uint8)
#     ref_np = ref.cpu().detach().numpy().astype(np.uint8)
#     idx_ref = np.squeeze(cv2.findContours(ref_np,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0][0])
#     idx_pred = np.squeeze(cv2.findContours(pred_np,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0][0])
#     hd_sym = spatial.distance.directed_hausdorff(idx_pred,idx_ref)[0]
#     print(hd_sym)
#     return(hd_sym)
def NMI(true_label,pred):
    assert true_label.shape == pred.shape, 'Make sure two inputs are in same shapes bro :(' 
    pred_np = pred.view(-1).cpu().detach().numpy()
    truel_np = true_label.view(-1).cpu().detach().numpy()
    nmi = metrics.normalized_mutual_info_score(truel_np,pred_np)
    return(nmi)

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0
    i_flat = prediction.reshape(-1)
    t_flat = target.reshape(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))

def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    assert prediction.shape == target.shape , 'Make sure two inputs are with same shapes.'
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    # bce = 0.0
    if prediction.size(0) == 1: # *when testing B = 1 input is C,H,W
        prediction = F.sigmoid(prediction)
    else:
        prediction = torch.softmax(prediction,dim=1)
    # *calculate DICE loss regardless backgroud (the last channel)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss

def jacobian_determinant(disp):
    """
    Input: displacement field with size [C,H,W] (numpy)
    Output: jacombian determinant [H,W]
    """
    # check inputs
    if type(disp).__module__ != 'numpy':
        dt = disp.cpu().detach().numpy()
    else:
        dt = disp
    _,H,W = disp.shape
    
    x,y = np.meshgrid(np.linspace(0,W-1,W),np.linspace(H-1,0,H))
    grids = np.stack([x,y],axis=0)
    # compute gradients
    J = np.gradient(dt + grids )

    dfdx = J[0]
    dfdy = J[1]
    return dfdx[0,...] * dfdy[1,...] - dfdy[0,...] * dfdx[1,...]
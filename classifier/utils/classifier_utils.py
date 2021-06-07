import os, time, copy, pickle
import os.path as osp 
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import PIL

from utils import AverageMeter


### CONSTANT
IMAGE_SIZE = (224,224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, PIL.Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def preprocess_input(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert('RGB')
    img = val_transform(img)
    return img

def iou_dice_score(pred, target):
    batch_size = pred.shape[0]
    inp = pred.view(batch_size, -1)
    target = target.view(batch_size, -1)
    inter = (inp*target).sum(dim=-1)
    union = (inp+target).sum(dim=-1) - inter 
    iou = (inter + 1)/(union + 1)
    dice = (2*inter + 1)/(union + inter + 1)

    return torch.mean(iou), torch.mean(dice)

def evaluate_tensor(y_pred, y_true, thres=0.5):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    y_pred = (y_pred >= thres).astype(np.int)

    return accuracy_score(y_true, y_pred)

def evaluate_fraction(y_pred, y_true, dist=1/3, thres=0.1, top_k=2):
    max_val = torch.max(y_pred, dim=1).values
    r, c = y_pred.shape
    min_val = torch.max(max_val - dist, torch.tensor([thres]*r).cuda())

    y_pred_idx = torch.topk(y_pred, k=top_k, dim=1).indices
    new_y_pred = torch.gather(y_pred, 1, y_pred_idx)
    new_y_pred = (new_y_pred >= torch.reshape(min_val, (r, -1))).float()

    r, c = new_y_pred.shape
    count_non_zero = c - (new_y_pred != 0).sum(dim=1)

    for i in range(len(y_pred_idx)):
        if count_non_zero[i]:
            y_pred_idx[i][-1] = y_pred_idx[i][0]

    new_y_true = torch.gather(y_true, 1, y_pred_idx)
    new_y_true = (new_y_true>0).float()

    return float((torch.sum(torch.mean(new_y_true, axis=-1))/r))

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, save_path=None):
    since = time.time()

    val_acc_history, train_acc_history = [], []
    val_loss_history, train_loss_history = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    step_count = 0

    trackers = {
        'train_acc': AverageMeter(), 'val_acc': AverageMeter(),
        'train_dice': AverageMeter(), 'val_dice': AverageMeter(),
        'train_iou': AverageMeter(), 'val_iou': AverageMeter(),
    }
    for epoch in range(num_epochs):
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0
            for k in trackers:
                trackers[k].reset()

            loader = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]))
            for it, data in loader:
                inputs = data['img'].cuda()
                labels = data['label'].cuda()

                batch_size = inputs.shape[0]
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                it_acc = evaluate_fraction(outputs, labels)
                it_iou, it_dice = iou_dice_score(outputs, labels)
                trackers[f'{phase}_acc'].update(it_acc, n=batch_size)
                trackers[f'{phase}_iou'].update(it_iou, n=batch_size)
                trackers[f'{phase}_dice'].update(it_dice, n=batch_size)

            # Print after each epoch
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = trackers[f'{phase}_iou'].avg
            
            print(f"[{phase}] Loss: {epoch_loss:.4f}, Acc: {trackers[f'{phase}_acc'].avg:.4f}, Iou: {trackers[f'{phase}_iou'].avg:.4f}, Dice: {trackers[f'{phase}_dice'].avg:.4f}, Lr: {optimizer.param_groups[0]['lr']}")
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if save_path is not None:
                        save_model_path = osp.join(save_path, "best_model.pt")
                        torch.save(model.state_dict(), save_model_path)
                        print(f"save best model at {epoch}")
                    else:
                        step_count += 1
                
                val_acc_history.append(epoch_acc)
                scheduler.step(epoch_acc)

    torch.save(model.state_dict(), osp.join(save_path, f'last_model.pt'))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

@torch.no_grad()
def get_feat_from_subject_box(crop, veh_model, col_model):
    crop = preprocess_input(Image.fromarray(crop).convert('RGB'))
    veh_feat = veh_model.extract_feature(crop.unsqueeze(0).cuda())
    col_feat = col_model.extract_feature(crop.unsqueeze(0).cuda())

    veh_feat = veh_feat.squeeze().cpu()
    col_feat = col_feat.squeeze().cpu()

    feat = torch.cat((veh_feat, col_feat), axis=0)
    return feat

@torch.no_grad()
def get_feat_from_model(list_crops, model):
    list_tensors = [preprocess_input(Image.fromarray(crop).convert('RGB')) for crop in list_crops]
    images = torch.stack(list_tensors, dim=0).cuda()

    return model.extract_feature(images)

def scan_data(data_loader, name=None):
    if name is not None:
        print(f'Scan {name} dataset')

    loader = tqdm(data_loader, total=len(data_loader))
    for sample in loader:
        pass
    pass

def pickle_save(data, save_path, verbose=True):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    if verbose:
        print(f'save result to {save_path}')

def pickle_load(save_path, verbose=False):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    
    if verbose:
        print(f'load result from {save_path}')
    return data
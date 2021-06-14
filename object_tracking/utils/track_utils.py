import numpy as np 


def get_veh_detection(preds, veh_thres=0.8):
    preds = np.mean(preds, axis=0)
    preds = (preds >= veh_thres).astype(np.int)
    return preds

def get_col_detection(preds, col_thres=0.8):
    preds = preds[preds >= col_thres]
    return preds
    
def get_valid_coor(x, delta, xmax, xmin=0):
    x_new = x+delta
    x_new = min(xmax, max(xmin, x_new))
    return x_new

def expand_box_area(box: list, W: int, H: int, ratio: float=0.1):
    # xyxy 
    x1, y1, x2, y2 = box 
    w, h = x2-x1, y2-y1
    exp_w = w*ratio
    exp_h = h*ratio

    new_x1 = int(get_valid_coor(x1, -exp_w, W-1, 1))
    new_x2 = int(get_valid_coor(x2, exp_w, W-1, 1))
    new_y1 = int(get_valid_coor(y1, -exp_h, H-1, 1))
    new_y2 = int(get_valid_coor(y2, exp_h, H-1, 1))

    return [new_x1, new_y1, new_x2, new_y2]

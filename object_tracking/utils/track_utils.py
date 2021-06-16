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

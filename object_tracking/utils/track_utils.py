import numpy as np 


def get_veh_detection(preds, veh_thres=0.8):
    preds = np.mean(preds, axis=0)
    preds = (preds >= veh_thres).astype(np.int)
    return preds

def get_col_detection(preds, col_thres=0.8):
    preds = preds[preds >= col_thres]
    return preds
    


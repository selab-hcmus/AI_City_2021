import numpy as np

def filter_track_veh_preds(preds, veh_thres=0.8, weight=None):
    list_preds = (preds >= veh_thres).astype(np.int)

    n_box = preds.shape[0]
    if weight is None:
        weight = np.ones(n_box, dtype=np.float)
    
    final_preds = np.sum(preds*weight[:, None], axis=0)/np.sum(weight)
    
    flag = None
    for thres in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        temp =  (final_preds >= thres).astype(np.int)
        is_all_zero = np.all((temp == 0))
        if not is_all_zero:
            flag = thres
            break

    if flag is None:
        final_preds = (final_preds > 1.0).astype(np.int)
    else:
        final_preds = (final_preds >= flag).astype(np.int)
    
    return list_preds.tolist(), final_preds.tolist(), flag

def filter_track_col_preds(preds, col_thres=0.4, weight=None):
    list_preds = (preds >= col_thres).astype(np.int)
    n_box = preds.shape[0]
    if weight is None:
        weight = np.ones(n_box, dtype=np.float)/n_box
    
    # print(f'weight')
    # print(weight)
    # print(f'preds')
    # print(list_preds)
    final_pred = np.sum(preds*weight[:, None], axis=0)/np.sum(weight)
    # print(f'final pred')
    # print(final_preds)
    
    thres_final_pred = (final_pred >= col_thres).astype(np.int)
    return list_preds.tolist(), final_pred.tolist(), thres_final_pred.tolist()

def split_data(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    ans = []
    for arr in out:
        ans.append(arr[len(arr)//2])
    return ans

def get_class_name(pred: list, class_map: dict):
    names = []
    for i, val in enumerate(pred):
        if val == 1:
            names.append(class_map[i])

    if len(names) == 0:
        return 'fail'

    return '_'.join(names)
    
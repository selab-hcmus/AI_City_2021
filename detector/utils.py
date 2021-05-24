import numpy as np 

def cal_distance(box_a, box_b):
    xa, ya, _, _ = box_a
    xb, yb, _, _ = box_b
    return np.sqrt((xa-xb)**2 + (ya-yb)**2)

def xyxy_to_xywh(coor):
    x1, y1, x2, y2 = coor
    return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]

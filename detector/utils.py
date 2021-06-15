import numpy as np 

class Point(object):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x 
        self.y = y

def cal_distance(box_a, box_b):
    xa, ya, _, _ = box_a
    xb, yb, _, _ = box_b
    return np.sqrt((xa-xb)**2 + (ya-yb)**2)

def xyxy_to_xywh(coor):
    x1, y1, x2, y2 = coor
    return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]

def cross(a: Point, b: Point):
    return a.x*b.y - a.y*b.x
    
def euclid_distance(a: Point, b: Point):
    return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def algebra_area(a: Point, b: Point, c: Point):
    return (cross(a, b) + cross(b, c) + cross(c, a))/2.0


 

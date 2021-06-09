import numpy as np
import os 
import os.path as osp
from tqdm import tqdm

from object_tracking.library.track_result import TrackResult
from object_tracking.library import VideoResult
from object_tracking.test.test_utils import calculate_iou, calculate_distance

from utils import AverageMeter, json_save, json_load


def get_center_point(box: list):
    # xyxy
    return (box[0] + box[1]) / 2, (box[2] + box[3]) / 2

def check_wrong_boxes(track_data: TrackResult):
    dist_meter = AverageMeter()
    iou_meter = AverageMeter()
    list_boxes = track_data.boxes
    
    for i in range(1, len(list_boxes)):
        dist = calculate_distance(list_boxes[i], list_boxes[i-1])
        iou = calculate_iou(list_boxes[i], list_boxes[i-1])
        
        if dist/dist_meter.avg > 6 and calculate_iou(list_boxes[i], list_boxes[i-1]) < 0.1:
            pass
        pass
    pass

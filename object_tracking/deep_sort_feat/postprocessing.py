import numpy as np
from .utils import AverageMeter, calculate_box_area

def is_track_with_extreme_box(track_data: dict):
    area_meter = AverageMeter() 
    for i, box in enumerate(track_data['boxes']):
        box_area = calculate_box_area(box)
        if i != 0:
            if box_area > 4*area_meter.avg:
                return True
        area_meter.update(box_area)

    return False

def remove_frame_with_extreme_box(list_boxes):
    prev_area = calculate_box_area(list_boxes[0])
    new_boxes = [list_boxes[0]]
    is_fail = False
    for i, box in enumerate(list_boxes[1:]):
        cur_area = calculate_box_area(box)
        if cur_area > 3*prev_area:
            new_boxes.append(None)
            is_fail=True
        else:
            prev_area = cur_area
        
    return new_boxes, is_fail

def remove_track_with_extreme_box(vid_data: dict):
    new_data = vid_data.copy()
    new_data['track_map'] = {}
    fail_ids = []
    for track_id in vid_data['track_map'].keys():
        new_boxes, is_fail = remove_frame_with_extreme_box(vid_data['track_map'][track_id]['boxes'])
        
        # if is_track_with_extreme_box(vid_data['track_map'][track_id]):
        if is_fail:
            fail_ids.append(track_id)
        else:
            new_data['track_map'][track_id] = vid_data['track_map'][track_id]
    
    return new_data, fail_ids

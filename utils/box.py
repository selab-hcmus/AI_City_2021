def xywh_to_xyxy(box):
    return [box[0], box[1], box[2] + box[0], box[3] + box[1]]

def xyxy_to_xywh(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

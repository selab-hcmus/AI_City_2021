from . import preprocessing as prep
from . import nn_matching
from .tracker import Tracker 
from .detection import Detection

import numpy as np

class TrackingManager(object):
    def __init__(self, cfg: dict):
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            cfg['METRIC']['NAME'], cfg['METRIC']['THRESHOLD'], cfg['METRIC']['BUDGET']
        )

        self.tracker= Tracker(
            self.metric, cfg['TRACKER']['MAX_IOU_DISTANCE'], cfg['TRACKER']['MAX_AGE'], cfg['TRACKER']['N_INIT']
        )
	

    def run_deep_sort(self, out_scores, out_boxes, features):
        if out_boxes==[]:			
            self.tracker.predict()
            print('No detections')
            trackers = self.tracker.tracks
            return trackers

        detections = np.array(out_boxes)
        dets = [Detection(bbox, score, feature) \
                for bbox, score, feature in zip(detections, out_scores, features)]

        outboxes = np.array([d.tlwh for d in dets])
        # outscores = np.array([d.confidence for d in dets])

        indices = prep.non_max_suppression(outboxes, 0.9, None)
        dets = [dets[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(dets)	

        return self.tracker, dets
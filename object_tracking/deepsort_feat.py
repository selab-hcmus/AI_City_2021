from object_tracking.deep_sort_feat import nn_matching
from object_tracking.deep_sort_feat.tracker import Tracker 
import object_tracking.deep_sort_feat.preprocessing as prep
from object_tracking.deep_sort_feat.detection import Detection


import numpy as np


class deepsort_rbc():
	def __init__(self):
		self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.3, 70)
		self.tracker= Tracker(self.metric, max_age=10, n_init=1)

		# self.gaussian_mask = get_gaussian_mask().cuda()
        
	def reset_tracker(self):
		# self.tracker= Tracker(self.metric, max_age=12, n_init=1)			
		self.tracker= Tracker(self.metric, max_age=4, n_init=1)			

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
		outscores = np.array([d.confidence for d in dets])

		indices = prep.non_max_suppression(outboxes, 0.8, None)
		dets = [dets[i] for i in indices]

		self.tracker.predict()
		self.tracker.update(dets)	

		return self.tracker, dets

from deep_sort import nn_matching
from deep_sort.tracker import Tracker 
import deep_sort.preprocessing as prep
from deep_sort.detection import Detection

import numpy as np


class deepsort_rbc():
	def __init__(self):
		self.metric = nn_matching.NearestNeighborDistanceMetric("cosine",.5 , 100)
		self.tracker= Tracker(self.metric)

		# self.gaussian_mask = get_gaussian_mask().cuda()
        
	def reset_tracker(self):
		self.tracker= Tracker(self.metric)			

	def run_deep_sort(self, out_scores, out_boxes, features):
		if out_boxes==[]:			
			self.tracker.predict()
			print('No detections')
			trackers = self.tracker.tracks
			return trackers

		detections = np.array(out_boxes)
		dets = [Detection(bbox, score, feature) \
					for bbox, score, feature in \
						zip(detections,out_scores, features)]

		outboxes = np.array([d.tlwh for d in dets])
		outscores = np.array([d.confidence for d in dets])

		indices = prep.non_max_suppression(outboxes, 0.8, outscores)
		dets = [dets[i] for i in indices]

		self.tracker.predict()
		self.tracker.update(dets)	

		return self.tracker, dets

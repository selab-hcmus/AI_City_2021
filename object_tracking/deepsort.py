from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker 
from deep_sort.application_util import preprocessing as prep
from deep_sort.application_util import visualization
from deep_sort.deep_sort.detection import Detection

import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision
from scipy.stats import multivariate_normal

# def get_gaussian_mask():
#     # TODO: try image size = 224
# 	# 128 is image size
# 	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
# 	xy = np.column_stack([x.flat, y.flat])
# 	mu = np.array([0.5,0.5])
# 	sigma = np.array([0.22,0.22])
# 	covariance = np.diag(sigma**2) 
# 	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
# 	z = z.reshape(x.shape) 

# 	z = z / z.max()
# 	z  = z.astype(np.float32)

# 	mask = torch.from_numpy(z)

# 	return mask


class deepsort_rbc():
	def __init__(self,wt_path=None):
		self.metric = nn_matching.NearestNeighborDistanceMetric("cosine",.5 , 100)
		self.tracker= Tracker(self.metric)

		# self.gaussian_mask = get_gaussian_mask().cuda()
        
	def reset_tracker(self):
		self.tracker= Tracker(self.metric)

	#Deep sort needs the format `top_left_x, top_left_y, width,height
	
	def format_yolo_output(self,out_boxes):
		for b in range(len(out_boxes)):
			out_boxes[b][0] = out_boxes[b][0] - out_boxes[b][2]/2
			out_boxes[b][1] = out_boxes[b][1] - out_boxes[b][3]/2
		return out_boxes				

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

## Object Tracking
Installation: Use this command to install all the necessary packages. Note that we are using ```python3```

Link to the blog [click here](https://blog.nanonets.com/object-tracking-deepsort/)
```sh
pip install -r requirements.txt
```
This module is built on top of the original deep sort module https://github.com/nwojke/deep_sort
Since, the primary objective is to track objects, We assume that the detections are already available to us, for the given video. The   ``` det/``` folder contains detections from Yolo, SSD and Mask-RCNN for the given video.

```deepsort.py``` is our bridge class that utilizes the original deep sort implementation, with our custom configs. We simply need to specify the encoder (feature extractor) we want to use and pass on the detection outputs to get the tracked bounding boxes. 
```test_on_video.py``` is our example code, that runs deepsort on a video whose detection bounding boxes are already given to us. 

# A simplified overview:
```sh
#Initialize deep sort object.
deepsort = deepsort_rbc(wt_path='ckpts/model640.pt') #path to the feature extractor model.

#Obtain all the detections for the given frame.
detections,out_scores = get_gt(frame,frame_id,gt_dict)

#Pass detections to the deepsort object and obtain the track information.
tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections)

#Obtain info from the tracks.
for track in tracker.tracks:
    bbox = track.to_tlbr() #Get the corrected/predicted bounding box
    id_num = str(track.track_id) #Get the ID for the particular track.
    features = track.features #Get the feature vector corresponding to the detection.
```
The ```tracker``` object returned by deepsort contains all necessary info like the track_id, the predicted bounding boxes and the corresponding feature vector of the object. 

## Object Tracking module

Output result format:
- train result: ./results/annotate_time_train
- test result: ./results/annotate_time_test
Each video track is saved as a json file: `<track_order>.json`.
- `list_frames`: list of frame paths in this track
- `n_frames`: number of frames
- `n_tracks`: number of detected tracks
- `frame_ids`:  annotate id for each frame in `list_frames`, s.t: `len(list_frames) == len(list_frames)`
- `track_map`: dictionary contains all track info: 
    - key: id of track
    - `frame_order`: list of frame_id (must appear in `frame_ids`)
    - `boxes`: list of boxes for each frame, s.t: `len(boxes) == len(frame_order)`

import cv2 
import os 
import os.path as osp 

GREEN = (0,255,0)
WHITE = (255,255,255)

def visualize(json_data: dict, list_track_ids: list, data_dir: str, vid_save_path: str):
    list_frames = [] 

    for frame_name in json_data['list_frames']:
        cv_frame = cv2.imread(osp.join(data_dir, frame_name))
        list_frames.append(cv_frame)
        pass
    
    vid_height, vid_width, _ = list_frames[0].shape
    if list_track_ids == None:
        list_track_ids = list(json_data['track_map'].keys())
        
    for track_id in list_track_ids:
        track_data = json_data['track_map'][track_id]
        for i in range(len(track_data['boxes'])):
            frame_order = track_data['frame_order'][i]
            bbox = track_data['boxes'][i]
            cv_frame = list_frames[frame_order]

            cv2.rectangle(cv_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),WHITE, 2)
            cv2.putText(cv_frame, str(track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, GREEN,2)

        pass
    pass

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_writer = cv2.VideoWriter(vid_save_path,fourcc, 1, (vid_width, vid_height))
    for cv_frame in list_frames:
        vid_writer.write(cv_frame)
        pass
    vid_writer.release()
    pass

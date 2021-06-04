import cv2 
import os 
import os.path as osp 

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
WHITE = (255,255,255)


def visualize_subject(
    json_data: dict, list_track_ids: list, data_dir: str, vid_save_path: str, subject_boxes: list
):
    list_frames = [] 
    for frame_name in json_data['list_frames']:
        cv_frame = cv2.imread(osp.join(data_dir, frame_name))
        list_frames.append(cv_frame)
        
    
    vid_height, vid_width, _ = list_frames[0].shape
    if list_track_ids == None:
        list_track_ids = list(json_data['track_map'].keys())
        if json_data['subject'] is not None:
            list_track_ids.remove(json_data['subject'])
        pass
    
    for i, cv_frame in enumerate(list_frames):
        if i == len(subject_boxes):
            break
        bbox = subject_boxes[i]
        cv2.rectangle(cv_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),RED, 2)
        cv2.putText(cv_frame, 'sb',(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, RED,2)
        pass

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



def visualize(json_data: dict, list_track_ids: list, data_dir: str, vid_save_path: str, data_handler: dict, vid_name = "results"):
    list_frames = [] 

    for frame_name in json_data['list_frames']:
        cv_frame = cv2.imread(osp.join(data_dir, frame_name))
        list_frames.append(cv_frame)
        pass
    
    vid_height, vid_width, _ = list_frames[0].shape
    if list_track_ids == None:
        list_track_ids = list(json_data['track_map'].keys())
        
    for track_id in list_track_ids:
        caption = None
        if track_id in data_handler["subject"]:
            caption = "A"
        elif track_id in data_handler["follow"]:
            caption = "A_follow_B"
        elif track_id in data_handler["follow_by"]:
            caption = "B_follow_A"
        
        if caption is None:
            continue

        track_data = json_data['track_map'][track_id]
        for i in range(len(track_data['boxes'])):
            frame_order = track_data['frame_order'][i]
            bbox = track_data['boxes'][i]
            cv_frame = list_frames[frame_order]

            cv2.rectangle(cv_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),WHITE, 2)
            cv2.putText(cv_frame, str(track_id) + "_" + caption,(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, RED,2)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_writer = cv2.VideoWriter(osp.join(vid_save_path, f"{vid_name}.avi"), fourcc, 1, (vid_width, vid_height))
    for cv_frame in list_frames:
        vid_writer.write(cv_frame)

    vid_writer.release()
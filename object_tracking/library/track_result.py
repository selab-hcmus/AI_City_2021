import cv2 
import os 
import os.path as osp
import numpy as np

from utils.data_manager import DATA_DIR
from utils import dict_save, dict_load
# from object_tracking.utils import expand_box_area

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
WHITE = (255,255,255)
YELLOW = (0, 255, 255)

from utils import create_logger
debug_logger = create_logger(
    '/home/ntphat/projects/AI_City_2021/results/object_tracking/class_name.log', 
    stdout=False
)

def get_valid_coor(x, delta, xmax, xmin=0):
    x_new = x+delta
    x_new = min(xmax, max(xmin, x_new))
    return x_new

def expand_box_area(box: list, W: int, H: int, ratio: float=0.1):
    # xyxy 
    x1, y1, x2, y2 = box 
    w, h = x2-x1, y2-y1
    exp_w = w*ratio
    exp_h = h*ratio

    new_x1 = int(get_valid_coor(x1, -exp_w, W-1, 1))
    new_x2 = int(get_valid_coor(x2, exp_w, W-1, 1))
    new_y1 = int(get_valid_coor(y1, -exp_h, H-1, 1))
    new_y2 = int(get_valid_coor(y2, exp_h, H-1, 1))

    return [new_x1, new_y1, new_x2, new_y2]


class TrackResult(object):
    def __init__(self, track_id: str, track_info: dict) -> None:
        super().__init__()
        self.track_id = track_id
        self.frame_order = track_info['frame_order']
        self.first_frame = track_info['frame_order'][0]
        self.last_frame = track_info['frame_order'][-1]
        self.boxes = track_info['boxes']
        # self.vehicle_type = track_info['vehicle_type']
        # self.color = track_info['color']
        self.cv_boxes = None
        self.features = []
        self.features = track_info.get('features')
        self.is_subject=False
        self.final_vehicle = None
        self.final_color = None
        self.flag_thres = None
        pass
    pass

    def get_boxes_data(self, list_frames: list, list_ids=None, expand_ratio=None):
        cv_boxes = []
        if list_ids is None:
            list_ids = range(len(self.frame_order))

        for i in list_ids:
            order = self.frame_order[i]
            fpath = osp.join(DATA_DIR, list_frames[order])
            cv_img = cv2.imread(fpath)
            H, W, C = cv_img.shape
            box = self.boxes[i] #xyxy
            box = [int(j) for j in box]
            
            if expand_ratio is not None:
                box = expand_box_area(box, W, H, expand_ratio)
            
            cv_box = cv_img[box[1]:box[3], box[0]: box[2], :]
            cv_boxes.append(cv_box)

        return cv_boxes
        
    def set_veh_class(self, list_frames, classifier_manager, thres=0.7):
        AREA_THRES = 10 #pixels
        EXPAND_BOX_RATIO = 0.1
        N = len(self.boxes)
        ids_to_use = sorted(list(set([0, N//6, 2*N//6, 3*N//6, 4*N//6, 5*N//6, N-1])))
        # ids_to_use = list(set([N//10, 2*N//10, 3*N//10, 4*N//10, 5*N//10, 6*N//10, 7*N//10, 8*N//10, 9*N//10]))
        weights = np.ones(len(ids_to_use))
        
        # Set boxes used for classification
        boxes_to_use = self.get_boxes_data(list_frames, ids_to_use, EXPAND_BOX_RATIO)

        # Set box weight
        first_shape = boxes_to_use[0].shape
        last_shape = boxes_to_use[-1].shape
        S_first = first_shape[0]*first_shape[1]
        S_last = last_shape[0]*last_shape[1]


        if abs(S_first - S_last) < AREA_THRES:
            weights[0] = 3
            weights[-1] = 3
        elif S_last > S_first:
            weights[-1] = 3
        else:
            weights[0] = 3
        
        box_names, self.final_vehicle, _, _, self.flag_thres = classifier_manager.get_veh_predictions(boxes_to_use, thres, weights)
        pass 

    def set_col_class(self, list_frames, classifier_manager, thres=0.3):
        AREA_THRES = 10 #pixels
        EXPAND_BOX_RATIO = 0.1
        N = len(self.boxes)
        ids_to_use = sorted(list(set([0, N//6, 2*N//6, 3*N//6, 4*N//6, 5*N//6, N-1])))
        # ids_to_use = list(set([N//10, 2*N//10, 3*N//10, 4*N//10, 5*N//10, 6*N//10, 7*N//10, 8*N//10, 9*N//10]))
        weights = np.ones(len(ids_to_use))
        
        # Set boxes used for classification
        boxes_to_use = self.get_boxes_data(list_frames, ids_to_use, EXPAND_BOX_RATIO)

        # Set box weight
        first_shape = boxes_to_use[0].shape
        last_shape = boxes_to_use[-1].shape
        S_first = first_shape[0]*first_shape[1]
        S_last = last_shape[0]*last_shape[1]

        if abs(S_first - S_last) < AREA_THRES:
            weights[0] = 3
            weights[-1] = 3
        elif S_last > S_first:
            weights[-1] = 3
        else:
            weights[0] = 3
        
        box_names, self.final_color, preds, final_pred, thres_final_pred = classifier_manager.get_col_predictions(boxes_to_use, thres, weights)
        
        str_pred = '['
        for val in final_pred:
            str_pred += f'{val:.4f}, '
        str_pred += ']'
        debug_logger.info(f'[{self.track_id}_{thres}]: Before thres= {str_pred}, After thres= {thres_final_pred}')
        pass 

    def to_json(self, is_feat=True):
        res = {}
        res['is_subject'] = self.is_subject
        res['vehicle'] = self.final_vehicle
        res['color'] = self.final_color
        res['frame_order'] = self.frame_order
        res['boxes'] = self.boxes
        res['features'] = self.features
        if is_feat == False:
            res['features'] = []
        return res

    def get_final_classname(self):
        res = None
        if self.final_color is None and self.final_vehicle is not None:
            res = self.final_vehicle
        
        elif self.final_color is not None and self.final_vehicle is not None:
            res = f'{self.final_color}-{self.final_vehicle}'
        return res


class VideoResult(object):
    def __init__(self, inp_data) -> None:
        super().__init__()
        self._setup(inp_data)
        pass
    
    ## SETUP
    def _setup(self, inp_data):
        if isinstance(inp_data, str):
            data = dict_load(inp_data)
            vid_id = inp_data.split('.')[0].split('/')[-1]
            self.vid_id = vid_id
        elif isinstance(inp_data, dict):
            data = inp_data
            self.vid_id = None

        self.attn_mask = None
        self.list_frames = data['list_frames']
        self.n_frames = data['n_frames']
        self.frame_ids = data['frame_ids']
        self.n_tracks = data['n_tracks']
        
        self.subject = data.get('subject', None)
        self.stop_tracks = data.get('stop_tracks', [])

        self.track_map = {}
        # Init track result
        for track_id, track_info in data['track_map'].items():
            self.track_map[track_id] = TrackResult(track_id, track_info)
        
        self._set_cam_info()
        pass

    def _set_cam_info(self):
        first_frame = self.list_frames[0] # Ex: "validation/S02/c006/img1/000520.jpg"
        cv_frame = cv2.imread(osp.join(DATA_DIR, first_frame))
        H, W, C = cv_frame.shape
        cam_infos = first_frame.split('/')
        cam_id, street_id = cam_infos[2], cam_infos[1]
        self.cam_id = cam_id
        self.street_id = street_id
        self.height = H
        self.width = W
        pass

    def set_attn_mask(self, mask_path: str):
        self.attn_mask = mask_path

    def set_class_names(self, classifier_manager, veh_thres=0.6, col_thres=0.3):
        debug_logger.info('='*20 +  f' Video {self.vid_id} result ' + '='*20)
        for track_id in self.track_map:
            # self.track_map[track_id].set_boxes_data(self.list_frames)
            self.track_map[track_id].set_veh_class(self.list_frames, classifier_manager, veh_thres)
            self.track_map[track_id].set_col_class(self.list_frames, classifier_manager, col_thres)
        pass

    ## MAIN FUNCTIONS
    def _find_subject(self):
        pass 

    def _find_stop_tracks(self):
        pass 


    ## UTILITIES
    def get_list_tracks(self):
        return list(self.track_map.values())
    
    def get_list_tracks_by_time(self):
        def sort_func(x: TrackResult):
            return x.first_frame

        list_tracks = list(self.track_map.values())
        list_tracks.sort(key=sort_func)
        return list_tracks
    
    def add_new_track(self, track_data: TrackResult, track_id: str):
        if self.track_map.get(track_id) is not None:
            print(f'Track id {track_id} existed')
            return
        self.track_map[track_id] = track_data
        pass 

    def remove_track(self, track_id: str):
        self.track_map.pop(track_id, None)
        pass
    
    def set_subject(self, track_id):
        self.track_map[track_id].is_subject = True
        self.subject = track_id
        pass

    def set_stop_tracks(self, stop_tracks: list):
        self.stop_tracks = stop_tracks
        pass

    def get_subject(self):
        return self.track_map[self.subject]

    def visualize(self, save_path: str, attn_mask: np.ndarray=None):
        list_frames = []
        for fname in self.list_frames:
            fpath = osp.join(DATA_DIR, fname)
            list_frames.append(cv2.imread(fpath))

        vid_height, vid_width, _ = list_frames[0].shape
        
        for track_id in self.track_map:
            is_subject = (track_id == self.subject)
            is_stop = (track_id in self.stop_tracks)

            track_data = self.track_map[track_id]

            for i, frame_order in enumerate(track_data.frame_order):
                cv_frame = list_frames[frame_order]
                bbox = track_data.boxes[i] #xyxy

                box_name = f'{track_id}'
                vis_color = GREEN
                if track_data.get_final_classname() is not None:
                    box_name += f'_{track_data.get_final_classname()}'

                if is_subject:
                    box_name += '_sb'
                    vis_color = RED

                if is_stop:
                    box_name += '_stop'
                    vis_color = BLUE
                
                # if track_data.flag_thres is not None:
                #     box_name = str(track_data.flag_thres) + "_" + box_name
                
                cv2.rectangle(cv_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),vis_color, 2)
                cv2.putText(cv_frame, box_name,(int(bbox[0]), int(bbox[1])),0, 5e-3 * 150, vis_color, 2)

                pass
            pass

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid_writer = cv2.VideoWriter(save_path,fourcc, 1, (vid_width, vid_height))
        
        for cv_frame in list_frames:
            if attn_mask is not None:
                attn_mask[np.where(attn_mask <= 0.3)] = 0.3
                cv_frame = (cv_frame*attn_mask).astype(np.uint8)
            vid_writer.write(cv_frame)
            pass
        vid_writer.release()
        pass

    def to_json(self, save_path: str=None, is_feat=False):
        if is_feat:
            save_path = save_path.replace('.json', '.pkl')          

        res = {}
        res['list_frames'] = self.list_frames 
        res['n_frames'] = self.n_frames
        res['frame_ids'] = self.frame_ids 
        res['n_tracks'] = self.n_tracks
        res['subject'] = self.subject
        res['stop_tracks'] = self.stop_tracks
        
        res['height'] = self.height
        res['width'] = self.width
        res['cam_id'] = self.cam_id
        res['street_id'] = self.street_id
        res['attn_mask'] = self.attn_mask

        res['track_map'] = {}
        for track_id in self.track_map:
            res['track_map'][track_id] = self.track_map[track_id].to_json(is_feat)
            pass

        if save_path is not None:
            dict_save(res, save_path)
            pass
        
        return res
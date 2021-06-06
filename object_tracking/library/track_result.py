"""Class to handle tracking result

Input: pkl file created by convert_video_track (object_tracking/tools/convert_track.py)
"""

from dataset.data_manager import json_load
from utils import pickle_load

class TrackResult(object):
    def __init__(self, track_id: str, track_info: dict) -> None:
        super().__init__()
        self.track_id = track_id
        self.frame_order = track_info['frame_order']
        self.first_frame = track_info['frame_order'][0]
        self.last_frame = track_info['frame_order'][-1]
        self.boxes = track_info['boxes']
        self.vehicle_type = track_info['vehicle_type']
        self.color = track_info['color']

        self.features = []
        if track_info.get('features') is not None:
            self.features = track_info['features']
        self.is_subject=False
        pass
    pass

    def interpolate_frames(self):
        # TODO
        pass


class VideoResult(object):
    def __init__(self, save_path: str) -> None:
        super().__init__()
        self._setup(save_path)
        pass
    
    def _setup(self, save_path):
        if '.pkl' in save_path:
            data = pickle_load(save_path)
        else:
            data = json_load(save_path)

        self.list_frames = data['list_frames']
        self.n_frames = data['n_frames']
        self.frame_ids = data['frame_ids']
        self.n_tracks = data['n_tracks']
        self.track_map = {}

        self.subject = None
        if data.get('subject') is not None:
            self.subject = data['subject']
        
        # Init track result
        for track_id, track_info in data['track_map'].items():
            self.track_map[track_id] = TrackResult(track_id, track_info)
        pass

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
        for tid in self.track_map:
            if tid == track_id:
                self.track_map[tid].is_subject = True
                break
        self.subject = self.track_map[tid]
        pass

    def get_subject(self):
        return self.subject
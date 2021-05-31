## GLOBAL CONSTANT
SAVE_DIR = '/home/ntphat/projects/AI_City_2021/object_tracking/results_exp'
ID_TO_COMPARE = [5, 6, 9, 20, 34, 40, 64, 84, 182, 188, 239, 310, 339, 349, 410, 436, 476]


## HELPER FUNCTIONS

def a_substract_b(list_a: list, list_b: list):
    return [ i for i in list_a if i not in list_b]
    
def is_miss_frame(track_data: dict):
    prev = int(track_data['frame_order'][0])
    for frame_order in track_data['frame_order'][1:]:
        order = int(frame_order)
        if order - prev != 1:
            return True
        prev = order
        pass
    return False

def get_miss_frame_tracks(vid_data: dict):
    fail_tracks = []
    for track_id in vid_data['track_map']:
        if is_miss_frame(vid_data['track_map'][track_id]):
            fail_tracks.append(track_id)
        pass
    return fail_tracks
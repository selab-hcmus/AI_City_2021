from detector.utils import (
    Point,
    euclid_distance, algebra_area, xyxy_to_xywh, cal_distance
)

class TurnState():
    NO_TURN: 0
    LEFT: 1
    RIGHT: 2

class TurnDetector(object):
    def __init__(self, eps=0.036) -> None:
        super().__init__()
        self.eps = eps

    def _get_direction(self, list_points: list):
        total_area = 0
        is_turn = False
        turn_type = TurnState.NO_TURN
        
        N = len(list_points)
        if N < 3:
            return is_turn, turn_type

        src_dst_dist = euclid_distance(list_points[0], list_points[-1])
        for i in range(2, N):
            total_area += algebra_area(
                list_points[0], list_points[i-1], list_points[i]
            )
            pass
        
        norm_area = total_area/(src_dst_dist**2)
        
        if abs(norm_area) < self.eps:
            is_turn = True
            if norm_area < 0:
                turn_type = TurnState.LEFT
            else:
                turn_type = TurnState.RIGHT
        
        return is_turn, turn_type
    
    def process(self, list_boxes: list):
        N = len(list_boxes)
        list_boxes = [xyxy_to_xywh(box) for box in list_boxes]
        list_points = [Point(box[0], -box[1]) for box in list_boxes] # Reverse y axis
        
        is_turn, turn_state = self._get_direction(list_points)
        return is_turn, turn_state

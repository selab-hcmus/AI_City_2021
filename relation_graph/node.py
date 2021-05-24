import numpy as np 

class NodeState:
    APPEAR = 1
    LEAVE = 2


class Node(object):
    def __init__(self, is_subject=False):
        self.is_subject = is_subject
        self.state = None
        self.history_box = []
        
        pass

    def update(self, data_dict):
        
        pass

    pass

import os, json 
import os.path as osp 

from .caption import Caption
from utils.common import (
    get_vehicle_name_map, is_list_in_list, 
    remove_redundant_subjects, convert_to_representation_subject,
    remove_redundant_colors,
    remove_redundant_actions
)
from utils.constant import (
    VEHICLE_VOCAB
)

class Query(object):
    def __init__(self, query_content: dict, query_id: str):
        self.query_id = query_id
        self._setup(query_content) 

    def _setup(self, query_content):
        # Init captions
        self.list_caps = []
        for cap_id in query_content.keys():
            self.list_caps.append(Caption(query_content[cap_id], cap_id))

        # Find subject
        self.subjects = [cap.main_subject for cap in self.list_caps]
        
        
        self._get_list_colors()
        self._refine_subjects()
        # self._refine_colors()

        self._get_list_action()
        self._refine_list_action()

    def _get_list_action(self):
        self.actions = []
        for cap in self.list_caps:
            if len(cap.sv_format) == 0:
                continue 
            for sv in cap.sv_format:
                self.actions.append(sv['V'])

        pass
    
    def _refine_list_action(self, unique=True):
        if unique:
            self.actions = list(set(self.actions))

        # self.actions = remove_redundant_actions(self.actions)
        pass

    def get_list_captions_str(self):
        list_cap_str = [c.caption for c in self.list_caps]
        return '\n'.join(list_cap_str)

    def _get_list_colors(self):
        self.colors = []
        for cap in self.list_caps:
            self.colors.extend(list(set(cap.subject.combines)))
            
    def get_all_SV_info(self):
        sv_samples = [] 
        for cap in self.list_caps:
            for sv in cap.sv_format:
                sv_samples.append(sv)
                
        return sv_samples

    def _refine_colors(self, unique=True):
        self.colors = remove_redundant_colors(self.colors)

        # if is_list_in_list(self.colors, ['orange']):
        #     self.colors = self.colors.remove('orange')
        #     if self.colors is None or len(self.colors) == 0:
        #         self.colors = ['red']
        #     else:
        #         self.colors.append('red')
            

        # if (is_list_in_list(self.colors, ['purple'])):
        #     self.colors = self.colors.remove('purple')
        #     if self.colors is None or len(self.colors) == 0:
        #         self.colors = ['black']

        if (is_list_in_list(self.colors, ['light_gray'])):
            self.colors = self.colors.remove('light_gray')
            if self.colors is None or len(self.colors) == 0:
                self.colors = ['gray']
            else:
                self.colors.append('gray')
            

        if (is_list_in_list(self.colors, ['dark_gray'])):
            self.colors = self.colors.remove('dark_gray')
            if self.colors is None or len(self.colors) == 0:
                self.colors = ['gray']
            else:
                self.colors.append('gray')
            
        if unique:
            self.colors = list(set(self.colors))
        
        pass

    def _refine_subjects(self, unique=True):
        """Add rules to refine vehicle list for the given query
        """
        # 1. Remove redundant subjects (car, vehicle)
        self.subjects = remove_redundant_subjects(self.subjects)

        # 2. Convert all subjects to their representation name of each groups
        self.subjects = convert_to_representation_subject(self.subjects)

        if unique:
            self.subjects = list(set(self.subjects))
            
            # 3. Handle ambiguous annotations
            # [SUV, bus-truck] = [bus-truck]
            if (is_list_in_list(self.subjects, ['suv', 'bus-truck'])):
                self.subjects = ['suv', 'pickup']

            # [jeep, SUV, ...] = [Jeep, SUV]
            elif is_list_in_list(self.subjects, ['jeep', 'suv']):
                self.subjects = ['jeep', 'suv']
            
            # [jeep, pickup] = jeep
            elif is_list_in_list(self.subjects, ['jeep', 'pickup']):
                self.subjects = ['jeep']

            # [sedan, suv, van], [sedan, van] = [suv, van]
            elif (is_list_in_list(self.subjects, ['sedan', 'suv', 'van']) or 
                is_list_in_list(self.subjects, ['sedan', 'van'])
                ):
                self.subjects = ['suv', 'van']

            # [pickup, truck] = [pickup]
            elif (is_list_in_list(self.subjects, ['pickup', 'bus-truck'])):
                self.subjects = ['pickup']

            # [pickup, sedan, suv] = [pickup, suv]
            elif (is_list_in_list(self.subjects, ['sedan', 'suv', 'pickup']) or
                is_list_in_list(self.subjects, ['van', 'suv', 'pickup']) or 
                is_list_in_list(self.subjects, ['van', 'pickup'])
                ):
                self.subjects = ['suv', 'pickup']

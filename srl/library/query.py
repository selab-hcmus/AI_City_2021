from .caption import Caption
from srl.utils.common import (
    refine_list_colors, refine_list_subjects
)

class Query(object):
    def __init__(self, query_content: dict, query_id: str, query_order: str):
        self.query_id = query_id
        self.query_order = query_order
        self._setup(query_content) 

    def _setup(self, query_content):
        # 1. Init captions
        self.list_caps = []
        for cap_id in query_content.keys():
            self.list_caps.append(Caption(query_content[cap_id], cap_id))

        # 2. Find subject
        self.subject_vehicle = [cap.main_subject for cap in self.list_caps]
        self.objects = [svo['O'] for svo in self.get_all_SVO_info()]
        self.relation_actions = [svo['V'] for svo in self.get_all_SVO_info()]
        
        self._get_list_colors()
        self._refine_subjects()
        # self._refine_colors()

        # 3. Find objects
        self._get_list_objects()

        # 4. Find actions
        self._get_list_action()
        self._refine_list_action()

    
    # ---------------------------------------------------
    # Subject info
    def _refine_list_action(self, unique=True):
        if unique:
            self.actions = list(set(self.actions))
        # self.actions = remove_redundant_actions(self.actions)
        pass
    
    def _refine_colors(self, unique=True):
        self.colors = refine_list_colors(self.colors, unique)
        pass
    
    def _refine_subjects(self, unique=True):
        """Add rules to refine vehicle list for the given query
        """
        self.subject_vehicle = refine_list_subjects(self.subject_vehicle, unique)
        
    
    def _get_list_colors(self):
        self.subject_color = []
        for cap in self.list_caps:
            self.subject_color.extend(list(set(cap.subject.combines)))

    def _get_list_action(self):
        self.actions = []
        for cap in self.list_caps:
            if len(cap.sv_format) == 0:
                continue 
            for sv in cap.sv_format:
                self.actions.append(sv['V'])
        pass

    def _get_relation_verbs(self):
        self.relation_actions = []

        pass
    
    # ---------------------------------------------------
    # Object info
    def _get_list_objects(self, unique=True):
        self.object_vehicle = [obj.vehicle for obj in self.objects]
        self.object_color = []
        for obj in self.objects:
            self.object_color.extend(obj.combines)
        
        self.object_vehicle = refine_list_subjects(self.object_vehicle, unique, is_subject=False)
        pass
    


    # ---------------------------------------------------
    # UTILITIES
    def get_all_SV_info(self):
        sv_samples = [] 
        for cap in self.list_caps:
            for sv in cap.sv_format:
                sv_samples.append(sv)
                
        return sv_samples

    def get_all_SVO_info(self):
        svo_samples = []
        for cap in self.list_caps:
            for svo in cap.svo_format:
                svo_samples.append(svo)

        return svo_samples

    def get_list_captions(self):
        list_captions = [c.caption for c in self.list_caps]
        return list_captions

    def get_list_cleaned_captions(self):
        list_cleaned_captions = [c.cleaned_caption for c in self.list_caps]
        return list_cleaned_captions

    def to_json(self, save_path: str):

        pass
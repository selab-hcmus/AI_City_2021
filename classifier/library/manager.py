from classifier.utils.config import (
    cfg_veh, cfg_col,
    VEH_CLASS_MAP, COL_CLASS_MAP
)
from classifier.utils import (
    filter_track_veh_preds, filter_track_col_preds, get_class_name
)
from .box_extractor import init_model


class ClassifierManager(object):
    def __init__(self, cuda=True, load_ckpt=True, eval=True) -> None:
        super().__init__()
        self.veh_model, self.col_model = init_model(cfg_veh, cfg_col, load_ckpt, eval)
        if cuda:
            self.veh_model = self.veh_model.cuda()
            self.col_model = self.col_model.cuda()
        pass
    
    def get_veh_predictions(self, images: list, thres: float=0.8, weight: list = None):
        """[summary]
        Args:
            images (list): list of np.array track boxes
            thres (float, optional): [description]. Defaults to 0.8.
            weight (list, optional): [description]. Defaults to None.

        Returns:
            names, final_name, preds, final_pred
        """
        preds = self.veh_model.predict(images)
        preds, final_pred, flag_thres = filter_track_veh_preds(preds, thres, weight)
        names = [get_class_name(pred, VEH_CLASS_MAP) for pred in preds]
        final_name = get_class_name(final_pred, VEH_CLASS_MAP)
        return names, final_name, preds, final_pred, flag_thres
    
    def get_col_predictions(self, images: list, thres: float=0.8, weight: list = None):
        """
        Args:
            images (list): list of np.array boxes
        """
        preds = self.col_model.predict(images)
        preds, final_pred = filter_track_col_preds(images, thres, weight)
        names = [get_class_name(pred, COL_CLASS_MAP) for pred in preds]
        final_name = get_class_name(final_pred, COL_CLASS_MAP)
        return names, final_name, preds, final_pred
    

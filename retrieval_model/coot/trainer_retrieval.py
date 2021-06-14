"""
Trainer for retrieval training and validation. Holds the main training loop.
"""
import logging
import os
from timeit import default_timer as timer
from typing import Dict, Optional, Tuple

import h5py
import torch as th
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

from coot import model_retrieval
from coot.configs_retrieval import (
    CootMetersConst as CMeters, ExperimentTypesConst, RetrievalConfig, RetrievalTrainerState)
from coot.aic_dataset import RetrievalDataBatchTuple
from coot.loss_fn import ContrastiveLoss, LossesConst
from nntrainer import lr_scheduler, optimization, retrieval, trainer_base


class RetrievalTrainer(trainer_base.BaseTrainer):
    def __init__(
            self, cfg: RetrievalConfig, model_mgr: model_retrieval.RetrievalModelManager,
            exp_group: str, exp_name: str, run_name: str, train_loader_length: int, *,
            log_dir: str = "experiments", log_level: Optional[int] = None,
            logger: Optional[logging.Logger] = None, print_graph: bool = False, reset: bool = False,
            load_best: bool = False, load_epoch: Optional[int] = None, load_model: Optional[str] = None,
            inference_only: bool = False):
        super().__init__(
            cfg, model_mgr, exp_group, exp_name, run_name, train_loader_length, ExperimentTypesConst.RETRIEVAL,
            log_dir=log_dir, log_level=log_level, logger=logger, print_graph=print_graph, reset=reset,
            load_best=load_best, load_epoch=load_epoch, load_model=load_model, is_test=inference_only)

        print(f'Save dir: {self.exp.path_base}')
        # ---------- setup ----------

        # update type hints from base classes to inherited classes
        self.cfg: RetrievalConfig = self.cfg
        self.model_mgr: model_retrieval.RetrievalModelManager = self.model_mgr

        # overwrite default state with inherited trainer state in case we need additional state fields
        self.state = RetrievalTrainerState()

        # ---------- loss ----------
        # contrastive loss
        self.loss_contr = ContrastiveLoss(self.cfg.train.contrastive_loss_config.margin, use_cuda=self.cfg.use_cuda)
        if self.cfg.use_cuda:
            self.loss_contr = self.loss_contr.cuda()

        # ---------- additional metrics ----------

        # loss proportions
        # self.metrics.add_meter(CMeters.VAL_LOSS_CC, use_avg=False)
        self.metrics.add_meter(CMeters.VAL_LOSS_CONTRASTIVE, use_avg=False)
        # self.metrics.add_meter(CMeters.TRAIN_LOSS_CC, per_step=True, use_avg=False)
        self.metrics.add_meter(CMeters.TRAIN_LOSS_CONTRASTIVE, per_step=True, use_avg=False)

        # retrieval validation metrics must be constructed as product of two lists
        for modality in CMeters.RET_MODALITIES:
            # modality: retrieval from where to where
            for metric in CMeters.RET_METRICS:
                # metric: retrieval@1, mean, ...
                if metric == "r1":
                    # log r1 metric to the overview class
                    metric_class = "val_base"
                else:
                    # log all other metrics to the detail class
                    metric_class = "val_ret"
                self.metrics.add_meter(f"{metric_class}/{modality}-{metric}", use_avg=False)

        # ---------- optimization ----------
        self.optimizer = None
        self.lr_scheduler = None
        # skip optimizer if not training
        if not self.is_test:
            # create optimizer
            params, _param_names, _params_flat = self.model_mgr.get_all_params()
            self.optimizer = optimization.make_optimizer(self.cfg.optimizer, params)

            # create lr scheduler
            self.lr_scheduler = lr_scheduler.make_lr_scheduler(
                self.optimizer, self.cfg.lr_scheduler, self.cfg.optimizer.lr, self.cfg.train.num_epochs,
                self.train_loader_length, logger=self.logger)

        # post init hook for checkpoint loading
        self.hook_post_init()

        # PHAT attributes
        self.best_mrr = 0.0
        self.best_p2v_r1 = 0.0


    # LOSS FUNCTIONS
    def compute_align_loss(self, visual_emb: th.Tensor, text_emb: th.Tensor) -> th.Tensor:
        return self.loss_contr(visual_emb, text_emb)

    def compute_cluster_loss(self, visual_emb: th.Tensor, text_emb: th.Tensor) -> th.Tensor:
        return (self.loss_contr(visual_emb, visual_emb))
        # return (self.loss_contr(visual_emb, visual_emb) + self.loss_contr(text_emb, text_emb)) / 2

    def compute_total_constrastive_loss(self, visual_data: model_retrieval.RetrievalVisualEmbTuple,
                                        text_data: model_retrieval.RetrievalTextEmbTuple) -> th.Tensor:
        # normalize embeddings with L2-Normalization
        vid_context_norm = F.normalize(visual_data.vid_context)
        par_context_norm = F.normalize(text_data.par_context)

        # sum weighted alignments and clustering losses
        cfg = self.cfg.train.contrastive_loss_config
        loss = 0
        
        if cfg.weight_context != 0:
            loss += cfg.weight_context * self.compute_align_loss(vid_context_norm, par_context_norm)

        if cfg.weight_context_internal != 0:
            loss += cfg.weight_low_internal * self.compute_cluster_loss(vid_context_norm, par_context_norm)

        return loss

    def compute_total_ce_loss(self, visual_data: model_retrieval.RetrievalVisualEmbTuple,
                              text_data: model_retrieval.RetrievalTextEmbTuple) -> th.Tensor:
        
        # # normalize embeddings with L2-Normalization
        vid_context_norm = F.normalize(visual_data.vid_context)
        par_context_norm = F.normalize(text_data.par_context)
        
        loss = self.loss_contr(visual_data.vid_context, text_data.par_context)
        # self.loss_contr(visual_data.vid_emb, text_data.par_emb),
        # self.loss_contr(visual_data.clip_emb, text_data.sent_emb))
        cluster_loss = self.compute_cluster_loss(vid_context_norm, par_context_norm)
        self.logger.info(f"{self.state.total_step}: " + ("{:.3f} " * 2).format(loss, cluster_loss))
        return loss + cluster_loss
    
    # UTILITIES
    def phat_save_model(self, postfix_name: str='r1'):
        # models
        models_file = self.exp.get_models_file(self.state.current_epoch)
        save_path = str(models_file)
        if postfix_name is not None:
            save_path = save_path.replace('.pth', f'_{postfix_name}.pth')
        state = self.model_mgr.get_model_state()
        th.save(state, save_path)
        pass


    # TRAIN
    def train_model(self, train_loader: data.DataLoader, val_loader: data.DataLoader) -> None:
        """
        Train epochs until done.

        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
        """
        self.hook_pre_train()  # pre-training hook: time book-keeping etc.
        self.steps_per_epoch = len(train_loader)  # save length of epoch

        # ---------- Epoch Loop ----------
        for _epoch in range(self.state.current_epoch, self.cfg.train.num_epochs):
            if self.check_early_stop():
                break
            self.hook_pre_train_epoch()  # pre-epoch hook: set models to train, time book-keeping

            # ---------- Dataloader Iteration ----------
            for step, batch in enumerate(train_loader):  # type: RetrievalDataBatchTuple
                if step == 0:
                    self.logger.info(f"First step data ids: {batch.data_key[:min(4, len(batch.data_key))]}...")
                if self.check_cuda():
                    batch.to_cuda(non_blocking=self.cfg.cuda_non_blocking)

                self.hook_pre_step_timer()  # hook for step timing

                self.optimizer.zero_grad()

                # ---------- forward pass ----------
                with autocast(enabled=self.cfg.fp16_train):
                    visual_data = self.model_mgr.encode_visual(batch)
                    text_data = self.model_mgr.encode_text(batch)

                    if self.cfg.train.loss_func == LossesConst.CONTRASTIVE:
                        contr_loss = self.compute_total_constrastive_loss(visual_data, text_data)
                    # elif self.cfg.train.loss_func == LossesConst.CROSSENTROPY:
                    #     contr_loss = self.compute_total_ce_loss(visual_data, text_data)
                    loss = contr_loss

                self.hook_post_forward_step_timer()  # hook for step timing

                # ---------- backward pass ----------
                if self.cfg.fp16_train:
                    # with fp16 amp
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    # with regular float32
                    loss.backward()
                    self.optimizer.step()

                additional_log = f"Loss Contrast: {contr_loss:.5f}"
                self.hook_post_backward_step_timer()  # hook for step timing

                # post-step hook: gradient clipping, profile gpu, update metrics, count step, step LR scheduler, log
                self.hook_post_step(step, loss, self.lr_scheduler.current_lr, additional_log=additional_log)

            # ---------- validation ----------
            do_val = self.check_is_val_epoch()

            is_best = False
            if do_val:
                print(f'[PHAT]: DO VAL')
                # check if clip retrieval should be validated (it's costly)
                val_clips = self.cfg.val.val_clips and (
                    (self.cfg.val.val_clips_freq > 0 and
                     self.state.current_epoch % self.cfg.val.val_clips_freq == 0))

                # run validation, collect video and clip retrieval metrics
                _val_loss, _val_score, is_best, _metrics = self.validate_epoch(val_loader)

            # post-epoch hook: scheduler, save checkpoint, time bookkeeping, feed tensorboard
            self.hook_post_train_and_val_epoch(do_val, is_best)

        # show end of training log message
        self.hook_post_train()


    # VALIDATE
    @th.no_grad()
    def validate_epoch(self, data_loader: data.DataLoader, val_clips: bool = False, save_embs: bool = False) -> (
            Tuple[float, float, bool, Tuple[Dict[str, float], Optional[Dict[str, float]]]]):

        self.hook_pre_val_epoch()  # pre val epoch hook: set models to val and start timers
        forward_time_total = 0
        loss_total: th.Tensor = 0.
        contr_loss_total: th.Tensor = 0.
        data_collector = {}

        # decide what to collect
        save_clip_num, save_sent_num, save_key = [], [], []
        collect_keys = ["vid_context", "par_context"]

        # ---------- Dataloader Iteration ----------
        num_steps = 0
        pbar = tqdm(total=len(data_loader), desc=f"Validate epoch {self.state.current_epoch}")
        for _step, batch in enumerate(data_loader):  # type: RetrievalDataBatchTuple
            # move data to cuda
            if self.check_cuda():
                batch.to_cuda(non_blocking=self.cfg.cuda_non_blocking)

            if save_embs:
                # collect meta information for saving
                save_clip_num.extend(batch.clip_num.cpu().numpy().tolist())
                save_sent_num.extend(batch.clip_num.cpu().numpy().tolist())
                save_key.extend(batch.key)

            # ---------- forward pass ----------
            self.hook_pre_step_timer()  # hook for step timing

            with autocast(enabled=self.cfg.fp16_val):
                visual_data = self.model_mgr.encode_visual(batch)
                text_data = self.model_mgr.encode_text(batch)
                if self.cfg.train.loss_func == LossesConst.CONTRASTIVE:
                    contr_loss = self.compute_total_constrastive_loss(visual_data, text_data)
                elif self.cfg.train.loss_func == LossesConst.CROSSENTROPY:
                    contr_loss = self.compute_total_ce_loss(visual_data, text_data)
                contr_loss_total += contr_loss
                loss_total += contr_loss

            self.hook_post_forward_step_timer()
            forward_time_total += self.timedelta_step_forward
            num_steps += 1

            # ---------- data collection ----------
            all_data = {**visual_data.dict(), **text_data.dict()}
            for key in collect_keys:
                emb = all_data.get(key)
                # collect embeddings into list, on CPU otherwise the gpu runs OOM
                if data_collector.get(key) is None:
                    data_collector[key] = [emb.data.cpu()]
                else:
                    data_collector[key] += [emb.data.cpu()]
            pbar.update()
        pbar.close()

        # ---------- validation done ----------

        # postprocess collected embeddings
        data_collector_norm = {}
        for key in collect_keys:
            data_collector[key] = th.cat(data_collector[key], dim=0).float()
            # data_collector_norm[key] = F.normalize(data_collector[key])
            data_collector_norm[key] = data_collector[key] / (data_collector[key] * data_collector[key]).sum(
                dim=-1).sqrt().unsqueeze(-1)

        # calculate total loss and feed meters
        loss_total /= num_steps
        contr_loss_total /= num_steps
        forward_time_total /= num_steps
        self.metrics.update_meter(CMeters.VAL_LOSS_CONTRASTIVE, contr_loss_total)

        # calculate video-paragraph retrieval and print output table
        self.logger.info(retrieval.VALHEADER)
        res_v2p, res_p2v, sum_vp_at_1, str_vp = retrieval.compute_retrieval(
            data_collector_norm, "vid_context", "par_context", print_fn=self.logger.info)

        # calculate clip-sentence retrieval and print output table
        res_c2s, res_s2c, sum_cs_at_1, clipsent_results = None, None, None, None
        str_cs = ""

        # feed retrieval results to meters
        for modality, dict_ret in zip(CMeters.RET_MODALITIES, [res_v2p, res_p2v]):
            if dict_ret is None:
                continue
            # iterate over result keys
            for metric in CMeters.RET_METRICS:
                # feed averagemeters
                logger_class = "val_ret"
                if metric == "r1":
                    logger_class = "val_base"
                self.metrics.update_meter(f"{logger_class}/{modality}-{metric}", dict_ret[metric])

        # print some more details about the retrieval (time, number of datapoints)
        self.logger.info(
            f"Loss {loss_total:.5f} (Contr: {contr_loss_total:.5f}"
            f"Retrieval: {str_vp}{str_cs}total {timer() - self.timer_val_epoch:.3f}s, "
            f"forward {forward_time_total:.3f}s")

        # find field which determines whether this is a new best epoch
        val_score = sum_vp_at_1
        
        if res_p2v['mrr'] > self.best_mrr:
            print(f"[PHAT] Achieve best MRR, before: {self.best_mrr:.04f}, after: {res_p2v['mrr']:.04f}")
            self.phat_save_model('mrr')
            self.best_mrr = res_p2v['mrr']

        if res_p2v['r1'] > self.best_p2v_r1:
            print(f"[PHAT] Achieve best p2v R@1, before: {self.best_p2v_r1:.04f}, after: {res_p2v['r1']:.04f}")
            self.phat_save_model('r1')
            self.best_p2v_r1 = res_p2v['r1']


        # check for a new best epoch and update validation results
        is_best = self.check_is_new_best(val_score)
        self.hook_post_val_epoch(loss_total, is_best)

        return loss_total, val_score, is_best, ((res_v2p, res_p2v, sum_vp_at_1), clipsent_results)

    def get_opt_state(self) -> Dict[str, Dict[str, nn.Parameter]]:
        """
        Return the current optimizer and scheduler state.

        Returns:
            Dictionary of optimizer and scheduler state dict.
        """
        return {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }

    def set_opt_state(self, opt_state: Dict[str, Dict[str, nn.Parameter]]) -> None:
        """
        Set the current optimizer and scheduler state from the given state.

        Args:
            opt_state: Dictionary of optimizer and scheduler state dict.
        """
        self.optimizer.load_state_dict(opt_state["optimizer"])
        self.lr_scheduler.load_state_dict(opt_state["lr_scheduler"])

import einops
from base.logger import ContinuousOutputHandler, ContinuousMetricsCalculator, PlotHandler
from base.scheduler import GradualWarmupScheduler

from base.utils import ensure_dir

import time
import copy
import os
from tqdm import tqdm
from einops import *


import pandas as pd

import numpy as np
import torch
from torch import optim
import wandb


class GenericTrainer(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.model_name = kwargs['model_name']
        self.model = kwargs['models'].to(self.device)
        self.save_path = kwargs['save_path']
        self.fold = kwargs['fold']
        self.min_epoch = kwargs['min_epoch']
        self.max_epoch = kwargs['max_epoch']
        self.start_epoch = 0
        self.early_stopping = kwargs['early_stopping']
        self.early_stopping_counter = self.early_stopping
        self.scheduler = kwargs['scheduler']
        self.learning_rate = kwargs['learning_rate']
        self.min_learning_rate = kwargs['min_learning_rate']
        self.patience = kwargs['patience']
        self.criterion = kwargs['criterion']
        self.factor = kwargs['factor']
        self.verbose = kwargs['verbose']
        self.milestone = kwargs['milestone']
        self.load_best_at_each_epoch = kwargs['load_best_at_each_epoch']
        self.use_scheduler = kwargs['use_scheduler']

        self.optimizer, self.scheduler = None, None

    def train(self, **kwargs):
        kwargs['train_mode'] = True
        self.model.train()
        loss = self.loop(**kwargs)
        return loss

    def validate(self, **kwargs):
        kwargs['train_mode'] = False
        with torch.no_grad():
            self.model.eval()
            loss = self.loop(**kwargs)
        return loss

    def test(self, checkpoint_controller, predict_only=0, **kwargs):
        kwargs['train_mode'] = False

        with torch.no_grad():
            self.model.eval()

            if predict_only:
                self.predict_loop(**kwargs)
            else:
                loss = self.loop(**kwargs)
                checkpoint_controller.save_log_to_csv(
                    kwargs['epoch'], mean_train_record=None, mean_validate_record=None, test_record=None)

                return loss

    def fit(self, **kwargs):
        raise NotImplementedError

    def loop(self, **kwargs):
        raise NotImplementedError

    def predict_loop(self, **kwargs):
        raise NotImplementedError

    def get_parameters(self):
        r"""
        Get the parameters to update.
        :return:
        """
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        return params_to_update


class GenericVideoTrainer(GenericTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = kwargs['batch_size']
        self.emotion = kwargs['emotion']
        self.metrics = kwargs['metrics']
        self.save_plot = kwargs['save_plot']

        # For checkpoint
        self.fit_finished = False
        self.fold_finished = False
        self.resume = False
        self.time_fit_start = None

        self.train_losses = []
        self.validate_losses = []
        self.csv_filename = None
        self.best_epoch_info = None
        self.global_step = 0


    def fit(self, dataloader_dict, checkpoint_controller, parameter_controller):

        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
            }

        for epoch in np.arange(start_epoch, self.max_epoch):

            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            improvement = False

            if epoch in self.milestone or parameter_controller.get_current_lr() < self.min_learning_rate:
                parameter_controller.release_param(self.model.spatial, epoch)
                if parameter_controller.early_stop:
                    break

                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            time_epoch_start = time.time()

            if self.verbose:
                print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            # Get the losses and the record dictionaries for training and validation.
            train_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            train_loss = self.train(**train_kwargs)

            validate_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            validate_loss = self.validate(**validate_kwargs)

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            if validate_loss < self.best_epoch_info['loss']:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_state_dict.pth"))

                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'epoch': epoch,
                }

            if self.verbose:
                print(
                    "\n Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} | Release_count={} | best={} | "
                    "improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        parameter_controller.release_count,
                        int(self.best_epoch_info['epoch']) + 1,
                        improvement,
                        self.early_stopping_counter))

            # checkpoint_controller.save_log_to_csv(
            #     epoch, train_record_dict['overall'], validate_record_dict['overall'])

            # Early stopping controller.
            if self.early_stopping and epoch > self.min_epoch:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            self.scheduler.step(metrics=validate_loss, epoch=epoch)
            self.start_epoch = epoch + 1

            if self.load_best_at_each_epoch:
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.model.load_state_dict(self.best_epoch_info['model_weights'])

    def loop(self, **kwargs):
        dataloader_dict, epoch, train_mode = kwargs['dataloader_dict'], kwargs['epoch'], kwargs['train_mode']

        if train_mode:
            dataloader = dataloader_dict['train']
        elif epoch is None:
            dataloader = dataloader_dict['extra']
        else:
            dataloader = dataloader_dict['validate']

        running_loss = 0.0
        total_batch_counter = 0
        inputs = {}

        num_batch_warm_up = len(dataloader) * self.min_epoch
        for batch_idx, (X, trials, lengths, indices, true_length) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # if batch_idx < 157:
            #     continue
            
            if train_mode:
                if self.use_scheduler:
                    self.scheduler.warmup_lr(self.learning_rate, self.global_step,  num_batch_warm_up)

            total_batch_counter += len(trials)

            for feature, value in X.items():
                inputs[feature] = X[feature].to(self.device)
            if "continuous_label" in inputs:
                labels = inputs.pop("continuous_label", None)
            elif "VA_continuous_label" in inputs:
                labels = inputs.pop("VA_continuous_label", None)
            
            labels = inputs.pop("labels", None)
            

            if len(torch.flatten(labels)) == self.batch_size:
                labels = torch.zeros((self.batch_size, len(indices[0]), 1), dtype=torch.float32).to(self.device)
                
            true_length = true_length.to(self.device)
            B, T = X['video'].shape[0], X['video'].shape[1]
            mask = repeat(torch.arange(T), 't -> b t', b=B).to(self.device)
            mask = (mask < rearrange(true_length, 'b -> b 1')) # [B, T]
            # mask = rearrange(mask, 'b t -> b t')

            if train_mode:
                self.optimizer.zero_grad()
            # output= self.model(inputs, mask=mask)
            logits0, logits_rest = self.model(inputs, mask=mask)
            
            # if not self.reduce_frame_level_features:
            #     outputs = outputs * rearrange(mask, 'b t -> b t 1')
            #     outputs = reduce(outputs, 'b t d -> b d', 'sum') / rearrange(true_length, 'b -> b 1')
            if not self.reduce_frame_level_features:
                logits0 = logits0 * rearrange(mask, 'b t -> b t 1')
                logits0 = torch.sum(logits0, dim=1) / rearrange(true_length, 'b -> b 1')

                logits_rest = logits_rest * rearrange(mask, 'b t -> b t 1 1')
                logits_rest = torch.sum(logits_rest, dim=1) / rearrange(true_length, 'b -> b 1 1')
            
            levels0 = torch.tensor([
                -3.00, -2.66, -2.33, -2.00, -1.66, -1.33, -1.00, -0.66, -0.33,
                0.00,
                0.33,  0.66,  1.00,  1.33,  1.66,  2.00,  2.33,  2.66,  3.00
            ], device=self.device)  # 19개

            levels_rest = torch.tensor([0.00, 0.33, 0.66, 1.00], device=self.device)  # 4개
            # logits0 = output0[:, 0]
            # idx0 = torch.argmin(torch.abs(logits0.unsqueeze(1) - levels0.to(self.device)), dim=-1)
            # logits_rest = output[:, 1:]
            # diff = logits_rest.unsqueeze(1) - levels_rest.to(self.device)
            # idx_rest = torch.argmin(torch.abs(diff), dim=-1)
            labels = labels.to(self.device)
            idx0 = torch.argmin(
                torch.abs(labels[:, 0].unsqueeze(-1) - levels0),  # (B,19)
                dim=-1
            )  # (B,)
            idx0 = idx0.long()

            # dim1~6: (B,6) → 각 위치마다 4개 레벨 중 가장 가까운 index
            # labels[:,1:]: (B,6)
            diff = labels[:, 1:].unsqueeze(-1) - levels_rest  # (B,6,4)
            idx_rest = torch.argmin(torch.abs(diff), dim=-1)  # (B,6)
            idx_rest = idx_rest.long()


            # 0번째 차원 Loss (19-class CE + class weight)
            loss0 = torch.nn.functional.cross_entropy(logits0, idx0)
            weights_rest = torch.tensor([1/8, 1/2, 1/2, 1/2],device=self.device, dtype=logits_rest.dtype)
            # 1~6번째 차원 Loss (6개의 4-class CE 평균)
            loss_rest = torch.nn.functional.cross_entropy(
            logits_rest.view(-1, 4),      # (B*6, 4)
            idx_rest.view(-1),            # (B*6,)
            weight=weights_rest.to(logits_rest.device)
            )
            loss = loss0 + loss_rest
            # loss = self.criterion(labels, outputs)

            running_loss += loss.mean().item()

            if train_mode:
                self.global_step += 1
                if self.no_wandb is False:
                    wandb.log({"train/loss": loss.mean().item(), "train/lr": self.optimizer.param_groups[0]['lr'], "epoch": epoch}, step=self.global_step)

            if train_mode:
                loss.backward()
                self.optimizer.step()

        epoch_loss = running_loss / total_batch_counter
        
        if not train_mode:
            if self.no_wandb is False:
                wandb.log({"validate/loss": epoch_loss}, step=self.global_step)

        return epoch_loss

    def predict_loop(self, **kwargs):
        partition = kwargs['partition']
        dataloader = kwargs['dataloader_dict'][partition]
        inputs = {}

        output_handler = ContinuousOutputHandler()
        for batch_idx, (X, trials, lengths, indices) in tqdm(enumerate(dataloader), total=len(dataloader)):

            for feature, value in X.items():
                if "label" in feature:
                    continue
                inputs[feature] = X[feature].to(self.device)

            outputs = self.model(inputs)
    
            output_handler.update_output_for_seen_trials(outputs.detach().cpu().numpy(), trials, indices, lengths)

        output_handler.average_trial_wise_records()
        output_handler.concat_records()

        for trial, result in output_handler.trialwise_records.items():

            txt_save_path = os.path.join(self.save_path, "predict", partition, self.emotion, trial + ".txt")
            ensure_dir(txt_save_path)
            df = pd.DataFrame(data=result, index=None, columns=[self.emotion])
            df.to_csv(txt_save_path, sep=",", index=None)


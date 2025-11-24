from base.trainer import GenericVideoTrainer
from base.scheduler import GradualWarmupScheduler, MyWarmupScheduler

from torch import optim
import torch

import time
import copy
import os

import numpy as np



def save_trainable_weights(model, path):
    trainable_params = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    # import ipdb; ipdb.set_trace()

    torch.save(trainable_params, path)
    print("model parameters saved at:", path)

class Trainer(GenericVideoTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'acc': -1,
            'p_r_f1': 0,
            'kappa': 0,
            'epoch': 0,
            'metrics': {
                'train_loss': -1,
                'val_loss': -1,
                'train_acc': -1,
                'val_acc': -1,
            }
        }
        
        self.no_wandb = kwargs.get('no_wandb')
        self.reduce_frame_level_features = kwargs.get('reduce_frame_level_features', False)

    def init_optimizer_and_scheduler(self, epoch=0):
        self.optimizer = optim.Adam(self.get_parameters(), lr=self.learning_rate, weight_decay=0.001)
        if self.use_scheduler:
            self.scheduler = MyWarmupScheduler(
                optimizer=self.optimizer, lr = self.learning_rate, min_lr=self.min_learning_rate,
                best=self.best_epoch_info['loss'], mode="min", patience=self.patience,
                factor=self.factor, num_warmup_epoch=self.min_epoch, init_epoch=epoch)
        

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

            # if epoch in self.milestone or (parameter_controller.get_current_lr() < self.min_learning_rate and epoch >= self.min_epoch and self.scheduler.relative_epoch > self.min_epoch):
            #     parameter_controller.release_param(self.model.spatial, epoch)
            #     if parameter_controller.early_stop:
            #         break

            #     self.model.load_state_dict(self.best_epoch_info['model_weights'])

            time_epoch_start = time.time()

            if self.verbose:
                print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            # Get the losses and the record dictionaries for training and validation.
            train_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            train_loss = self.train(**train_kwargs)

            validate_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            validate_loss = self.validate(**validate_kwargs)

            # if epoch % 1 == 0:
            #     test_kwargs = {"dataloader_dict": dataloader_dict, "epoch": None, "train_mode": 0}
            #     validate_loss, test_record_dict = self.test(checkpoint_controller=checkpoint_controller, feature_extraction=0, **test_kwargs)
            #     print(test_record_dict['overall']['ccc'])

            if validate_loss < 0:
                raise ValueError('validate loss negative')

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            # validate_ccc = validate_record_dict['overall']['ccc']

            if self.use_scheduler:
                self.scheduler.best = self.best_epoch_info['loss']


            if validate_loss < self.best_epoch_info['loss']:
                save_trainable_weights(self.model, os.path.join(self.save_path, "best.pt"))

                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'epoch': epoch,
                }

            if self.verbose:
                print(
                    "\n Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} | best={} | "
                    "improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        int(self.best_epoch_info['epoch']) + 1,
                        improvement,
                        self.early_stopping_counter))

            # checkpoint_controller.save_log_to_csv(
            #     epoch, train_record_dict['overall'], validate_record_dict['overall'])

            # Early stopping controller.
            if self.early_stopping and (not self.use_scheduler or self.scheduler.relative_epoch > self.min_epoch):
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            if self.use_scheduler:
                self.scheduler.step(metrics=validate_loss, epoch=epoch)

            self.start_epoch = epoch + 1

            if self.load_best_at_each_epoch:
                self.model.load_state_dict(self.best_epoch_info['model_weights'])
            
            save_trainable_weights(self.model, os.path.join(self.save_path, f'epoch_{epoch+1}.pt'))

            # checkpoint_controller.save_checkpoint(self, parameter_controller, epoch, self.save_path)

        self.fit_finished = True
        
        save_trainable_weights(self.model, os.path.join(self.save_path, f'last.pt'))
        # checkpoint_controller.save_checkpoint(self, parameter_controller, epoch, self.save_path)

        self.model.load_state_dict(self.best_epoch_info['model_weights'])

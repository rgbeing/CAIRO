import gc
import json
import time
import yaml

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm

from profiler.preprocess import convert_objective_profiles_to_table
from extraction.objective_profile import read_key_values




class EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save({'state_dict': model.state_dict()}, self.save_path) 
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

    


def train(model, device,
    optimizer, optimizer_model, optimizer_darts, 
    train_data_loader, valid_data_loader, 
    criterion, 
    log_interval, 
    controller, 
    darts_frequency):

    model.train()
    total_loss = 0
    tk0 = tqdm(train_data_loader, smoothing=0, mininterval=1.0)
    valid_data_loader_iter = iter(valid_data_loader)

    torch.autograd.set_detect_anomaly(True) # Enable anomaly detection for debugging

    for i, (fields, target) in enumerate(tk0):
        # Extract categorical and vector features from fields dict
        cat_fields = fields['categorical'].to(device) if fields['categorical'].numel() > 0 else None
        vec_fields = fields['vector'].to(device) if fields['vector'].numel() > 0 else None
        target = target.to(device)

        # Convert categorical to long if present
        if cat_fields is not None:
            cat_fields = cat_fields.long()

        y_hat, _ = model(cat_field=cat_fields, vec_field=vec_fields)
        loss = criterion(y_hat, target.float())

        model.zero_grad()
        loss.backward()

        # Update all params of model if you do not use controller
        if not controller:
            optimizer.step()

        # Pretraining case
        if controller and model.stage == 0:
            optimizer_model.step()
        
        # Search stage, alternatively update main RS network and Darts weights
        if controller and model.stage == 1:
            optimizer_model.step()

            if (i + 1) % darts_frequency == 0:
                try:
                    fields, target = next(valid_data_loader_iter)
                except StopIteration:
                    del valid_data_loader_iter
                    gc.collect()

                    valid_data_loader_iter = iter(valid_data_loader)
                    fields, target = next(valid_data_loader_iter)
                
                cat_fields = fields['categorical'].to(device) if fields['categorical'].numel() > 0 else None
                vec_fields = fields['vector'].to(device) if fields['vector'].numel() > 0 else None
                target = target.to(device)
                
                if cat_fields is not None:
                    cat_fields = cat_fields.long()
                
                y_hat, _ = model(cat_field=cat_fields, vec_field=vec_fields)
                loss_val = criterion(y_hat, target.float())

                model.zero_grad()
                loss_val.backward()
                optimizer_darts.step()
        
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0




def test(model, data_loader, device):
    model.eval()

    targets, predicts, infer_time, masks = list(), list(), list(), list()
    with torch.no_grad():
        for fields, target in tqdm(data_loader, smoothing=0, mininterval=1.0):
            # Extract categorical and vector features from fields dict
            cat_fields = fields['categorical'].to(device) if fields['categorical'].numel() > 0 else None
            vec_fields = fields['vector'].to(device) if fields['vector'].numel() > 0 else None
            target = target.to(device)
            
            # Convert categorical to long if present
            if cat_fields is not None:
                cat_fields = cat_fields.long()

            start = time.time()
            y, mask_for_this_fields = model(cat_field=cat_fields, vec_field=vec_fields)
            infer_cost = time.time() - start

            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            infer_time.append(infer_cost)
            
            # Only collect masks if they exist (during stage 1 with controller)
            if mask_for_this_fields is not None:
                masks.extend(mask_for_this_fields)

    # Stack masks only if they were collected, otherwise return None
    stacked_masks = torch.stack(masks) if len(masks) > 0 else None
    return roc_auc_score(targets, predicts), log_loss(targets, predicts), sum(infer_time), stacked_masks

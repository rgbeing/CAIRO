import argparse
import json
import os
import random
import yaml
from pathlib import Path
from typing import List

import gc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

from extraction.subjective_profile import read_user_profiles
from profiler.AdaFS.dataset import AmazonDataset
from profiler.AdaFS.model import AdaFS_hard
from profiler.train_fs import EarlyStopper, train, test
from profiler.preprocess import convert_objective_profiles_to_table, convert_user_profiles_to_embeddings




def add_negative_samples(df, user_non_interacted_items, n_neg_per_pos):
    # Add label column to existing interactions
    df = df.copy()
    df['label'] = 1
    
    # Generate negative samples
    negative_samples = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        user = row['user_id']
        available_items = user_non_interacted_items[user]
        
        # Sample n_neg_per_pos items (or fewer if not enough available)
        n_samples = min(n_neg_per_pos, len(available_items))
        sampled_items = random.sample(available_items, n_samples)
        
        for item in sampled_items:
            negative_samples.append({
                'user_id': user,
                'parent_asin': item,
                'label': 0
            })
    
    # Create negative samples dataframe and combine
    negative_df = pd.DataFrame(negative_samples)
    result_df = pd.concat([df, negative_df], ignore_index=True)
    
    return result_df




def construct_dataset(interactions_df, paths_config, n_neg_per_pos):
    print("Constructing train, valid, test datasets with negative sampling")

    print("Loading BGE-M3 model for feature encoding")
    # BGE-M3 to use
    bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    ## Train-valid-test split
    # Sort by user_id and timestamp
    interactions_df = interactions_df.sort_values(['user_id', 'timestamp'])

    # Group by user and get the rank of each interaction (last, second-last, etc.)
    interactions_df['rank'] = interactions_df.groupby('user_id').cumcount(ascending=False)

    # Split based on rank
    test_interactions_df = interactions_df[interactions_df['rank'] == 0].drop(['rank', 'timestamp'], axis=1)
    valid_interactions_df = interactions_df[interactions_df['rank'] == 1].drop(['rank', 'timestamp'], axis=1)
    train_interactions_df = interactions_df[interactions_df['rank'] >= 2].drop(['rank', 'timestamp'], axis=1)

    ## Negative sampling
    # For each user, create a set of items they HAVE NOT interacted with
    unique_items = set(interactions_df['parent_asin'].unique())
    user_non_interacted_items = {}

    for user in interactions_df['user_id'].unique():
        user_items = set(interactions_df[interactions_df['user_id'] == user]['parent_asin'])
        user_non_interacted_items[user] = list(unique_items - user_items)

    print("Add negative samples to train set")
    train_interactions_df = add_negative_samples(train_interactions_df, user_non_interacted_items, n_neg_per_pos)
    print("Add negative samples to valid set")
    valid_interactions_df = add_negative_samples(valid_interactions_df, user_non_interacted_items, n_neg_per_pos)
    print("Add negative samples to test set")
    test_interactions_df = add_negative_samples(test_interactions_df, user_non_interacted_items, n_neg_per_pos)


    ## Join features from profiles
    # Load objective profiles
    item_encoded_df = convert_objective_profiles_to_table(paths_config, bge_model)
    # To save memory: BGE is removed
    del bge_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load user profiles
    user_df = read_user_profiles(paths_config)
    # Keep only user_id and user_profile_embedding, rename embedding column to user_profile
    user_encoded_df = user_df.drop('user_profile', axis=1)
    user_encoded_df = user_encoded_df.rename(columns={'embedding': 'user_profile'})

    print("Create dataset objects")

    train_interactions_df = pd.merge(train_interactions_df, user_encoded_df, on='user_id', how='left')
    train_interactions_df = pd.merge(train_interactions_df, item_encoded_df, on='parent_asin', how='left')

    test_interactions_df = pd.merge(test_interactions_df, user_encoded_df, on='user_id', how='left')
    test_interactions_df = pd.merge(test_interactions_df, item_encoded_df, on='parent_asin', how='left')

    valid_interactions_df = pd.merge(valid_interactions_df, user_encoded_df, on='user_id', how='left')
    valid_interactions_df = pd.merge(valid_interactions_df, item_encoded_df, on='parent_asin', how='left')

    # To save memory
    del item_encoded_df, user_encoded_df
    gc.collect()

    # User and Item IDs are now strings, convert it to integer indices
    # Create mapping tables from string IDs to integers
    user_id_to_int = {user_id: idx for idx, user_id in enumerate(interactions_df['user_id'].unique())}
    parent_asin_to_int = {asin: idx for idx, asin in enumerate(interactions_df['parent_asin'].unique())}

    # Apply mappings to train_interactions_df
    train_interactions_df['user_id'] = train_interactions_df['user_id'].map(user_id_to_int)
    train_interactions_df['parent_asin'] = train_interactions_df['parent_asin'].map(parent_asin_to_int)

    # Apply mappings to test_interactions_df
    test_interactions_df['user_id'] = test_interactions_df['user_id'].map(user_id_to_int)
    test_interactions_df['parent_asin'] = test_interactions_df['parent_asin'].map(parent_asin_to_int)

    # Apply mappings to valid_interactions_df
    valid_interactions_df['user_id'] = valid_interactions_df['user_id'].map(user_id_to_int)
    valid_interactions_df['parent_asin'] = valid_interactions_df['parent_asin'].map(parent_asin_to_int)

    test_dataset = AmazonDataset(test_interactions_df)
    del test_interactions_df
    valid_dataset = AmazonDataset(valid_interactions_df)
    del valid_interactions_df
    train_dataset = AmazonDataset(train_interactions_df)
    del train_interactions_df
    gc.collect()

    # Save datasets
    print("Save datasets")
    test_path = Path(paths_config['fs_dataset_save_path'].format(type='test'))
    valid_path = Path(paths_config['fs_dataset_save_path'].format(type='valid'))
    train_path = Path(paths_config['fs_dataset_save_path'].format(type='train'))

    torch.save(test_dataset, test_path, pickle_protocol=5)
    print("\tTest saved")
    torch.save(valid_dataset, valid_path, pickle_protocol=5)
    print("\tValid saved")
    torch.save(train_dataset, train_path, pickle_protocol=5)
    print("\tTrain saved")
    print("Datasets for feature selection saved successfully")

    return train_dataset, valid_dataset, test_dataset




def read_dataset(paths_config, types: List):
    print(f"Loading {types} dataset for feature selection")
    datasets = []
    for type in types:
        dataset_path = Path(paths_config['fs_dataset_save_path'].format(type=type))
        datasets.append(torch.load(dataset_path, weights_only=False))
    
    return datasets




def main(device, args, paths_config):
    # Read reviews
    print("Reading review (interaction) file to construct the datasets")
    reviews_path = Path(paths_config['reviews_path'])
    reviews_list = []
    with open(reviews_path, 'r') as f:
        for line in f:
            reviews_list.append(json.loads(line))
    reviews_df = pd.DataFrame(reviews_list)[["parent_asin", "user_id", "timestamp"]]

    # construct datasets
    train_dataset, valid_dataset, test_dataset = construct_dataset(
        interactions_df=reviews_df,
        paths_config=paths_config,
        n_neg_per_pos=args.n_neg_per_pos
    )

    # Inspect dataset to get necessary parameters for model initialization
    print("Categorical features:", train_dataset.categorical_features_name)
    print("Vector features:", train_dataset.vector_features_name)
    print("Number of categorical features:", len(train_dataset.categorical_features_name))
    print("Number of vector features:", len(train_dataset.vector_features_name))

    if train_dataset.categorical_features is not None:
        print("Categorical features shape:", train_dataset.categorical_features.shape)
        # Get categorical field dimensions from ALL datasets (train, valid, test)
        # to ensure embeddings can handle all possible values
        categorical_field_dims = []
        for i in range(train_dataset.categorical_features.shape[1]):
            max_val = max(
                int(train_dataset.categorical_features[:, i].max()),
                int(valid_dataset.categorical_features[:, i].max()),
                int(test_dataset.categorical_features[:, i].max())
            )
            categorical_field_dims.append(max_val + 1)
        print("Categorical field dims:", categorical_field_dims)
    else:
        categorical_field_dims = []

    if train_dataset.vector_features is not None:
        print("Vector features shape:", train_dataset.vector_features.shape)
        vec_input_dim = train_dataset.vector_features.shape[2]  # dimension of each vector
        print("Vector input dimension:", vec_input_dim)
    else:
        vec_input_dim = 0

    model = AdaFS_hard(
        args=args,
        categorical_features=train_dataset.categorical_features_name,
        vector_features=train_dataset.vector_features_name,
        categorical_field_dims=categorical_field_dims,
        vector_input_dim=vec_input_dim,
        embed_dim=args.embed_dim,
        mlp_dims=eval(args.mlp_dims),
        device=device
    ).to(device)
    print(f"Model created with {len(train_dataset.categorical_features_name)} categorical features "
        f"and {len(train_dataset.vector_features_name)} vector features")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizers
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_model = torch.optim.Adam(params=[param for name, param in model.named_parameters() if 'controller' not in name], lr=args.lr, weight_decay=args.weight_decay)
    optimizer_darts = torch.optim.Adam(params=[param for name, param in model.named_parameters() if 'controller' in name], lr=args.lr_darts, weight_decay=args.weight_decay)

    # dataloaders
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # loss function
    criterion = torch.nn.BCELoss()


    print('\n********************************************* Pretrain *********************************************\n')
    model.stage = 0

    early_stopper = EarlyStopper(num_trials=3, save_path=f"{paths_config['fs_model_save_dir']}/{args.dataset_name}_pretrain.pt")

    for epoch_i in range(args.pretrain_epoch):
        print('Pretrain epoch:', epoch_i)
        train(model, device, 
              optimizer, optimizer_model, optimizer_darts, 
              train_data_loader, valid_data_loader, 
              criterion, args.log_interval, args.controller, args.darts_frequency)

        auc, logloss, infer_time, _ = test(model, valid_data_loader, device)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
        
        print('Pretrain epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)
    
    auc, logloss, infer_time, _ = test(model, test_data_loader, device)
    print(f'Pretrain test auc: {auc} logloss: {logloss}, infer time:{infer_time}\n')

    
    print('\n********************************************* Main_train *********************************************\n')
    model.stage = 1

    if args.controller:
        early_stopper = EarlyStopper(num_trials=3, save_path=f"{paths_config['fs_model_save_dir']}/{args.dataset_name}_controller.pt")
    else:
        early_stopper = EarlyStopper(num_trials=3, save_path=f"{paths_config['fs_model_save_dir']}/{args.dataset_name}_noController.pt")

    for epoch_i in range(args.main_epoch):
        print('epoch:', epoch_i)
        train(model, device, 
              optimizer, optimizer_model, optimizer_darts, 
              train_data_loader, valid_data_loader, 
              criterion, args.log_interval, args.controller, args.darts_frequency)

        auc, logloss, _, _ = test(model, valid_data_loader, device)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

        print('epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)

    auc, logloss, infer_time, test_masks = test(model, test_data_loader, device)
    print(f'test auc: {auc} logloss: {logloss}\n')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_neg_per_pos', type=int, default=2, help='Number of negative samples per positive sample')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--k', type=int, default=4, help='Number of top features to select')
    parser.add_argument('--controller', action='store_true', default=True, help='Whether to use the controller')
    parser.add_argument('--useWeight', action='store_true', default=True, help='Whether to use weight values in masking')
    parser.add_argument('--reWeight', action='store_true', default=True, help='Whether to renormalize weights to sum to 1')
    parser.add_argument('--useBN', action='store_true', default=True, help='Whether to use batch normalization')
    parser.add_argument('--pretrain_epoch', type=int, default=5, help='Number of pretraining epochs')
    parser.add_argument('--main_epoch', type=int, default=30, help='Total number of main training epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for main model')
    parser.add_argument('--lr_darts', type=float, default=0.001, help='Learning rate for DARTS controller')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for main model optimizer')
    parser.add_argument('--dataset_name', type=str, default='Video_Games', help='Dataset name')
    parser.add_argument('--embed_dim', type=int, default=16, help='Embedding dimension for features')
    parser.add_argument('--mlp_dims', type=str, default='[64,16]', help='MLP dimensions as a list string, e.g., "[64,32]"')
    parser.add_argument('--device', type=str, default='0', help='Device to use for training')
    parser.add_argument('--darts_frequency', type=int, default=1, help='Frequency of DARTS controller updates')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval during training')

    args = parser.parse_args()

    with open("./paths_config.yaml", 'r') as f:
        paths_config = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else 'cpu')

    main(device, args, paths_config)


import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from pathlib import Path

class AmazonDataset(Dataset):

    def __init__(self, data_df: pd.DataFrame):
        features_df = data_df.drop(['label'], axis=1)
        self.labels = data_df['label'].values
        
        # Separate vector features from categorical features
        vector_cols = []
        categorical_cols = []
        
        for col in features_df.columns:
            # Check the type of the first entry to determine if the column has a vector or categorical feature
            first_val = features_df[col].iloc[0]
            if isinstance(first_val, (np.ndarray, list, torch.Tensor)):
                vector_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Store feature names
        self.vector_features_name = vector_cols
        self.categorical_features_name = categorical_cols
        
        # Stack vector features: shape (n_samples, n_vector_features, vec_dim)
        if vector_cols:
            self.vector_features = np.stack([np.stack(features_df[col].values) for col in vector_cols], axis=1)
        else:
            self.vector_features = None
        
        # Stack categorical features: shape (n_samples, n_categorical_features)
        if categorical_cols:
            self.categorical_features = features_df[categorical_cols].values
        else:
            self.categorical_features = None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.vector_features is not None:
            # shape: (n_vector_features, vec_dim)
            vector_tensor = torch.tensor(self.vector_features[idx], dtype=torch.float32)
        else:
            vector_tensor = torch.tensor([], dtype=torch.float32)
        
        if self.categorical_features is not None:
            # shape: (n_categorical_features,)
            categorical_tensor = torch.tensor(self.categorical_features[idx], dtype=torch.float32)
        else:
            categorical_tensor = torch.tensor([], dtype=torch.float32)
        
        x = {'vector': vector_tensor, 'categorical': categorical_tensor}
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


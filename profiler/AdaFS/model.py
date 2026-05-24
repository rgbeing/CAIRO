from typing import List

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F





def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0] 
    return index, x.gather(dim, index)





class EMB(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x).transpose(1,2)





class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=False):
        super().__init__()
        layers = list()
        self.mlps = nn.ModuleList()
        self.out_layer = output_layer
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
            self.mlps.append(nn.Sequential(*layers))
            layers = list()
        if self.out_layer:
            self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        for layer in self.mlps:
            x = layer(x)
        if self.out_layer:
            x = self.out(x)
        return x





class controller_mlp(nn.Module):
    def __init__(self, args, input_dim, embed_dims):
        super().__init__()
        self.inputdim = input_dim
        self.mlp = MultiLayerPerceptron(input_dim=self.inputdim,
                                        embed_dims=embed_dims, output_layer=False, dropout=args.dropout)
        self.weight_init(self.mlp)
    
    def forward(self, emb_fields):
        input_mlp = emb_fields.flatten(start_dim=1).float()
        output_layer = self.mlp(input_mlp)
        return torch.softmax(output_layer, dim=1)

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)





class AdaFS_hard(nn.Module): 
    def __init__(self, args,
                 categorical_features: List, 
                 vector_features: List, 
                 categorical_field_dims: List, 
                 vector_input_dim: int,
                 embed_dim: int,
                 mlp_dims: List[int],
                 device):
        
        super().__init__()
        self.num_cat = len(categorical_features)
        self.num_vec = len(vector_features)
        self.num_total = self.num_cat + self.num_vec
        
        # Store feature names for reference
        self.categorical_features = categorical_features
        self.vector_features = vector_features
        
        self.embed_dim = embed_dim
        
        # Embedding for categorical features
        if self.num_cat > 0:
            self.emb = EMB(categorical_field_dims, self.embed_dim)
        
        # Linear projections for vector features (one per feature)
        if self.num_vec > 0:
            self.vec_projections = nn.ModuleList([
                nn.Linear(vector_input_dim, self.embed_dim) 
                for _ in range(self.num_vec)
            ])
        
        self.mlp = MultiLayerPerceptron(input_dim=self.num_total * self.embed_dim,
                                        embed_dims=mlp_dims, output_layer=True, dropout=args.dropout) # output layer
        
        self.controller = controller_mlp(args, input_dim=self.num_total * self.embed_dim, 
                                        embed_dims=[self.num_total])
        self.UseController = args.controller

        self.BN = nn.BatchNorm1d(self.embed_dim)
        self.useBN = args.useBN

        self.k = args.k

        self.useWeight = args.useWeight 
        self.reWeight = args.reWeight

        self.device = device
        self.stage = -1

    def forward(self, cat_field=None, vec_field=None):
        features = []
        
        # Process categorical features through embeddings
        if cat_field is not None and self.num_cat > 0:
            cat_emb = self.emb(cat_field)  # shape: (batch_size, embed_dim, num_cat)
            features.append(cat_emb)
        
        # Process vector features through individual linear layers
        if vec_field is not None and self.num_vec > 0:
            vec_embs = []
            for i in range(self.num_vec):
                # Apply linear projection to each vector feature
                vec_emb = self.vec_projections[i](vec_field[:, i, :])  # shape: (batch_size, embed_dim)
                vec_embs.append(vec_emb.unsqueeze(2))  # shape: (batch_size, embed_dim, 1)
            vec_embs = torch.cat(vec_embs, dim=2)  # shape: (batch_size, embed_dim, num_vec)
            features.append(vec_embs)
        
        # Concatenate all features
        field = torch.cat(features, dim=2)  # shape: (batch_size, embed_dim, num_total)

        # Apply batch normalization to each feature
        if self.useBN == True:
            field = self.BN(field)
        
        # Initialize mask to None
        mask = None
        
        if self.UseController and self.stage == 1:
            weight = self.controller(field) # shape: (batch_size, num_total)
            kmax_index, kmax_weight = kmax_pooling(weight, 1, self.k)
            if self.reWeight == True:
                kmax_weight = kmax_weight/torch.sum(kmax_weight,dim=1).unsqueeze(1) # reweight, make the sum equal to 1
            
            # Create a mask with the same dimensions as weight, assign values at index positions, others are 0
            mask = torch.zeros(weight.shape[0], weight.shape[1]).to(self.device)

            if self.useWeight:
                mask = mask.scatter_(1, kmax_index, kmax_weight) # fill the corresponding index position with weight values
            else:
                mask = mask.scatter_(1, kmax_index, torch.ones(kmax_weight.shape[0], kmax_weight.shape[1])) # fill the corresponding index position with 1

            field = field * torch.unsqueeze(mask, 1)    
        
        input_mlp = field.flatten(start_dim=1).float() # shape: (batch_size, num_total * embed_dim)
        res = self.mlp(input_mlp) # shape: (batch_size, 1)
        return torch.sigmoid(res.squeeze(1)), mask # shape: (batch_size,)
    
    def get_feature_names(self):
        return self.categorical_features + self.vector_features
    
    def get_mask_indices(self):
        return {
            'categorical': range(0, self.num_cat),
            'vector': range(self.num_cat, self.num_total),
            'categorical_names': self.categorical_features,
            'vector_names': self.vector_features
        }

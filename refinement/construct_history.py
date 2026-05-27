import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader

from profiler.AdaFS.dataset import AmazonDataset
from profiler.AdaFS.model import AdaFS_hard




def rank_subjectives(subjective_profiles_df, user_profiles_df, user_id, parent_asin, similarity_measure='dot'):

    assert similarity_measure in ['dot', 'cosine'], f"similarity_measure must be 'dot' or 'cosine', got: {similarity_measure}"

    if isinstance(parent_asin, list):
        return [
            rank_subjectives(subjective_profiles_df, user_profiles_df, user_id, asin, similarity_measure)
            for asin in parent_asin
        ]

    item_profiles = subjective_profiles_df[subjective_profiles_df['parent_asin'] == parent_asin].copy()
    user_profile = user_profiles_df[user_profiles_df['user_id'] == user_id]
    
    if len(user_profile) == 0:
        raise ValueError(f"User profile not found for user_id: {user_id}")
    if len(item_profiles) == 0:
        raise ValueError(f"No item profiles found for parent_asin: {parent_asin}")
    
    # Get user embedding
    user_embedding = np.array(user_profile.iloc[0]['embedding'])
    
    # Compute similarities
    similarities = []
    for _, row in item_profiles.iterrows():
        item_embedding = np.array(row['embedding'])
        
        if similarity_measure == 'dot':
            similarity = np.dot(user_embedding, item_embedding)
        elif similarity_measure == 'cosine':
            similarity = np.dot(user_embedding, item_embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding)
            )
        else:
            raise ValueError(f"Unknown similarity_measure: {similarity_measure}")
        
        similarities.append(similarity)
    
    # Add similarities to the dataframe
    item_profiles['similarity'] = similarities
    
    # Sort by similarity (descending)
    item_profiles = item_profiles.sort_values('similarity', ascending=False)
    
    # Convert to list of dicts
    result = item_profiles.to_dict(orient='records')
    
    return result




def select_objectives(fs_module, device,
                      user_ids, item_ids,
                      encoded_item_table_df, user_profiles_df,
                      vector_features_order, categorical_features_order, item_feature_names,
                      item_categorical_feature_names,
                      batch_size=65536):
    assert len(user_ids) == len(item_ids), (
        f"user_ids and item_ids must have the same length, "
        f"got {len(user_ids)} and {len(item_ids)}"
    )

    # Deduplicate pairs
    pair_set = list(dict.fromkeys(zip(user_ids, item_ids)))  # unique, order-preserved
    pair_user_ids = [p[0] for p in pair_set]
    pair_item_ids = [p[1] for p in pair_set]

    # Filter to present ASINs 
    all_unique_asins = set(pair_item_ids)
    encoded_subset = encoded_item_table_df[encoded_item_table_df['parent_asin'].isin(all_unique_asins)]
    present_asin_set = set(encoded_subset['parent_asin'].values)
    missing_asins = all_unique_asins - present_asin_set
    if missing_asins:
        print(f"WARNING: ASINs not found in encoded item table: {missing_asins}")
    asin_to_row = encoded_subset.set_index('parent_asin')

    # Pre-cache user embeddings 
    unique_user_ids = set(pair_user_ids)
    user_emb_cache = {}
    for uid in unique_user_ids:
        user_row = user_profiles_df[user_profiles_df['user_id'] == uid]
        if user_row.empty:
            raise ValueError(f"User {uid} not found in user_profiles_df")
        user_emb_cache[uid] = np.array(user_row.iloc[0]['embedding'], dtype=np.float32)

    # Build rows for every valid (user, item) pair 
    valid_pairs = []   # (user_id, asin) for rows actually built
    rows = []
    print("Building input rows for AdaFS controller...")
    for uid, asin in tqdm(pair_set, desc="Building input rows"):
        if asin not in present_asin_set:
            continue
        item_feats = asin_to_row.loc[asin]
        row = {
            'user_id':      0,   # placeholder categorical
            'parent_asin':  0,   # placeholder categorical
            'user_profile': user_emb_cache[uid],
            **{feat: np.array(item_feats[feat], dtype=np.float32) for feat in item_feature_names},
            **{feat: int(item_feats[feat]) for feat in item_categorical_feature_names},
            'label': 0,
        }
        rows.append(row)
        valid_pairs.append((uid, asin))

    inference_df = pd.DataFrame(rows)
    col_order = categorical_features_order + vector_features_order + ['label']
    inference_df = inference_df[col_order]

    # Create AmazonDataset & DataLoader 
    inference_dataset = AmazonDataset(inference_df)
    loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
    print(f"Built inference dataset: {len(inference_dataset)} (user, item) pairs")

    all_masks = []
    with torch.no_grad():
        for fields, _ in tqdm(loader, desc="Running AdaFS controller", mininterval=1.0):
            cat_f = fields['categorical'].long().to(device) if fields['categorical'].numel() > 0 else None
            vec_f = fields['vector'].to(device)             if fields['vector'].numel()       > 0 else None
            _, mask = fs_module(cat_field=cat_f, vec_field=vec_f)
            all_masks.append(mask.cpu())

    all_masks = torch.cat(all_masks, dim=0)    # (n_pairs, num_total_features)
    all_feat_names = fs_module.get_feature_names()  # categorical names + vector names

    # Report item-level vector features (excluding user_profile) and
    # item-level categorical features (excluding user_id and parent_asin)
    item_feat_set = set(item_feature_names) | set(item_categorical_feature_names)

    # Group results: user_id -> asin -> [selected feature names] 
    selected_features = {}
    for i, (uid, asin) in enumerate(valid_pairs):
        selected = sorted(
            [(all_feat_names[j], all_masks[i, j].item())
             for j in range(len(all_feat_names))
             if all_feat_names[j] in item_feat_set and all_masks[i, j].item() > 0],
            key=lambda x: -x[1]
        )
        if uid not in selected_features:
            selected_features[uid] = {}
        selected_features[uid][asin] = [feat for feat, _ in selected]

    return selected_features




def select_objectives_by_similarity(
    user_ids, item_ids,
    encoded_item_table_df, user_profiles_df,
    similarity_threshold=0.5,
):

    assert len(user_ids) == len(item_ids), (
        f"user_ids and item_ids must have the same length, "
        f"got {len(user_ids)} and {len(item_ids)}"
    )

    # Deduplicate pairs 
    pair_set = list(dict.fromkeys(zip(user_ids, item_ids)))
    pair_user_ids = [p[0] for p in pair_set]
    pair_item_ids = [p[1] for p in pair_set]

    # Filter to present ASINs 
    all_unique_asins = set(pair_item_ids)
    encoded_subset = encoded_item_table_df[encoded_item_table_df['parent_asin'].isin(all_unique_asins)]
    present_asin_set = set(encoded_subset['parent_asin'].values)
    missing_asins = all_unique_asins - present_asin_set
    if missing_asins:
        print(f"WARNING: ASINs not found in encoded item table: {missing_asins}")
    asin_to_row = encoded_subset.set_index('parent_asin')

    # Identify vector feature columns (numpy arrays, not integers) 
    sample_row = asin_to_row.iloc[0] if len(asin_to_row) > 0 else None
    if sample_row is None:
        return {}
    vector_feat_names = [
        col for col in asin_to_row.columns
        if isinstance(sample_row[col], (np.ndarray, list, torch.Tensor))
    ]

    # Pre-cache normalized user embeddings 
    unique_user_ids = set(pair_user_ids)
    user_emb_cache  = {}
    for uid in unique_user_ids:
        user_row = user_profiles_df[user_profiles_df['user_id'] == uid]
        if user_row.empty:
            raise ValueError(f"User {uid} not found in user_profiles_df")
        emb = np.array(user_row.iloc[0]['embedding'], dtype=np.float32)
        norm = np.linalg.norm(emb)
        user_emb_cache[uid] = emb / max(norm, 1e-8)

    # Compute similarities and select features 
    selected_features = {}
    for uid, asin in pair_set:
        if asin not in present_asin_set:
            continue
        item_row = asin_to_row.loc[asin]
        user_emb_n = user_emb_cache[uid]

        feat_embs = np.stack([
            np.array(item_row[feat], dtype=np.float32) for feat in vector_feat_names
        ])                                                      # (n_feats, 1024)
        norms = np.linalg.norm(feat_embs, axis=1, keepdims=True)
        feat_embs_n = feat_embs / np.clip(norms, 1e-8, None)  # (n_feats, 1024)
        sims = feat_embs_n @ user_emb_n                  # (n_feats,)

        selected = sorted(
            [(vector_feat_names[j], float(sims[j]))
             for j in range(len(vector_feat_names)) if sims[j] > similarity_threshold],
            key=lambda x: -x[1],
        )

        if uid not in selected_features:
            selected_features[uid] = {}
        selected_features[uid][asin] = [feat for feat, _ in selected]

    return selected_features




def generate_rerank_prompt(history_dict,
                           subjective_profiles_df, objective_profiles_df, user_profiles_df,
                           precomputed_masks,
                           rerank_prompt_template,
                           item_meta_df,
                           max_history_len=50):
    user_id = history_dict['user_id']
    history_asins = history_dict['history_asins'][-max_history_len:]
    item_ids = history_asins + history_dict['candidate_ids']
    asin_labels  = (['history'] * len(history_asins) +
                ['candidate'] * len(history_dict['candidate_ids']))

    all_subjectives_rank = rank_subjectives(subjective_profiles_df, user_profiles_df, user_id, item_ids)
    selected_subjectives = {profiles[0]['parent_asin']: f"{profiles[0]['name']}; {profiles[0]['description']}" 
                            for profiles in all_subjectives_rank}

    # Look up pre-computed masks for this user
    masks = precomputed_masks.get(user_id, {})

    # Build a lookup: asin -> row of raw string feature values
    obj_lookup = objective_profiles_df.set_index('parent_asin')

    # Build a lookup: asin -> title
    title_lookup = item_meta_df.set_index('parent_asin')['title'].to_dict()

    # Build a lookup: asin -> categorical metadata feature values
    meta_feat_lookup = item_meta_df.set_index('parent_asin')[
        ['price', 'average_rating', 'rating_number']
    ].to_dict(orient='index')

    def describe_item(asin):
        """Build a JSON-like string describing a single item."""
        selected_feat_names = masks.get(asin, [])
        obj_row = obj_lookup.loc[asin] if asin in obj_lookup.index else None

        parts = {}
        parts['Item Name'] = title_lookup.get(asin, asin)

        for feat in selected_feat_names:
            if obj_row is not None and feat in obj_row.index:
                value = obj_row[feat]
                parts[feat] = str(value) if value is not None else 'N/A'
            elif feat in meta_feat_lookup.get(asin, {}):
                value = meta_feat_lookup[asin][feat]
                parts[feat] = str(value) if value is not None else 'N/A'
            else:
                parts[feat] = 'N/A'

        parts['Feature that can appeal to the user'] = selected_subjectives.get(asin, 'N/A')

        lines = '\n'.join(f'    "{k}": "{v}"' for k, v in parts.items())
        return '{\n' + lines + '\n}'

    history_blocks = '\n\n'.join(describe_item(asin) for asin in history_asins)
    candidate_blocks = '\n\n'.join(describe_item(asin) for asin in history_dict['candidate_ids'])

    prompt = rerank_prompt_template.format(
        history=history_blocks,
        candidates=candidate_blocks
    )

    return prompt, history_blocks, candidate_blocks
    

    

import json
import yaml
import pandas as pd
import numpy as np

from tqdm import tqdm

from refinement.construct_history import rank_subjectives




def retrieve_refinement_dataset(subjective_profiles_df, user_profiles_df, paths_config, num_candidates=4):
    df = pd.read_json(paths_config['reviews_path'], lines=True)
    df['orig_index'] = range(len(df))

    # Pre-calculate the history indices excluding the current review
    # We group by user and for each user, collect the indices up to but not including the current row
    df['history_indices'] = None
    for user_id, group in df.groupby('user_id'):
        idx_list = group['orig_index'].tolist()
        for i, row_idx in enumerate(group.index):
            df.at[row_idx, 'history_indices'] = idx_list[:i]

    # Re-index by parent_asin to match your itemwise_dataset structure
    itemwise_dataset = df.groupby('parent_asin')['history_indices'].apply(list).to_dict()

    # Pre-compute the pool of all valid parent_asins from subjective profiles
    all_valid_asins = np.array(subjective_profiles_df['parent_asin'].unique())

    # Classify histories into item profiles based on user similarity
    profile_classified_dataset = {}

    for parent_asin, histories in tqdm(itemwise_dataset.items()):
        # Initialize the structure for this item
        profile_classified_dataset[parent_asin] = {}
        
        for history in histories:
            # Skip first-time reviews with no prior history
            if len(history) == 0:
                continue

            # Get the user who wrote the last (current) review
            last_review_idx = history[-1]
            user_id = df.iloc[last_review_idx]['user_id']
            
            # Rank subjective profiles for this user and item
            ranked_profiles = rank_subjectives(
                subjective_profiles_df, 
                user_profiles_df, 
                user_id, 
                parent_asin, 
                similarity_measure='dot'
            )
            
            # Get the best matching profile ID (first in ranked list)
            if ranked_profiles and len(ranked_profiles) > 0:
                best_profile_id = ranked_profiles[0]['ID']
                
                # Initialize the profile list if not exists
                if best_profile_id not in profile_classified_dataset[parent_asin]:
                    profile_classified_dataset[parent_asin][best_profile_id] = []
                
                # Collect parent_asins that appear in this history (ordered)
                history_asins = df.iloc[history]['parent_asin'].tolist()
                history_asins_set = set(history_asins)

                # Sample negative candidate_ids from valid asins not in this history
                # and not the target item itself (which is added explicitly)
                negative_pool = all_valid_asins[
                    ~np.isin(all_valid_asins, list(history_asins_set) + [parent_asin])
                ]
                sampled_negatives = np.random.choice(
                    negative_pool,
                    size=min(num_candidates, len(negative_pool)),
                    replace=False
                ).tolist()
                # Include the ground-truth item so the LLM can select it
                candidate_ids = [parent_asin] + sampled_negatives

                # Add the user, history, history_asins, and candidate_ids to this profile
                profile_classified_dataset[parent_asin][best_profile_id].append({
                    'user_id': user_id,
                    'history': history,
                    'history_asins': history_asins,
                    'candidate_ids': candidate_ids
                })
    
    with open(paths_config['refine_dataset_save_path'], 'w') as f:
        json.dump(profile_classified_dataset, f, indent=2)
    df.to_pickle(paths_config['refine_dataset_df_save_path'])
    
    # df (reviews) is returned as profile_classified_dataset contains indices to retrieve reviews from df
    return profile_classified_dataset, df




def read_refinement_dataset(paths_config):
    with open(paths_config['refine_dataset_save_path'], 'r') as f:
        profile_classified_dataset = json.load(f)
    df = pd.read_pickle(paths_config['refine_dataset_df_save_path'])
    return profile_classified_dataset, df




def filter_refinement_dataset(profile_classified_dataset, profile_min_histories=7, min_history_len=3):
    filtered_dataset = {}

    for parent_asin, profiles in profile_classified_dataset.items():
        filtered_dataset[parent_asin] = {}

        for profile_id, histories in profiles.items():
            # Filter out short histories
            filtered_histories = [h for h in histories if len(h['history']) >= min_history_len]

            # Keep only profiles with enough histories
            if len(filtered_histories) < profile_min_histories:
                continue

            filtered_dataset[parent_asin][profile_id] = filtered_histories

        # Remove items with no valid profiles
        if len(filtered_dataset[parent_asin]) == 0:
            del filtered_dataset[parent_asin]

    return filtered_dataset

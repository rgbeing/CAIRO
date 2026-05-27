import random

import pandas as pd

from refinement.construct_history import rank_subjectives

_OBJ_SKIP_COLS = {'parent_asin', 'title'}


def build_rerank_info(
    test_sample,
    subjective_profiles_df,
    objective_profiles_df,
    user_profiles_df,
    precomputed_masks,
    item_meta_df,
    max_history_len=50,
    user_profile_text=None,
    obj_for_sub=False,
    objective_ablation='selection',
    subjective_ablation='selection',
    raw_meta_lookup=None):

    user_id = test_sample['user_id']
    history_asins = test_sample['history'][-max_history_len:]
    candidate_asins = test_sample['candidates']

    all_asins = list(dict.fromkeys(history_asins + candidate_asins))  # unique, order-preserved

    # Subjective feature selection 
    selected_subjectives = {}           # asin -> list of display strings (for prompt)
    selected_sub_ids     = {}           # asin -> list of profile IDs (for saving)
    subjective_related_features = {}
    for asin in all_asins:
        try:
            if subjective_ablation == 'selection':
                profiles = rank_subjectives(subjective_profiles_df, user_profiles_df, user_id, asin)
                if profiles:
                    top = profiles[0]
                    selected_subjectives[asin] = [f"{top['name']}; {top['description']}"]
                    selected_sub_ids[asin]     = [top['ID']]
                    if obj_for_sub:
                        subjective_related_features[asin] = top.get('related_feature') or []
            elif subjective_ablation == 'random':
                item_profiles = subjective_profiles_df[
                    subjective_profiles_df['parent_asin'] == asin
                ].to_dict(orient='records')
                if item_profiles:
                    pick = random.choice(item_profiles)
                    selected_subjectives[asin] = [f"{pick['name']}; {pick['description']}"]
                    selected_sub_ids[asin]     = [pick['ID']]
            elif subjective_ablation == 'all':
                item_profiles = subjective_profiles_df[
                    subjective_profiles_df['parent_asin'] == asin
                ].to_dict(orient='records')
                if item_profiles:
                    selected_subjectives[asin] = [
                        f"{p['name']}; {p['description']}" for p in item_profiles
                    ]
                    selected_sub_ids[asin] = [p['ID'] for p in item_profiles]
        except (ValueError, KeyError, IndexError):
            pass  # item or user not found; fall back to 'N/A'

    # Lookups
    obj_lookup        = objective_profiles_df.set_index('parent_asin')
    title_lookup      = item_meta_df.set_index('parent_asin')['title'].to_dict()
    meta_feat_lookup  = item_meta_df.set_index('parent_asin')[
        ['price', 'average_rating', 'rating_number']
    ].to_dict(orient='index')
    user_masks        = precomputed_masks.get(user_id, {})

    item_obj_features = {}  # asin -> {'adafs': {...}, 'sub_related': {...}}

    def describe_item(asin):
        """Build an ASIN-keyed JSON-like block describing a single item."""
        obj_row = obj_lookup.loc[asin] if asin in obj_lookup.index else None
        parts = {}
        parts['Title'] = title_lookup.get(asin, asin)

        adafs_parts       = {}
        sub_related_parts = {}

        if objective_ablation == 'selection':
            adafs_feats    = user_masks.get(asin, [])
            sub_feats      = [f for f in subjective_related_features.get(asin, [])
                              if f not in set(adafs_feats)]
            chosen_feats   = list(adafs_feats) + sub_feats
            adafs_feat_set = set(adafs_feats)

            for feat in chosen_feats:
                value = None
                if obj_row is not None and feat in obj_row.index:
                    v = obj_row[feat]
                    if not pd.isna(v) and v is not None:
                        value = str(v)
                elif feat in meta_feat_lookup.get(asin, {}):
                    v = meta_feat_lookup[asin][feat]
                    if v is not None:
                        value = str(v)
                if value is not None:
                    parts[feat] = value
                    if feat in adafs_feat_set:
                        adafs_parts[feat] = value
                    else:
                        sub_related_parts[feat] = value

        elif objective_ablation == 'random':
            if obj_row is not None:
                candidate_cols = [c for c in obj_row.index if c not in _OBJ_SKIP_COLS]
                chosen_cols = random.sample(candidate_cols, min(5, len(candidate_cols)))
            else:
                chosen_cols = []
            for feat in chosen_cols:
                value = obj_row[feat]
                if not pd.isna(value) and value is not None:
                    parts[feat] = str(value)
                    adafs_parts[feat] = str(value)

        elif objective_ablation == 'all':
            if obj_row is not None:
                chosen_cols = [c for c in obj_row.index if c not in _OBJ_SKIP_COLS]
            else:
                chosen_cols = []
            for feat in chosen_cols:
                value = obj_row[feat]
                if not pd.isna(value) and value is not None:
                    parts[feat] = str(value)
                    adafs_parts[feat] = str(value)

        elif objective_ablation == 'raw':
            raw_str = (raw_meta_lookup or {}).get(asin, '')
            if raw_str:
                parts['Item Metadata'] = raw_str
                adafs_parts['Item Metadata'] = raw_str

        # Save objective features split by source track
        item_obj_features[asin] = {
            'adafs':       adafs_parts,
            'sub_related': sub_related_parts,
        }

        obj_lines = '\n'.join(f'"{k}": "{v}"' for k, v in parts.items())
        sub_texts = selected_subjectives.get(asin, ['N/A'])
        sub_lines = '\n'.join(
            f'"Feature that can appeal to the user": "{t}"' for t in sub_texts
        )
        inner = '{\n' + f'"parent_asin": "{asin}"\n' + obj_lines + '\n' + sub_lines + '\n}'
        return f'"{asin}": {inner}'

    history_blocks   = '\n\n'.join(f'{describe_item(asin)}' for history_num, asin in enumerate(history_asins))
    candidate_blocks = '\n\n'.join(f'{describe_item(asin)}' for i, asin in enumerate(candidate_asins))

    asin_list = '[' + ',\n'.join(f'"{asin}"' for asin in candidate_asins) + ']'

    format_kwargs = dict(
        history=history_blocks,
        candidates=candidate_blocks,
        asin_list=asin_list,
    )
    if user_profile_text is not None:
        format_kwargs['user_profile'] = user_profile_text

    asin_to_title = {asin: title_lookup.get(asin, asin) for asin in candidate_asins}

    return format_kwargs, asin_to_title, item_obj_features, selected_sub_ids, history_asins, candidate_asins


def build_rerank_prompt(
    test_sample,
    subjective_profiles_df,
    objective_profiles_df,
    user_profiles_df,
    precomputed_masks,
    rerank_prompt_template,
    item_meta_df,
    max_history_len=50,
    user_profile_text=None,
    obj_for_sub=False,
    objective_ablation='selection',
    subjective_ablation='selection',
    raw_meta_lookup=None,
):
    format_kwargs, asin_to_title, _, _, _, _ = build_rerank_info(
        test_sample,
        subjective_profiles_df,
        objective_profiles_df,
        user_profiles_df,
        precomputed_masks,
        item_meta_df,
        max_history_len=max_history_len,
        user_profile_text=user_profile_text,
        obj_for_sub=obj_for_sub,
        objective_ablation=objective_ablation,
        subjective_ablation=subjective_ablation,
        raw_meta_lookup=raw_meta_lookup,
    )
    prompt = rerank_prompt_template.format(**format_kwargs)
    return prompt, asin_to_title

"""
Reranking evaluation on the test set (ASIN-based variant).
Items are identified by parent_asin in prompts; the LLM returns ASINs directly.

Usage:
    python -m reranking.main [--device DEVICE] [--use-refined-profiles]
                                 [--batch-size BATCH_SIZE] [--no-batch]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import pandas as pd
import torch
import yaml
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI
from tqdm import tqdm

from extraction.subjective_profile import (
    generate_subjective_profiles_df,
    read_user_profiles,
)
from refinement.construct_history import select_objectives
from reranking.metrics import compute_metrics
from reranking.prompt_builder import build_rerank_info, build_rerank_prompt
from profiler.AdaFS.model import AdaFS_hard
from utils.openai_batch import ask_prompts


OBJ_FOR_SUB_THRESHOLD = 0.5   # cosine similarity threshold for Related_Feature selection
_OBJ_SKIP_COLS = {'parent_asin', 'title'}

###################################
#  Helpers
###################################

def _enrich_subjective_profiles(base_path, objective_profiles_df, text_embedding_model):
    enriched_path = base_path.parent / f"enriched_{base_path.name}"

    with open(base_path, 'r') as f:
        profiles_data = json.load(f)

    obj_lookup = objective_profiles_df.set_index('parent_asin')

    print(f"Enriching {len(profiles_data)} items' subjective profiles...")
    for item in tqdm(profiles_data, desc="Enriching profiles"):
        asin = item['parent_asin']

        if asin not in obj_lookup.index:
            for profile in item['profiles']:
                profile['Related_Feature'] = None
            continue

        obj_row = obj_lookup.loc[asin]

        # Collect non-null objective features, skipping metadata columns
        feat_names, feat_texts = [], []
        for col, val in obj_row.items():
            if col in _OBJ_SKIP_COLS:
                continue
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            val_str = str(val).strip()
            if not val_str or val_str.lower() in ('none', 'null', 'nan'):
                continue
            feat_names.append(col)
            feat_texts.append(f"{col}: {val_str}")

        if not feat_names:
            for profile in item['profiles']:
                profile['Related_Feature'] = None
            continue

        # Embed all profile texts and objective feature texts in one batch
        profile_texts = [
            f"{p.get('Name', '')}: {p.get('Description', '')}"
            for p in item['profiles']
        ]
        embeddings    = text_embedding_model.encode(profile_texts + feat_texts)['dense_vecs']
        profile_embs  = embeddings[:len(profile_texts)]
        feat_embs     = embeddings[len(profile_texts):]

        # Cosine similarity: (n_profiles, n_feats)
        profile_embs_n = profile_embs / np.clip(np.linalg.norm(profile_embs, axis=1, keepdims=True), 1e-8, None)
        feat_embs_n    = feat_embs    / np.clip(np.linalg.norm(feat_embs,    axis=1, keepdims=True), 1e-8, None)
        sim = profile_embs_n @ feat_embs_n.T   # (n_profiles, n_feats)

        for i, profile in enumerate(item['profiles']):
            related = [feat_names[j] for j in range(len(feat_names)) if sim[i, j] > OBJ_FOR_SUB_THRESHOLD]
            profile['Related_Feature'] = related if related else None

    with open(enriched_path, 'w') as f:
        json.dump(profiles_data, f, indent=2, ensure_ascii=False)
    
    print(f"Enriched profiles saved in {enriched_path}")


def _load_subjective_profiles_df(paths_config, text_embedding_model, use_refined=True, obj_for_sub=False):

    path_key  = 'refined_subjective_profiles_save_path' if use_refined else 'subjective_profiles_save_path'
    base_path = Path(paths_config[path_key])
    load_path = base_path.parent / f"enriched_{base_path.name}" if obj_for_sub else base_path

    with open(load_path, 'r') as f:
        profiles = json.load(f)

    rows = []
    for item in profiles:
        parent_asin = item['parent_asin']
        for profile in item['profiles']:
            rows.append({
                'parent_asin':        parent_asin,
                'ID':                 profile.get('ID'),
                'name':               profile.get('Name'),
                'description':        profile.get('Description'),
                'evidence_rationale': profile.get('Evidence_Rationale'),
                'related_feature':    profile.get('Related_Feature'),
            })
    df = pd.DataFrame(rows)
    texts = [f"{row['name']}: {row['description']}" for _, row in df.iterrows()]

    embeddings = text_embedding_model.encode(texts)['dense_vecs']
    df['embedding'] = embeddings.tolist()
    return df


def _load_adafs(paths_config, args, device):
    print("Loading FS dataset to recover feature order & model shape...")
    train_ds = torch.load(
        Path(paths_config['fs_dataset_save_path'].format(type='test')),
        weights_only=False,
    )
    vector_features_order      = train_ds.vector_features_name
    categorical_features_order = train_ds.categorical_features_name
    categorical_field_dims     = [
        int(train_ds.categorical_features[:, i].max()) + 1
        for i in range(train_ds.categorical_features.shape[1])
    ]
    vec_input_dim      = train_ds.vector_features.shape[2]
    item_feature_names = [f for f in vector_features_order if f != 'user_profile']
    del train_ds

    dataset_name = Path(paths_config['reviews_path']).parent.name
    ckpt_path    = Path(paths_config['fs_model_save_dir']) / f"{dataset_name}_controller.pt"

    fs_module = AdaFS_hard(
        args=args,
        categorical_features=categorical_features_order,
        vector_features=vector_features_order,
        categorical_field_dims=categorical_field_dims,
        vector_input_dim=vec_input_dim,
        embed_dim=16,
        mlp_dims=[64, 16],
        device=device,
    ).to(device)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    fs_module.load_state_dict(ckpt['state_dict'])
    fs_module.stage = 1
    fs_module.eval()
    print("AdaFS checkpoint loaded.")

    return fs_module, vector_features_order, categorical_features_order, item_feature_names


def _parse_ranking_asin(raw_response, candidate_asins, n_candidates):
    try:
        parsed  = json.loads(raw_response)
        ranking = parsed.get('ranking', [])
    except (json.JSONDecodeError, TypeError):
        ranking = []

    valid_asins  = set(candidate_asins)
    ranked_asins = []
    for entry in ranking:
        if isinstance(entry, dict):
            asin = entry.get('asin', '')
        elif isinstance(entry, str):
            asin = entry
        else:
            continue
        asin = asin.strip()
        if asin in valid_asins and asin not in ranked_asins:
            ranked_asins.append(asin)
    return ranked_asins


def _gt_rank(ranked_asins, gt_asin, n_candidates):
    try:
        return ranked_asins.index(gt_asin) + 1
    except ValueError:
        return n_candidates


###################################
#  Main
###################################

def main(args, device, client, paths_config):
    # Load embedding model 
    print("Loading BGE-M3 text embedding model...")
    text_embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # Load objective profiles 
    print("Loading objective profiles...")
    records = []
    with open(paths_config['objective_profiles_save_path'], 'r') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    objective_profiles_df = pd.DataFrame(records)

    # Enrich subjective profiles (if needed)
    effective_obj_for_sub = (
        args.obj_for_sub
        and (args.objective_ablation == 'selection')
        and (args.subjective_ablation == 'selection')
    )
    if effective_obj_for_sub:
        path_key     = 'refined_subjective_profiles_save_path' if args.use_refined_profiles else 'subjective_profiles_save_path'
        base_path    = Path(paths_config[path_key])
        enriched_path = base_path.parent / f"enriched_{base_path.name}"
        if not enriched_path.exists():
            print("Enriched profiles not found — computing now (one-time step)...")
            _enrich_subjective_profiles(base_path, objective_profiles_df, text_embedding_model)
        else:
            print(f"Enriched profiles found at {enriched_path}, skipping recomputation.")

    # Load subjective profiles
    print(f"Loading {'refined' if args.use_refined_profiles else 'original'} subjective profiles...")
    subjective_profiles_df = _load_subjective_profiles_df(
        paths_config, text_embedding_model, use_refined=args.use_refined_profiles,
        obj_for_sub=effective_obj_for_sub,
    )

    # Load encoded item table
    if args.objective_ablation == 'selection':
        encoded_item_table_df = pd.read_pickle(paths_config['encoded_item_table_save_path'])

    # Load user profiles 
    print("Loading user profiles...")
    user_profiles_df = read_user_profiles(paths_config)
    # Handle both 'embedding' and legacy 'user_profile_embedding' column name
    if 'user_profile_embedding' in user_profiles_df.columns:
        user_profiles_df.rename(columns={'user_profile_embedding': 'embedding'}, inplace=True)

    # Load item metadata (for titles) 
    print("Loading item metadata...")
    item_meta_df = pd.read_json(paths_config['data_path'], lines=True)[
        ['parent_asin', 'title', 'price', 'average_rating', 'rating_number']
    ]
    item_meta_df = item_meta_df.drop_duplicates(subset='parent_asin')

    # Load raw metadata lookup (raw ablation only)
    if args.objective_ablation == 'raw':
        print("Loading raw metadata lookup...")
        raw_meta_lookup = {}
        with open(paths_config['data_path'], 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    asin = obj.get('parent_asin')
                    if asin:
                        raw_meta_lookup[asin] = line[:1600]
    else:
        raw_meta_lookup = {}

    # Load test set 
    print("Loading test set...")
    test_samples = []
    with open(paths_config['test_path'], 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                sample['history'] = sample['history'][-args.max_history_len:]
                test_samples.append(sample)
    print(f"Loaded {len(test_samples)} test samples.")

    # Load AdaFS model and precompute feature masks (selection mode only) 
    if args.objective_ablation == 'selection':
        fs_module, vector_features_order, categorical_features_order, item_feature_names = \
            _load_adafs(paths_config, args, device)
        item_categorical_feature_names = [
            f for f in categorical_features_order if f not in ('user_id', 'parent_asin')
        ]

        print("Collecting (user, item) pairs for AdaFS precomputation...")
        all_user_ids = []
        all_item_ids = []
        for sample in test_samples:
            uid  = sample['user_id']
            asins = sample['history'] + sample['candidates']
            for asin in asins:
                all_user_ids.append(uid)
                all_item_ids.append(asin)

        print(f"Running select_objectives on {len(all_user_ids)} (user, item) pairs...")
        precomputed_masks = select_objectives(
            fs_module, device,
            all_user_ids, all_item_ids,
            encoded_item_table_df, user_profiles_df,
            vector_features_order, categorical_features_order, item_feature_names,
            item_categorical_feature_names,
            batch_size=args.adafs_batch_size,
        )
    else:
        precomputed_masks = {}

    # Load prompt template
    with open(paths_config['rerank_prompt_path'], 'r') as f:
        rerank_prompt_template = f.read()

    # Build user profile lookup
    user_profile_lookup = user_profiles_df.set_index('user_id')['user_profile'].to_dict()

    # Build prompts for all test samples
    print("Building reranking prompts...")
    prompts              = []
    asin_to_title_maps   = []
    all_format_kwargs    = []
    all_item_obj_features = []
    all_item_sub_features = []
    all_history_asins    = []
    all_candidate_asins  = []
    for sample in tqdm(test_samples, desc="Building prompts"):
        user_profile_text = user_profile_lookup.get(sample['user_id'])
        format_kwargs, asin_to_title, item_obj_features, selected_sub_ids, history_asins, candidate_asins = build_rerank_info(
            sample,
            subjective_profiles_df, objective_profiles_df, user_profiles_df,
            precomputed_masks, item_meta_df,
            user_profile_text=user_profile_text,
            obj_for_sub=effective_obj_for_sub,
            objective_ablation=args.objective_ablation,
            subjective_ablation=args.subjective_ablation,
            raw_meta_lookup=raw_meta_lookup,
        )
        prompt = rerank_prompt_template.format(**format_kwargs)
        prompts.append(prompt)
        asin_to_title_maps.append(asin_to_title)
        all_format_kwargs.append(format_kwargs)
        all_item_obj_features.append(item_obj_features)
        all_item_sub_features.append(selected_sub_ids)
        all_history_asins.append(history_asins)
        all_candidate_asins.append(candidate_asins)


    # Query LLM (chunked to avoid overwhelming the Batch API) 
    chunk_size = args.llm_batch_chunk_size
    n_chunks   = (len(prompts) + chunk_size - 1) // chunk_size
    print(f"Sending {len(prompts)} prompts to LLM in {n_chunks} chunk(s) of ≤{chunk_size}...")
    raw_responses = []
    for chunk_idx in range(n_chunks):
        chunk = prompts[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        print(f"  Chunk {chunk_idx + 1}/{n_chunks} ({len(chunk)} prompts)...")
        chunk_responses = ask_prompts(
            client, chunk,
            batch_process=not args.no_batch,
            response_format_type='json_object',
            temperature=0.2,
        )
        raw_responses.extend(chunk_responses)

    # Parse responses & compute per-sample rank 
    print("Parsing LLM responses...")
    gt_ranks = []
    results  = []
    for sample, raw, prompt in zip(test_samples, raw_responses, prompts):
        gt_asin      = sample['gt']
        n_candidates = len(sample['candidates'])
        ranked_asins = _parse_ranking_asin(raw, sample['candidates'], n_candidates)
        rank         = _gt_rank(ranked_asins, gt_asin, n_candidates)
        gt_ranks.append(rank)
        results.append({
            'user_id':      sample['user_id'],
            'gt':           gt_asin,
            'gt_rank':      rank,
            'n_candidates': n_candidates,
            'ranked_asins': ranked_asins,
            'prompt':       prompt,
            'raw_response': raw,
        })

    # Compute metrics 
    metrics = compute_metrics(gt_ranks, ks=(5, 10, 20))

    print("\n── Reranking Results ─────────────────────────────────────────────")
    col_w = 12
    header = f"{'Metric':<{col_w}}" + "".join(f"{'@'+str(k):>{col_w}}" for k in (5, 10, 20))
    print(header)
    print("-" * (col_w * 4))
    for metric_name in ('nDCG', 'Recall'):
        row = f"{metric_name:<{col_w}}"
        for k in (5, 10, 20):
            row += f"{metrics[f'{metric_name}@{k}']:>{col_w}.4f}"
        print(row)
    print(f"\nEvaluated on {len(gt_ranks)} test samples.")

    # Save results 
    out_path = paths_config.get('rerank_results_save_path')
    if out_path:
        output = {'metrics': metrics, 'results': results}
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank test set candidates with AdaFS + LLM (ASIN-based).")
    parser.add_argument('--device',               type=str,  default='0,1',
                        help="CUDA device index (default: 0,1)")
    parser.add_argument('--dropout',              type=float, default=0.2)
    parser.add_argument('--k',                    type=int,   default=5,
                        help="AdaFS top-k features to select")
    parser.add_argument('--controller',           action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--useWeight',            action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--reWeight',             action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--useBN',                action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use-refined-profiles', action=argparse.BooleanOptionalAction, default=True,
                        help="Use refined subjective profiles instead of original ones (use --no-use-refined-profiles to disable)")
    parser.add_argument('--adafs-batch-size',     type=int,   default=65536,
                        help="Batch size for AdaFS forward passes (default: 65536)")
    parser.add_argument('--no-batch',             action='store_true', default=False,
                        help="Use sequential (async) LLM calls instead of the Batch API")
    parser.add_argument('--llm-batch-chunk-size', type=int,   default=5000,
                        help="Max prompts per Batch API job (default: 5000)")
    parser.add_argument('--max_history_len',      type=int,   default=50,
                        help="Truncate each test sample's history to the most recent N items (default: 50)")
    parser.add_argument('--obj_for_sub', action='store_true',
                        help="Use objective profiles to supplement a subjective profile")
    parser.add_argument('--objective_ablation', type=str, default='selection', choices=['selection', 'random', 'all', 'raw', 'no'])
    parser.add_argument('--subjective_ablation', type=str, default='selection', choices=['selection', 'random', 'all', 'no'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    with open("paths_config.yaml", "r") as f:
        paths_config = yaml.safe_load(f)
    with open("llm_config.yaml", "r") as f:
        llm_config = yaml.safe_load(f)

    client = OpenAI(api_key=llm_config['openai_api_key'])
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    main(args, device, client, paths_config)

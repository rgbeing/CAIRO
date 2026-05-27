import argparse
import os
import random
import json
import pandas as pd
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

from FlagEmbedding import BGEM3FlagModel
import numpy as np

from extraction.subjective_profile import read_subjective_profiles, read_user_profiles, generate_subjective_profiles_df
from refinement.construct_dataset import retrieve_refinement_dataset, read_refinement_dataset, filter_refinement_dataset
from refinement.construct_history import rank_subjectives, select_objectives, generate_rerank_prompt
from profiler.AdaFS.model import AdaFS_hard
from profiler.train_fs import test
from utils.openai_batch import ask_prompts

from openai import OpenAI


#########################################
# Helper 
#########################################

def _parse_json_answer(raw: str) -> dict:
    """Best-effort parse of an LLM JSON response string into a dict."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def is_correct_answer(answer: dict, gt_item_title: str) -> bool:
    """
    Check if the answer from the LLM matches the ground truth item title.
    Assume `answer` is a dictionary containing a 'title' key with the predicted item title.
    """
    return answer.get('title', '').strip().lower() == gt_item_title.strip().lower()



# Evaluate a profile on a set of histories
def evaluate_profile(
    client,
    histories,
    parent_asin,
    subjective_profiles_df,
    objective_profiles_df,
    user_profiles_df,
    precomputed_masks,
    rerank_prompt_template,
    item_meta_df,
):

    # Build prompts
    prompts = []
    history_blocks_list = []
    candidate_blocks_list = []
    for h in histories:
        p, hb, cb = generate_rerank_prompt(
            h,
            subjective_profiles_df, objective_profiles_df, user_profiles_df,
            precomputed_masks, rerank_prompt_template, item_meta_df,
        )
        prompts.append(p)
        history_blocks_list.append(hb)
        candidate_blocks_list.append(cb)

    #print(prompts[0])

    # Query LLM
    raw_answers = ask_prompts(client, prompts, batch_process=False)

    # Resolve ground-truth title
    gt_row = item_meta_df[item_meta_df['parent_asin'] == parent_asin]
    gt_title = gt_row.iloc[0]['title'] if len(gt_row) > 0 else parent_asin

    # Score answers and collect errors
    n_correct = 0
    error_cases = []
    for h, prompt, hb, cb, raw in zip(histories, prompts, history_blocks_list, candidate_blocks_list, raw_answers):
        answer = _parse_json_answer(raw)
        if is_correct_answer(answer, gt_title):
            n_correct += 1
        else:
            error_cases.append({
                'history_dict': h,
                'prompt': prompt,
                'history_blocks': hb,
                'candidate_blocks': cb,
                'answer': answer,
                'gt_asin': parent_asin,
                'gt_title': gt_title,
            })

    accuracy = n_correct / len(histories) if histories else 0.0
    return accuracy, error_cases


#  Diagnose why errors occurred
def diagnose_errors(client, error_cases, error_reason_template, profile_name, profile_description):
    """
    For each error case, ask the LLM to explain what went wrong.

    Returns:
        list[str]: one diagnostic sentence per error case.
    """
    if not error_cases:
        return []

    diag_prompts = []
    for ec in error_cases:
        prediction = ec['answer'].get('title', 'N/A')
        gt = ec['gt_title']
        history_block = ec['history_blocks']
        candidate_block = ec['candidate_blocks']

        p = error_reason_template.format(
            history=history_block,
            candidates=candidate_block,
            prediction=prediction,
            gt=gt,
            profile_name=profile_name,
            profile_description=profile_description,
        )
        diag_prompts.append(p)

    #print(diag_prompts[0] if diag_prompts else "No error cases to diagnose.")

    # response_format_type=None because this prompt returns free-text sentences
    raw_responses = ask_prompts(
        client, diag_prompts, batch_process=False, response_format_type="text"
    )
    return [r.strip() if isinstance(r, str) else '' for r in raw_responses]




# Generate refined profile candidates
def generate_refined_profiles(
    client,
    current_name,
    current_description,
    weaknesses,
    refine_prompt_template,
    parent_asin,
    objective_profiles_df,
    n_refined_versions=5):
    weakness_text = '\n'.join(f'- {w}' for w in weaknesses) if weaknesses else '- No specific weaknesses identified.'

    # Build full objective profile text for the target item
    obj_row = objective_profiles_df[objective_profiles_df['parent_asin'] == parent_asin]
    if not obj_row.empty:
        obj_row = obj_row.iloc[0]
        objective_profile_text = '\n'.join(
            f'- {col}: {obj_row[col]}'
            for col in objective_profiles_df.columns
            if col != 'parent_asin' and obj_row[col] is not None
        )
    else:
        objective_profile_text = '- No objective profile available.'

    prompt = refine_prompt_template.format(
        name=current_name,
        description=current_description,
        weaknesses=weakness_text,
        n_versions=n_refined_versions,
        objective_profile=objective_profile_text,
    )

    #print(prompt if prompt is not None else "No prompt generated for refinement.")

    raw = ask_prompts(client, [prompt], batch_process=False)
    parsed = _parse_json_answer(raw[0]) if raw else {}
    profiles = parsed.get('refined_profiles', [])

    # Validate structure
    valid = []
    for p in profiles:
        if isinstance(p, dict) and 'name' in p and 'description' in p:
            valid.append({'name': p['name'], 'description': p['description']})
    return valid[:n_refined_versions]



#  Select the best profile via test-set evaluation
def select_best_profile_emb(
    original_profile,
    refined_profiles,
    histories,
    parent_asin,
    pid,
    subjective_profiles_df,
    user_profiles_df,
    text_embedding_model,
):


    # Build mean user embedding from all history owners
    user_embeddings = []
    for h in histories:
        uid = h['user_id']
        row = user_profiles_df[user_profiles_df['user_id'] == uid]
        if not row.empty:
            user_embeddings.append(np.array(row.iloc[0]['embedding'], dtype=np.float32))
    if not user_embeddings:
        # Fallback: return original if no user embeddings found
        return original_profile, 0.0
    mean_user_emb = np.mean(user_embeddings, axis=0)
    mean_user_emb = mean_user_emb / (np.linalg.norm(mean_user_emb) + 1e-10)

    # Embed all candidates in one batch
    candidates = [original_profile] + refined_profiles
    texts = [f"{c['name']}: {c['description']}" for c in candidates]
    candidate_embs = text_embedding_model.encode(texts)['dense_vecs']  # (K+1, 1024)

    # Pick candidate with highest cosine similarity
    best_sim = -float('inf')
    best_profile = original_profile
    for i, (cand, emb) in enumerate(zip(candidates, candidate_embs)):
        emb = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        emb = emb / (norm + 1e-10)
        sim = float(np.dot(mean_user_emb, emb))
        label = 'original' if i == 0 else f'refined-{i}'
        print(f"    [{label}] cosine similarity = {sim:.4f}")
        if sim > best_sim:
            best_sim = sim
            best_profile = cand

    # Persist winner into subjective_profiles_df
    _patch_profile(subjective_profiles_df, parent_asin, pid,
                   best_profile['name'], best_profile['description'])

    return best_profile, best_sim

def _patch_profile(subjective_profiles_df, parent_asin, pid, name, description):
    mask = (
        (subjective_profiles_df['parent_asin'] == parent_asin) &
        (subjective_profiles_df['ID'] == pid)
    )
    subjective_profiles_df.loc[mask, 'name'] = name
    subjective_profiles_df.loc[mask, 'description'] = description


def select_best_profile(
    client,
    original_profile,
    refined_profiles,
    histories,
    parent_asin,
    pid,
    subjective_profiles_df,
    objective_profiles_df,
    user_profiles_df,
    precomputed_masks,
    rerank_prompt_template,
    item_meta_df,
    original_accuracy=None,
):
    candidates = [original_profile] + refined_profiles  # index 0 = original

    best_acc = -1.0
    best_profile = original_profile

    for i, cand in enumerate(candidates):
        _patch_profile(subjective_profiles_df, parent_asin, pid,
                       cand['name'], cand['description'])

        if i == 0 and original_accuracy is not None:
            acc = original_accuracy
        else:
            acc, _ = evaluate_profile(
                client, histories, parent_asin,
                subjective_profiles_df, objective_profiles_df, user_profiles_df,
                precomputed_masks, rerank_prompt_template, item_meta_df,
            )

        label = 'original' if i == 0 else f'refined-{i}'
        print(f"    [{label}] accuracy = {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_profile = cand

    # Ensure the winning profile is written into the df
    _patch_profile(subjective_profiles_df, parent_asin, pid,
                   best_profile['name'], best_profile['description'])

    return best_profile, best_acc


##############################################
#  Top-level refinement loop
##############################################

def refine_single_profile(
    client,
    parent_asin,
    pid,
    histories,
    subjective_profiles_df,
    objective_profiles_df,
    user_profiles_df,
    precomputed_masks,
    rerank_prompt_template,
    error_reason_template,
    refine_prompt_template,
    item_meta_df,
    n_refined_versions=3,
    n_refine_iter=2,
    max_histories_per_iter=50,
    update_by='llm',
    text_embedding_model=None):
    # Look up current profile text
    profile_row = subjective_profiles_df[
        (subjective_profiles_df['parent_asin'] == parent_asin) &
        (subjective_profiles_df['ID'] == pid)
    ]
    if profile_row.empty:
        print(f"  WARNING: profile ({parent_asin}, {pid}) not found – skipping.")
        return None, False

    current_name = profile_row.iloc[0]['name']
    current_description = profile_row.iloc[0]['description']

    for iteration in range(n_refine_iter):
        print(f"  == Iteration {iteration + 1}/{n_refine_iter} ==")

        # Sample histories for this iteration
        k = min(max_histories_per_iter, len(histories))
        iter_histories = random.sample(histories, k)

        # Evaluate on sampled histories
        train_acc, error_cases = evaluate_profile(
            client, iter_histories, parent_asin,
            subjective_profiles_df, objective_profiles_df, user_profiles_df,
            precomputed_masks, rerank_prompt_template, item_meta_df,
        )
        print(f"  Accuracy: {train_acc:.3f}  ({len(error_cases)} errors, {k} histories)")

        if not error_cases:
            print("  No errors on train set — stopping early.")
            return {'name': current_name, 'description': current_description}, True

        # Diagnose errors, collect weaknesses
        weaknesses = diagnose_errors(client, error_cases, error_reason_template,
                                     profile_name=current_name,
                                     profile_description=current_description)
        print(f"  Collected {len(weaknesses)} weakness diagnostics.")

        # Generate refined profile candidates
        refined = generate_refined_profiles(
            client, current_name, current_description,
            weaknesses, refine_prompt_template,
            parent_asin, objective_profiles_df,
            n_refined_versions=n_refined_versions,
        )
        print(f"  Generated {len(refined)} refined profile candidates.")

        if not refined:
            print("  LLM returned no valid refined profiles — stopping early.")
            return {'name': current_name, 'description': current_description}, False

        # Select the best profile on the same histories
        original = {'name': current_name, 'description': current_description}
        if update_by == 'emb':
            best_profile, best_score = select_best_profile_emb(
                original, refined, iter_histories,
                parent_asin, pid,
                subjective_profiles_df, user_profiles_df, text_embedding_model,
            )
            print(f"  Best profile: \"{best_profile['name']}\" (cosine sim {best_score:.4f})")
        else:
            best_profile, best_score = select_best_profile(
                client, original, refined, iter_histories,
                parent_asin, pid,
                subjective_profiles_df, objective_profiles_df, user_profiles_df,
                precomputed_masks, rerank_prompt_template, item_meta_df,
                original_accuracy=train_acc,
            )
            print(f"  Best profile: \"{best_profile['name']}\" (test acc {best_score:.3f})")
        current_name = best_profile['name']
        current_description = best_profile['description']

    return {'name': current_name, 'description': current_description}, True




def main(text_embedding_model, args, device, client, paths_config):
    # read subjective profiles
    subjective_profiles = read_subjective_profiles(paths_config)
    subjective_profiles_df = generate_subjective_profiles_df(subjective_profiles, text_embedding_model, paths_config)

    # read objective profiles
    records = []
    with open(paths_config['objective_profiles_save_path'], 'r') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    objective_profiles_df = pd.DataFrame(records)

    # read encoded version of objective profiles
    encoded_item_table_df = pd.read_pickle(paths_config['encoded_item_table_save_path'])

    # read user profiles
    user_profiles_df = read_user_profiles(paths_config)
    user_profiles_df.rename(columns={'user_profile_embedding': 'embedding'}, inplace=True)

    # generate refinement dataset
    constructed_dataset, df = retrieve_refinement_dataset(subjective_profiles_df, user_profiles_df, paths_config)
    #constructed_dataset, df = read_refinement_dataset(paths_config)

    # init fs module — recover model shape from saved train dataset
    # Load FS test dataset to recover feature order & model shape
    print("Loading FS test dataset to recover feature order & model shape...")
    _train_ds = torch.load(
        Path(paths_config['fs_dataset_save_path'].format(type='test')), weights_only=False
    )
    vector_features_order = _train_ds.vector_features_name              # ['user_profile', feat1, ...]
    categorical_features_order = _train_ds.categorical_features_name    # ['user_id', 'parent_asin']
    categorical_field_dims = [int(_train_ds.categorical_features[:, i].max()) + 1 for i in range(_train_ds.categorical_features.shape[1])]
    vec_input_dim = _train_ds.vector_features.shape[2]                  # 1024
    item_feature_names = [f for f in vector_features_order if f != 'user_profile']
    item_categorical_feature_names  = [f for f in categorical_features_order if f not in ('user_id', 'parent_asin')]
    del _train_ds

    # Init model & load controller checkpoint
    print("Initializing AdaFS model and loading checkpoint...")
    dataset_name = Path(paths_config['reviews_path']).parent.name
    ckpt_path = Path(paths_config['fs_model_save_dir']) / f"{dataset_name}_controller.pt"

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
    print("AdaFS Checkpoint loaded.")

    # Load prompt templates
    with open(paths_config['rerank_to_refine_prompt_path'], 'r') as f:
        rerank_prompt_template = f.read()
    with open(paths_config['infer_error_reason_prompt_path'], 'r') as f:
        error_reason_template = f.read()
    with open(paths_config['refine_profile_prompt_path'], 'r') as f:
        refine_prompt_template = f.read()

    # Load item metadata (parent_asin -> title) for answer verification
    item_meta_df = pd.read_json(paths_config['data_path'], lines=True)[
        ['parent_asin', 'title', 'price', 'average_rating', 'rating_number']
    ]
    item_meta_df = item_meta_df.drop_duplicates(subset='parent_asin')

    # Filter dataset to qualifying profiles
    filtered_dataset = filter_refinement_dataset(constructed_dataset)
    print(f"Filtered dataset: {len(filtered_dataset)} items with qualifying profiles.")

    # Collect all (user_id, item_id) pairs from filtered_dataset
    all_user_ids = []
    all_item_ids = []
    for parent_asin, profiles in filtered_dataset.items():
        for pid, histories in profiles.items():
            for current_history in histories:
                uid = current_history['user_id']
                for asin in current_history['history_asins'] + current_history['candidate_ids']:
                    all_user_ids.append(uid)
                    all_item_ids.append(asin)

    # Run select_objectives once for all pairs
    print(f"Running select_objectives on {len(all_user_ids)} (user, item) pairs...")
    precomputed_masks = select_objectives(
        fs_module, device,
        all_user_ids, all_item_ids,
        encoded_item_table_df, user_profiles_df,
        vector_features_order, categorical_features_order, item_feature_names,
        item_categorical_feature_names,
        batch_size=65536
    )

    # Refine each profile
    out_path = paths_config['refined_subjective_profiles_save_path']
    ckpt_path = paths_config['refine_checkpoint_save_path']

    def _build_nested_profiles(df):
        nested = []
        for parent_asin, group in df.groupby('parent_asin'):
            profiles = [
                {
                    'ID': row['ID'],
                    'Name': row['name'],
                    'Description': row['description'],
                    'Evidence_Rationale': row['evidence_rationale'],
                }
                for _, row in group.iterrows()
            ]
            nested.append({'parent_asin': parent_asin, 'profiles': profiles})
        return nested

    def _save_profiles(label=""):
        nested = _build_nested_profiles(subjective_profiles_df)
        with open(out_path, 'w') as f:
            json.dump(nested, f, indent=2)
        suffix = f" ({label})" if label else ""
        print(f"Profiles saved to {out_path}{suffix}")

    # Load or initialise checkpoint
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path, 'r') as f:
            completed = set(tuple(pair) for pair in json.load(f))
        print(f"Resuming: {len(completed)} profiles already completed.")
    else:
        completed = set()
        with open(ckpt_path, 'w') as f:
            json.dump([], f)

    try:
        for parent_asin in tqdm(filtered_dataset.keys(), desc="Refining profiles"):
            for pid, histories in filtered_dataset[parent_asin].items():
                if (parent_asin, pid) in completed:
                    print(f"\nSkipping profile ({parent_asin}, {pid}) — already completed.")
                    continue

                print(f"\nRefining profile ({parent_asin}, {pid})  histories={len(histories)}")
                _, success = refine_single_profile(
                    client,
                    parent_asin, pid,
                    histories,
                    subjective_profiles_df, objective_profiles_df, user_profiles_df,
                    precomputed_masks, rerank_prompt_template,
                    error_reason_template, refine_prompt_template,
                    item_meta_df,
                    update_by=args.update_by,
                    text_embedding_model=text_embedding_model,
                )

                if success:
                    completed.add((parent_asin, pid))
                    with open(ckpt_path, 'w') as f:
                        json.dump([list(pair) for pair in completed], f)
    finally:
        _save_profiles()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--controller', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--useWeight', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--reWeight', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--useBN', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--update_by', type=str, default='llm', choices=['llm', 'emb'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    text_embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    with open("paths_config.yaml", "r") as f:
        paths_config = yaml.safe_load(f)
    
    with open("llm_config.yaml", "r") as f:
        llm_config = yaml.safe_load(f)
    client = OpenAI(
        api_key=llm_config['openai_api_key']
    )
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    main(text_embedding_model, args, device, client, paths_config)

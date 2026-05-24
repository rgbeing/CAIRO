import json

import numpy as np
import torch
import pandas as pd

from utils.process_string import strip_markdown_code_fences
from extraction.objective_profile import read_key
from extraction.subjective_profile import read_user_profiles
from profiler.convert_num_cat import convert_numeric_to_categorical


# Feature names added directly from item metadata (not from LLM extraction pipeline)
TEXT_META_FEATURES = ['main_category', 'store', 'title']
NUMERIC_META_FEATURES = ['average_rating', 'rating_number', 'price']


def get_null_text_embedding(text_embedding_model):
    """Encode the string 'No information' once to use as the null imputation vector."""
    output = text_embedding_model.encode(["No information"], batch_size=1, max_length=2048)
    vector = output['dense_vecs'][0]
    return vector


def _encode_text_feature_list(raw_list, text_embedding_model, no_info_vector):
    # Gather: filter Nones
    valid_entries = [(i, str(val)) for i, val in enumerate(raw_list) if val is not None]

    if not valid_entries:
        # All None: return no_info_vector
        no_info_tensor = torch.tensor(no_info_vector) if not isinstance(no_info_vector, torch.Tensor) else no_info_vector
        return [no_info_tensor] * len(raw_list)

    valid_indices, valid_strings = zip(*valid_entries)

    # Encode
    output = text_embedding_model.encode(list(valid_strings), batch_size=24, max_length=2048)
    dense_vectors = output['dense_vecs']

    # Scatter
    result = [None] * len(raw_list)
    for index, vector in zip(valid_indices, dense_vectors):
        result[index] = vector  # type: ignore

    # Impute Nones with no_info_vector 
    no_info_tensor = torch.tensor(no_info_vector) if not isinstance(no_info_vector, torch.Tensor) else no_info_vector
    return [
        no_info_tensor if v is None else (torch.tensor(v) if not isinstance(v, torch.Tensor) else v)
        for v in result
    ]


def _coerce_to_float_or_none(value):
    """Try to cast value to float. Return None if it is None or conversion fails."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _encode_numeric_feature_list(raw_list, feature_name):
    # Coerce each value to float or None
    coerced = [_coerce_to_float_or_none(v) for v in raw_list]

    # Separate valid entries from missing ones
    valid_entries = [(i, v) for i, v in enumerate(coerced) if v is not None]

    result = [-1] * len(raw_list)  # -1 = missing placeholder

    if valid_entries:
        valid_indices, valid_values = zip(*valid_entries)
        categorized = convert_numeric_to_categorical(np.array(valid_values, dtype=np.float64), feature_name)
        for index, cat_val in zip(valid_indices, categorized):
            result[index] = int(cat_val)

    # Shift all values by +1: missing (-1) -> 0, valid categories (0, 1, ...) -> (1, 2, ...)
    result = [v + 1 for v in result]
    return result


def convert_objective_profiles_to_table(paths_config, text_embedding_model):
    print("Convert objective profiles to embeddings")

    # Encode "No information" once for null imputation
    no_info_vector = get_null_text_embedding(text_embedding_model)

    # Load objective feature keys (LLM-extracted text features)
    features = json.loads(strip_markdown_code_fences(read_key(paths_config)))['features']
    llm_feature_names = [feature['feature_name'] for feature in features]

    # Load objective profiles (flat format)
    profiles_list = []
    with open(paths_config['objective_profiles_save_path'], 'r') as fp:
        for line in fp:
            profiles_list.append(json.loads(line.strip()))
    print(f"Loaded {len(profiles_list)} profiles from {paths_config['objective_profiles_save_path']}")

    parent_asins = [item["parent_asin"] for item in profiles_list]

    encoded_feature_table = {}

    # LLM-extracted text features
    print("Encoding LLM-extracted text features")
    for name in llm_feature_names:
        raw_list = [record.get(name) for record in profiles_list]
        encoded_feature_table[name] = _encode_text_feature_list(raw_list, text_embedding_model, no_info_vector)

    # Metadata text features: main_category, store
    print("Encoding metadata text features (main_category, store)")
    for name in TEXT_META_FEATURES:
        raw_list = [record.get(name) for record in profiles_list]
        encoded_feature_table[name] = _encode_text_feature_list(raw_list, text_embedding_model, no_info_vector)

    # Metadata numeric features: average_rating, rating_number, price
    print("Encoding metadata numeric features (average_rating, rating_number, price)")
    for name in NUMERIC_META_FEATURES:
        raw_list = [record.get(name) for record in profiles_list]
        encoded_feature_table[name] = _encode_numeric_feature_list(raw_list, name)

    encoded_feature_table["parent_asin"] = parent_asins
    df = pd.DataFrame(encoded_feature_table)

    # Save encoded feature table
    df.to_pickle(paths_config['encoded_item_table_save_path'])
    print(f"Saved encoded feature table to {paths_config['encoded_item_table_save_path']}")

    return df




def read_objective_profiles_embeddings(paths_config):
    df = pd.read_pickle(paths_config['encoded_item_table_save_path'])
    return df




def convert_user_profiles_to_embeddings(paths_config, text_embedding_model):
    print("Convert user profiles to embeddings")

    # Load user profile df
    user_df = read_user_profiles(paths_config)

    # BGE-M3 Embedding
    profile_texts = user_df['user_profile'].tolist()
    encoded_vectors = text_embedding_model.encode(profile_texts, batch_size=24, max_length=2048)['dense_vecs']
    user_df['user_profile'] = encoded_vectors

    user_df.to_pickle(paths_config['encoded_user_table_save_path'])
    print(f"Saved encoded user profile table to {paths_config['encoded_user_table_save_path']}")

    return user_df



def read_user_profiles_embeddings(paths_config):
    df = pd.read_pickle(paths_config['encoded_user_table_save_path'])
    return df

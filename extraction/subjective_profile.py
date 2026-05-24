import json
from pathlib import Path

import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel

from utils.openai_batch import ask_prompts




def read_reviews(review_path):
    reviews = []

    with open(review_path, 'r') as fp:
        for line in fp:
            reviews.append(json.loads(line.strip()))

    reviews_df = pd.DataFrame(reviews)

    return reviews_df




def generate_review_aspects(client, reviews, dataset_domain, paths_config, chunk_size=5000):
    # Read the prompt template
    aspect_extraction_prompt_file = paths_config['review_aspects_prompt_path']
    with open(aspect_extraction_prompt_file, 'r') as fp:
        aspect_extraction_prompt_template = fp.read()

    # Define JSON schema for review aspects
    review_aspects_schema = {
        "name": "review_aspects_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "category_preference_reason": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ]
                },
                "category_preference": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ]
                },
                "purchase_purpose_reason": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ]
                },
                "purchase_purpose": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ]
                },
                "quality_criteria_reason": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ]
                },
                "quality_criteria": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ]
                },
                "usage_context_reason": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ]
                },
                "usage_context": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ]
                }
            },
            "required": [
                "category_preference_reason",
                "category_preference",
                "purchase_purpose_reason",
                "purchase_purpose",
                "quality_criteria_reason",
                "quality_criteria",
                "usage_context_reason",
                "usage_context"
            ],
            "additionalProperties": False
        }
    }

    # Build prompts list and keep track of df indices
    index_list = list(reviews.index)
    prompts = []
    for idx, row in reviews.iterrows():
        review_str = json.dumps({"title": row['title'], "text": row["text"]})
        prompt = aspect_extraction_prompt_template.format(
            dataset_domain=dataset_domain,
            review=review_str
        )
        prompts.append(prompt)

    # Ask the prompts in chunks, keying results by df index
    answers = {}   # {df_index: answer_string}

    for i in range(0, len(prompts), chunk_size):
        print("Processing prompts {} to {}".format(i, min(i+chunk_size, len(prompts))))
        chunk_indices = index_list[i:i+chunk_size]
        chunk_prompts = prompts[i:i+chunk_size]
        chunk_answers = ask_prompts(
            client, 
            chunk_prompts, 
            'gpt-4o-mini', 
            batch_process=True, 
            json_schema=review_aspects_schema
        )
        for df_idx, answer in zip(chunk_indices, chunk_answers):
            answers[df_idx] = answer

    save_path = paths_config['review_aspects_save_path']
    with open(save_path, "w") as f:
        json.dump(answers, f, indent=2)

    return answers




def read_review_aspects(paths_config):
    with open(paths_config['review_aspects_save_path'], "r") as f:
        review_aspects = json.load(f)
    # JSON saves dict keys as strings; restore to int to match DataFrame index
    return {int(k): v for k, v in review_aspects.items()}




def generate_subjective_profiles(client, reviews, review_aspects, dataset_domain, item_objectives_df, paths_config, chunk_size=8000):
    subject_profile_prompt_file = paths_config['subjective_profile_prompt_path']
    with open(subject_profile_prompt_file, 'r') as fp:
        subject_profile_prompt_template = fp.read()

    # Define JSON schema for subjective profiles
    subjective_profiles_schema = {
        "name": "subjective_profiles_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "profiles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ID": {
                                "type": "string"
                            },
                            "Name": {
                                "type": "string"
                            },
                            "Description": {
                                "type": "string"
                            },
                            "Evidence_Rationale": {
                                "type": "string"
                            }
                        },
                        "required": ["ID", "Name", "Description", "Evidence_Rationale"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["profiles"],
            "additionalProperties": False
        }
    }

    review_aspects_df = pd.DataFrame({
        "parent_asin": reviews['parent_asin'].values,
        "user_id": reviews['user_id'].values,
        "review_title": reviews['title'].values,
        "review_text": reviews['text'].values,
        "aspects": reviews.index.map(review_aspects)
    })

    profile_generating_df = (
        review_aspects_df
        .groupby("parent_asin")["aspects"]
        .apply(lambda x: "\n".join(map(str, x)))
        .reset_index(name="review_extractions")
    )

    item_objectives_df['features'] = (
        item_objectives_df.set_index('parent_asin')
        .apply(lambda row: json.dumps(row.to_dict()), axis=1)
        .values
    )
    item_objectives_df = item_objectives_df[['parent_asin', 'features']]
    
    merged_df = item_objectives_df.merge(
        profile_generating_df,
        on="parent_asin",
        how="right"
    )

    profile_generating_prompts = []
    parent_asins = []
    for _, row in merged_df.iterrows():
        prompt = subject_profile_prompt_template.format(
            dataset_domain=dataset_domain,
            item_summary=row['features'],
            review_extractions=row['review_extractions']
        )
        profile_generating_prompts.append(prompt)
        parent_asins.append(row['parent_asin'])

    generated_profiles = []
    for i in range(0, len(profile_generating_prompts), chunk_size):
        print("Processing prompts {} to {}".format(i, min(i + chunk_size, len(profile_generating_prompts))))
        chunk_prompts = profile_generating_prompts[i:i + chunk_size]
        chunk_results = ask_prompts(
            client,
            chunk_prompts,
            model='gpt-4o-mini',
            batch_process=True,
            json_schema=subjective_profiles_schema
        )
        generated_profiles.extend(chunk_results)
    

    with open(paths_config['subjective_profiles_save_path'] + '.raw.txt', "w") as f:
        raw_data = [
            {
                'parent_asin': asin,
                'raw_response': profile
            }
            for asin, profile in zip(parent_asins, generated_profiles)
        ]
        json.dump(raw_data, f, indent=2)
    
    # Pair each generated profile with its parent_asin
    # Parse the JSON and extract the profiles array to avoid redundancy
    generated_profiles = [
        {
            'parent_asin': asin,
            'profiles': json.loads(profile)['profiles']
        }
        for asin, profile in zip(parent_asins, generated_profiles)
    ]
    
    # save the generated profiles
    with open(paths_config['subjective_profiles_save_path'], "w") as f:
        json.dump(generated_profiles, f, indent=2)
    
    return generated_profiles




def read_subjective_profiles(paths_config):
    with open(paths_config['subjective_profiles_save_path'], "r") as f:
        subjective_profiles = json.load(f)
    
    return subjective_profiles




def generate_subjective_profiles_df(subjective_profiles, text_embedding_model, paths_config):
    # Normalize subjective_profiles into a DataFrame (1NF)
    rows = []
    for item in subjective_profiles:
        parent_asin = item['parent_asin']
        for profile in item['profiles']:
            rows.append({
                'parent_asin': parent_asin,
                'ID': profile.get('ID'),
                'name': profile.get('Name'),
                'description': profile.get('Description'),
                'evidence_rationale': profile.get('Evidence_Rationale')
            })

    subjective_profiles_df = pd.DataFrame(rows)
    
    profile_texts = [
        f"{row['name']}: {row['description']}" 
        for _, row in subjective_profiles_df.iterrows()
    ]

    embeddings = text_embedding_model.encode(profile_texts)['dense_vecs']
    subjective_profiles_df['embedding'] = embeddings.tolist()

    return subjective_profiles_df




def generate_user_profiles(reviews, review_aspects, paths_config):
    text_embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    reviews["review_extractions"] = reviews.index.map(review_aspects)

    user_profile_df = reviews.groupby('user_id')['review_extractions'].apply(
        lambda x: ';'.join(x.astype(str))
    ).reset_index()
    user_profile_df.columns = ['user_id', 'user_profile']

    # Generate embeddings for user profiles
    user_profiles = user_profile_df['user_profile'].tolist()
    embeddings = text_embedding_model.encode(user_profiles)['dense_vecs']
    user_profile_df['embedding'] = embeddings.tolist()

    with open(paths_config['user_profiles_save_path'], "w") as f:
        json.dump(user_profile_df.to_dict(orient='records'), f, indent=2)

    return user_profile_df




def generate_user_profiles_text(client, reviews, review_aspects, dataset_domain, paths_config, chunk_size=8000):
    text_embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    with open(paths_config['user_profile_prompt_path'], 'r') as fp:
        user_profile_prompt_template = fp.read()

    reviews["review_extractions"] = reviews.index.map(review_aspects)

    user_profile_df = reviews.groupby('user_id')['review_extractions'].apply(
        lambda x: ';'.join(x.astype(str))
    ).reset_index()
    user_profile_df.columns = ['user_id', 'review_extractions']

    prompts = [
        user_profile_prompt_template.format(
            dataset_domain=dataset_domain,
            review_extractions=row['review_extractions']
        )
        for _, row in user_profile_df.iterrows()
    ]

    generated_texts = []
    for i in range(0, len(prompts), chunk_size):
        print("Processing prompts {} to {}".format(i, min(i + chunk_size, len(prompts))))
        chunk_prompts = prompts[i:i + chunk_size]
        chunk_results = ask_prompts(
            client,
            chunk_prompts,
            model='gpt-4o-mini',
            batch_process=True,
            response_format_type="text"
        )
        generated_texts.extend(chunk_results)

    user_profile_df['user_profile'] = generated_texts
    user_profile_df = user_profile_df[['user_id', 'user_profile']]

    embeddings = text_embedding_model.encode(generated_texts)['dense_vecs']
    user_profile_df['embedding'] = embeddings.tolist()

    with open(paths_config['user_profiles_save_path'], "w") as f:
        json.dump(user_profile_df.to_dict(orient='records'), f, indent=2)

    return user_profile_df




def read_user_profiles(paths_config):
    with open(paths_config['user_profiles_save_path'], "r") as f:
        user_profiles = json.load(f)
    user_profiles_df = pd.DataFrame(user_profiles)
    return user_profiles_df




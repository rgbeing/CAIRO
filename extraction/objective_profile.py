import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from FlagEmbedding import BGEM3FlagModel

from utils.openai_batch import ask_prompts
from utils.process_string import strip_markdown_code_fences




################################
# Functions for Key Init
################################




def cluster_items(item_descriptions, paths_config, device, n_clusters=20):
    # Designate visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    print(f"Device to use in clustering: {device}")

    # BGE-M3 Encoding
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    embeddings_raw = model.encode(item_descriptions, 
                            batch_size=16, 
                            max_length=8192,
                            )
    embeddings = embeddings_raw['dense_vecs']

    # Clustering items
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embeddings)
    # labels is your array of cluster assignments
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    print("Cluster sizes:", cluster_sizes)
    np.save(paths_config['cluster_save_path'], labels)

    return labels




def read_cluster_items(paths_config):
    """Read previously saved cluster labels."""
    labels = np.load(paths_config['cluster_save_path'])
    print(f"Loaded cluster labels from {paths_config['cluster_save_path']}")

    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    print("Cluster sizes:", cluster_sizes)

    return labels




def aggregate_candidate_keys(client, feature_init_answers, dataset_domain, paths_config):
    with open(paths_config['feature_agg_prompt_path'], 'r', encoding='utf-8') as f:
        feature_agg_prompt_template = f.read()

        expected_answer_json_schema = {
        "name": "feature_list",
        "strict": True,
        "schema":{
            "type": "object",
            "properties": {
                "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "feature_name": {
                        "type": "string"
                    },
                    "feature_description": {
                        "type": "string"
                    },
                    "reason_of_importance": {
                        "type": "string"
                    }
                    },
                    "required": [
                    "feature_name",
                    "feature_description",
                    "reason_of_importance"
                    ],
                    "additionalProperties": False
                }
                }
            },
            "required": ["features"],
            "additionalProperties": False
        }
    }
        
    print("Aggregating candidate keys within each cluster")

    # Iterative aggregation parameters
    chunk_size = 10
    iteration = 0
    #current_answers = [ans for ans in feature_init_answers if ans is not None]  # Filter out None values

    print("\n========================================")
    print(f"ITERATIVE AGGREGATION")
    print(f"Starting with {len(feature_init_answers)} answers")
    print(f"Chunk size: {chunk_size}")
    print("========================================\n")

    # Store all iterations for inspection
    current_answers = feature_init_answers.copy()
    all_iterations = [current_answers]

    # Keep aggregating until we have only 1 answer
    while len(current_answers) > 1:
        iteration += 1
        aggregated_answers = []
        
        print("\n========================================")
        print(f"ITERATION {iteration}")
        print("========================================\n")

        # Build prompts for all chunks
        chunk_prompts = []
        for i in range(0, len(current_answers), chunk_size):
            chunk = current_answers[i:i+chunk_size]
            prompt = feature_agg_prompt_template.format(
                dataset_domain=dataset_domain,
                features_sets=chunk
            )
            chunk_prompts.append(prompt)

        # Query the API for all chunks concurrently
        aggregated_answers = ask_prompts(client, chunk_prompts, batch_process=False, response_format_type='json_object', json_schema=expected_answer_json_schema)

        # Update current_answers for next iteration
        current_answers = aggregated_answers
        all_iterations.append(current_answers)
        
        print(f"\n Iteration {iteration} complete: {len(current_answers)} aggregated answers")
    
    print("\n========================================")
    print(f"AGGREGATION COMPLETE")
    print("========================================\n")

    # The final answer
    final_answer = current_answers[0]
    # Save the final answer
    with open(paths_config['final_key_save_path'], 'w') as f:
        json.dump({
            'final_answer': final_answer,
            'total_iterations': iteration,
            'initial_count': len(all_iterations[0]),
            'chunk_size': chunk_size
        }, f, indent=2)

    # Save all iterations for inspection
    with open(paths_config['all_iter_save_path'], 'w') as f:
        json.dump({
            f'iteration_{i}': answers 
            for i, answers in enumerate(all_iterations)
        }, f, indent=2)

    print(f"Final answer saved to '{paths_config['final_key_save_path']}'")
    print(f"All iterations saved to '{paths_config['all_iter_save_path']}'")

    return final_answer



    
def generate_candidate_keys_per_cluster(client, dataset_domain, item_descriptions, labels, paths_config, max_chunk_size=100):
    expected_answer_json_schema = {
        "name": "feature_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature_name": {"type": "string"},
                            "feature_description": {"type": "string"},
                            "reason_of_importance": {"type": "string"}
                        },
                        "required": ["feature_name", "feature_description", "reason_of_importance"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["features"],
            "additionalProperties": False
        }
    }

    print("Generating candidate keys per cluster")

    with open(paths_config['feature_init_prompt_path'], 'r', encoding='utf-8') as f:
        feature_init_prompt_template = f.read()

    all_prompts = []
    prompt_to_cluster = []

    for cluster_id in np.unique(labels):
        cluster_id = int(cluster_id)
        cluster_indices = np.where(labels == cluster_id)[0]
        print(f"Cluster {cluster_id}: {len(cluster_indices)} items")

        n_chunks = int(np.ceil(len(cluster_indices) / max_chunk_size))
        chunks = np.array_split(cluster_indices, n_chunks)
        print(f"  → split into {len(chunks)} chunks")

        for chunk_idx, chunk in enumerate(chunks):
            sampled_descriptions = [item_descriptions[i][:2869] for i in chunk]
            prompt = feature_init_prompt_template.format(
                number_of_items=len(sampled_descriptions),
                dataset_domain=dataset_domain,
                metadata=sampled_descriptions
            )
            all_prompts.append(prompt)
            prompt_to_cluster.append((cluster_id, chunk_idx))

    print(f"Total prompts to send: {len(all_prompts)}")
    print("Sending prompts to LLM...")

    batch_size = 10000
    all_responses = []
    for i in range(0, len(all_prompts), batch_size):
        prompt_batch = all_prompts[i:i + batch_size]
        batch_responses = ask_prompts(
            client,
            prompt_batch,
            batch_process=False,
            response_format_type='json_object',
            json_schema=expected_answer_json_schema
        )
        all_responses.extend(batch_responses)

    candidate_keys_by_cluster = {}
    for (cluster_id, chunk_idx), response in zip(prompt_to_cluster, all_responses):
        if cluster_id not in candidate_keys_by_cluster:
            candidate_keys_by_cluster[cluster_id] = []
        candidate_keys_by_cluster[cluster_id].append(response)

    with open(paths_config['candidate_key_save_path'], 'w') as f:
        json.dump(candidate_keys_by_cluster, f, indent=2)

    print(f"Saved raw candidate keys to {paths_config['candidate_key_save_path']}")
    return candidate_keys_by_cluster




def aggregate_keys_per_cluster(client, dataset_domain, candidate_keys_by_cluster, paths_config):
    expected_answer_json_schema = {
        "name": "feature_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature_name": {"type": "string"},
                            "feature_description": {"type": "string"},
                            "reason_of_importance": {"type": "string"}
                        },
                        "required": ["feature_name", "feature_description", "reason_of_importance"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["features"],
            "additionalProperties": False
        }
    }

    print("Aggregating candidate keys within each cluster")

    with open(paths_config['feature_agg_prompt_path'], 'r', encoding='utf-8') as f:
        feature_agg_prompt_template = f.read()

    cluster_agg_keys = {}
    chunk_size = 10

    for cluster_id in sorted(candidate_keys_by_cluster.keys()):
        cluster_id = int(cluster_id)
        current_answers = list(candidate_keys_by_cluster[cluster_id])
        print(f"\nCluster {cluster_id}: {len(current_answers)} candidate(s)")

        if len(current_answers) == 1:
            cluster_agg_keys[cluster_id] = current_answers[0]
            continue

        iteration = 0
        while len(current_answers) > 1:
            iteration += 1
            print(f"  Iteration {iteration}: {len(current_answers)} -> {(len(current_answers) + chunk_size - 1) // chunk_size}")
            chunk_prompts = []
            for i in range(0, len(current_answers), chunk_size):
                chunk = current_answers[i:i + chunk_size]
                prompt = feature_agg_prompt_template.format(
                    dataset_domain=dataset_domain,
                    features_sets=chunk
                )
                chunk_prompts.append(prompt)
            current_answers = ask_prompts(
                client,
                chunk_prompts,
                batch_process=False,
                response_format_type='json_object',
                json_schema=expected_answer_json_schema
            )

        cluster_agg_keys[cluster_id] = current_answers[0]
        print(f"  Aggregation complete")

    with open(paths_config['cluster_agg_key_save_path'], 'w') as f:
        json.dump(cluster_agg_keys, f, indent=2)

    print(f"\nSaved cluster aggregated keys to {paths_config['cluster_agg_key_save_path']}")

    ordered_keys = [cluster_agg_keys[i] for i in sorted(cluster_agg_keys.keys())]
    return ordered_keys




def initialize_key_per_cluster(client, dataset_domain, item_descriptions, paths_config, device, max_chunk_size=100):
    print("Initializing objective profile key (per-cluster strategy)...")

    print("\n=== Clustering items ===")
    labels = cluster_items(item_descriptions, paths_config, device, n_clusters=20)

    print("\n=== Generating candidate keys per cluster ===")
    candidate_keys_by_cluster = generate_candidate_keys_per_cluster(
        client, dataset_domain, item_descriptions, labels, paths_config, max_chunk_size
    )

    print("\n=== Aggregating candidate keys within each cluster ===")
    cluster_level_keys = aggregate_keys_per_cluster(
        client, dataset_domain, candidate_keys_by_cluster, paths_config
    )

    print("\n=== Aggregating all cluster-level keys to global key ===")
    final_key = aggregate_candidate_keys(
        client, cluster_level_keys, dataset_domain, paths_config
    )

    print("Objective profile key initialization complete.")
    return final_key




def read_key(paths_config):
    """Read previously saved final key."""
    with open(paths_config['final_key_save_path'], 'r') as f:
        data = json.load(f)
    
    final_key = data['final_answer']
    print(f"Loaded final key from {paths_config['final_key_save_path']}")
    print(f"Total iterations: {data.get('total_iterations', 'N/A')}")
    print(f"Initial count: {data.get('initial_count', 'N/A')}")
    return final_key




################################
# Functions for Key Value Extraction
################################




def extract_key_values(client, dataset_domain, item_descriptions, item_ids, aggregated_keys, paths_config):
    EXTRACT_ANYWAY = "- Even if a feature value is not directly specified, if you can easily infer a feature value from the description, you are encouraged to put the feature value with a corresponding reason.\n- Use null ONLY when there is insufficient evidence to extract or infer."

    print("Loading objective profile keys")
    final_answer_str = strip_markdown_code_fences(aggregated_keys)
    features = json.loads(final_answer_str)

    with open(paths_config['extract_keys_prompt_path'], 'r', encoding='utf-8') as f:
        extract_prompt_template = f.read()

    # Build JSON schema with dynamic properties
    # Accept any property name, each must have feature_value and feature_value_reason (string or null)
    extracted_values_schema = {
        "name": "extracted_values_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "extracted_values": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature_name": {
                                "type": "string"
                            },
                            "feature_value": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "null"}
                                ]
                            },
                            "feature_value_reason": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "null"}
                                ]
                            }
                        },
                        "required": ["feature_name", "feature_value", "feature_value_reason"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["extracted_values"],
            "additionalProperties": False
        }
    }

    first_extractions = []

    # Step 1: Create batch requests file
    prompts = []
    for idx, item_description in enumerate(item_descriptions):
        prompt = extract_prompt_template.format(
        dataset_domain = dataset_domain,
        #extract_only_grounded = EXTRACT_ANYWAY,
        item_description = item_description,
        features = features
        )
        
        prompts.append(prompt)

    # Use json_schema for strict enforcement of structure
    # Note: feature_value is restricted to string or null (complex values should be JSON strings)
    first_extractions = ask_prompts(
        client, 
        prompts, 
        'gpt-4o-mini', 
        batch_process=True,
        json_schema=extracted_values_schema
    )

    # Process extractions to add parent_asin
    extractions_with_ids = []
    for idx, extraction in enumerate(first_extractions):
        try:
            # Parse the extraction JSON string
            extraction_obj = json.loads(extraction)
            # Convert extracted_values array to flat {feature_name: {feature_value, feature_value_reason}} dict
            flat_features = {
                item["feature_name"]: {
                    "feature_value": item["feature_value"],
                    "feature_value_reason": item["feature_value_reason"]
                }
                for item in extraction_obj["extracted_values"]
            }
            # Add parent_asin at the beginning
            extraction_with_id = {
                "parent_asin": {"feature_value": item_ids[idx]},
                **flat_features
            }
            extractions_with_ids.append(extraction_with_id)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for item_id {item_ids[idx]}: {extraction}")
        except TypeError:
            print(f"Skipping item_id {item_ids[idx]}: extraction is None")
    
    with open(paths_config['extracted_values_save_path'], 'w') as fp:
        for extraction_with_id in extractions_with_ids:
            fp.write(json.dumps(extraction_with_id) + '\n')

    print(f"Saved {len(extractions_with_ids)} extractions to {paths_config['extracted_values_save_path']}")
    print(f"Length of item_id: {len(item_ids)}")
    return extractions_with_ids




def read_key_values(paths_config):
    """Read previously saved extracted key values."""
    extractions_with_ids = []
    with open(paths_config['extracted_values_save_path'], 'r') as fp:
        for line in fp:
            extraction_with_id = json.loads(line)
            extractions_with_ids.append(extraction_with_id)
    
    print(f"Loaded {len(extractions_with_ids)} extractions from {paths_config['extracted_values_save_path']}")
    return extractions_with_ids




def generate_objective_profiles(extracted_key_values, item_metadata_df, paths_config):
    item_objectives_df = pd.DataFrame([
        {k: v['feature_value'] for k, v in item.items() if v is not None}
        for item in extracted_key_values
    ])

    meta_cols = item_metadata_df[["parent_asin", "title", "main_category", "average_rating", "rating_number", "price", "store"]]
    item_objectives_df = item_objectives_df.merge(meta_cols, on="parent_asin", how="left")
    for col in ["average_rating", "rating_number", "price"]:
        item_objectives_df[col] = pd.to_numeric(item_objectives_df[col], errors="coerce")

    item_objectives_df.to_json(paths_config['objective_profiles_save_path'], orient='records', lines=True)

    return item_objectives_df




def read_objective_profiles(paths_config):
    item_metadata_df = pd.read_json(paths_config['objective_profiles_save_path'], lines=True)
    return item_metadata_df



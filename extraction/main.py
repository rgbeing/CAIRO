import argparse
import json
import yaml

import pandas as pd
from openai import OpenAI
from extraction.objective_profile import initialize_key_per_cluster, extract_key_values, generate_objective_profiles
from extraction.subjective_profile import read_reviews, generate_review_aspects, generate_subjective_profiles, generate_user_profiles_text


def main(args):  
    ###### Settings

    # Read llm config
    with open('./llm_config.yaml', 'r') as f:
        llm_config = yaml.safe_load(f)
    
    # Initialize OpenAI client
    API_KEY = llm_config['openai_api_key']
    client = OpenAI(api_key=API_KEY)

    # Read paths_config.yaml
    with open('./paths_config.yaml', 'r') as f:
        paths_config = yaml.safe_load(f)

    # Read item description
    print("Loading item metadata")
    item_metadata = []                              # item metadata, as a list of dictionaries
    with open(paths_config['data_path'], 'r') as fp:
        for line in fp:
            item_metadata.append(json.loads(line.strip()))

    item_metadata_df = pd.DataFrame(item_metadata)  # item metadata, as a dataframe

    item_descriptions = []                          # selected item metadata, as a list of json strings
    for idx, item in item_metadata_df[["title", "features", "description", "details"]].iterrows():
        item_descriptions.append(item.to_json())  

    ###### Main process

    # Find appropriate keys
    final_key = initialize_key_per_cluster(
        client=client,
        dataset_domain=args.dataset_domain,
        item_descriptions=item_descriptions,
        paths_config=paths_config,
        device=args.device
    )

    # Extract key values
    extracted_key_values = extract_key_values(
        client=client,
        dataset_domain=args.dataset_domain,
        item_descriptions=item_descriptions,
        item_ids=item_metadata_df['parent_asin'].values,
        aggregated_keys=final_key,
        paths_config=paths_config
    )

    item_objectives_df = generate_objective_profiles(
        extracted_key_values=extracted_key_values, 
        item_metadata_df=item_metadata_df, 
        paths_config=paths_config)
    print(item_objectives_df)

    # Generate subjective profiles
    reviews_df = read_reviews(paths_config['reviews_path'])
    review_aspects = generate_review_aspects(
        client=client,
        reviews=reviews_df,
        dataset_domain=args.dataset_domain,
        paths_config=paths_config,
        chunk_size=20000
    )
    subjective_profiles = generate_subjective_profiles(
        client=client,
        reviews=reviews_df,
        review_aspects=review_aspects,
        dataset_domain=args.dataset_domain,
        item_objectives_df=item_objectives_df,
        paths_config=paths_config,
        chunk_size=20000
    )

    # Generate user profiles
    user_profiles = generate_user_profiles_text(
        client=client,
        reviews=reviews_df,
        review_aspects=review_aspects,
        dataset_domain=args.dataset_domain,
        paths_config=paths_config,
        chunk_size=20000
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_domain', type=str, help="Domain of the dataset")
    parser.add_argument('--device', type=str, default="0,1,2,3", help="CUDA device(s) to use for embedding")
    args = parser.parse_args()

    main(args)

# CAIRO

This repository contains the implementation for **CAIRO**, from **"Profiling What Matters: Context-Aware Item Profiles from Large-Scale Metadata for LLM Recommenders"**.

## Overview

This repository contains four main components:
1. **Extraction**: Extracts object features and subject traits for each item.
    * Under `extraction` folder
2. **Profiler**: Trains the objective feature selector and subjective trait selector.
    * Under `profiler` folder
3. **Reranking**: Final reranking.
    * Under `reranking` folder
4. **Refinement**: Optional stage which refines subjective traits.
    * Under `refinement` folder

### Prompt Templates

We have the prompt templates we use under `prompts` folder.

* $p_\text{rerank}$: `rerank.txt`
* $p_\text{explore-keys}$: `explore-keys.txt`
* $p_\text{reassess-\&-merge}$: `reassess-merge.txt`
* $p_\text{extract-obj}$: `extract-obj.txt`
* $p_\text{extract-sub}$: `extract-sub.txt`
* $p_\text{extract-user-profile}$: `extract-user-profile.txt`
* $p_\text{analyze-error}$: `analyze-error.txt`
* $p_\text{refine-trait}$: `refine-trait.txt`

Additionally we have two more prompt templates:

* `extract-aspects.txt`: Used in extracting review aspects from each review.
* `rerank-proxy.txt`: Used in the proxy task in the refinement stage.

## How to Run

### Prerequisites

`requirements.txt` lists the project dependencies.
Install the dependencies in your environment:
```Python
pip install -r requirements.txt
```

Also, put your openai API key under `llm_config.yaml` file.

### Data

Due to large sizes of data, we could not provide data directly.
Instead we are providing a small number of datum in each data, which shows the format of each file.

### Main Framework Execution

Once you are located at the base directory of this repository, you can run the framework by sequentially running the following scripts:
```Shell
# Run Extraction
./scripts/run_extraction.sh
# Run Profiler
./scripts/run_profiler.sh
# Run Refinement
./scripts/run_refinement.sh
# Run Reranking
./scripts/run_reranking.sh
```

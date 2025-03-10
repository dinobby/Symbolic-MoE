# [Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Heterogeneous Reasoning](https://arxiv.org/abs/2503.05641)

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2503.05641)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Justin Chih-Yao Chen*](https://dinobby.github.io/) | [Sukwon Yun*](https://sukwonyun.github.io/) | [Elias Stengel-Eskin*](https://esteng.github.io/) | [Tianlong Chen](https://tianlong-chen.github.io/) | [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

*Equal contribution

<div align="center">
  <img width="1000" alt="Symbolic Mixture-of-Experts Architecture" src="https://i.imgur.com/83vc7bz.png">
</div>

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
  - [Shared Arguments](#shared-arguments)
  - [Step-by-Step Guide](#step-by-step-guide)
- [Quick Start with Pre-Generated Outputs](#quick-start-with-pre-generated-outputs)
- [Citation](#citation)

## Overview
This repository contains the implementation of Symbolic Mixture-of-Experts, a novel approach for adaptive skill-based routing to enable scalable heterogeneous reasoning across multiple domains.

## Installation
This repository is tested on Python 3.10.12. All dependencies can be installed as follows:

```bash
pip install -r requirements.txt
```

## Running Experiments

### Shared Arguments
- `--task`: Specifies which dataset to run. Options include:
  - `MMLU_Pro`
  - `AIME24`
  - `GPQA`
  - `MedMCQA`
- `--gpus`: Number of GPUs to use for the experiment
- `--seed`: Random seed for reproducibility

### Step-by-Step Guide

#### Step 1: Annotate Keywords
Annotate keywords for the validation and testing data:
```bash
python annotate_keywords.py
```

#### Step 2: Create Model Profiles
Create profiles for all models:
1. Edit `run_create_profile.sh` to ensure that the `--task` and `--gpus` arguments are correct
2. Run the script:
```bash
bash run_create_profile.sh
```

#### Step 3: Create Aggregator Benchmark
Create the aggregator benchmark for all models:
1. Edit `run_bench_aggr.sh` to ensure that the `--task` and `--gpus` arguments are correct
2. Run the script:
```bash
bash run_aggr_bench.sh
```

#### Step 4: Recruit Experts
Recruit experts for each instance:
```bash
python recruit_agents.py --task GPQA --seed 0
```

#### Step 5: Generate Initial Responses
Generate the initial responses from the k experts:
```bash
CUDA_VISIBLE_DEVICES=0 python expert_inference.py --task GPQA --gpus 1 --seed 0
```

#### Step 6: Aggregate Outputs
Use the aggregator to generate the final output and evaluate the results:
```bash
CUDA_VISIBLE_DEVICES=0 python aggregate.py --task GPQA --aggregator QwenR1 --gpus 1 --seed 0
```

The aggregator we use for each task can be found in the following table (and Table 9 in the paper):

| Dataset | Model |
|------|-------|
| MMLU-Pro | LlamaR1 |
| AIME | QwenR1 |
| GPQA | QwenR1 |
| MedMCQA | Exaone |

## Quick Start with Pre-Generated Outputs
You can skip Steps 1-4 by downloading the stored outputs here ([Google Drive](https://drive.google.com/file/d/1niCr2H7eec7CDMM6cvyK3pJoEXOFkLa1/view?usp=sharing)). 

Extract all files in `outputs.zip` and placing all the folders in the root directory of this project:

```bash
unzip outputs.zip
```

After extraction, you can proceed directly to Steps 5-7 to complete the experiment.

## Citation
If you find this work useful, please consider citing us:

```bibtex
@article{chen2025symbolic,
  title={Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Scalable Heterogeneous Reasoning},
  author={Chen, Justin Chih-Yao and Yun, Sukwon and Stengel-Eskin, Elias and Chen, Tianlong and Bansal, Mohit},
  journal={arXiv preprint arXiv:2503.05641},
  year={2025}
}
```

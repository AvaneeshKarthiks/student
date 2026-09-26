# Business Entity Resolution Pipeline

This document describes how to run the business entity resolution software.
The system uses a two-stage filter-and-refine architecture.

## 1. System Requirements

- Operating System: Linux, Windows, or macOS
- Python version: 3.8 or higher
- GPU: NVIDIA GPU with CUDA support recommended

## 2. Installation

Install the required software libraries.
Run this command in your terminal:

```bash
pip install -r requirements.txt
```

## 3. Directory Structure

Ensure the files are in the correct directories:

```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
└── code/
    └── business_entity_resolution/
        ├── requirements.txt
        ├── README.md
        └── src/
```

## 4. How to Run the Pipeline

Execute the master pipeline to process test records and generate submission files.

Run the following command from the root directory:

```bash
python -m code.business_entity_resolution.src.run_pipeline \
    --test-dir dataset/test \
    --output-dir output \
    --top-k 30 \
    --threshold 0.85 \
    --device cuda
```

If you do not have a GPU, set `--device cpu`.

## 5. Pipeline Stages

The pipeline performs five sequential operations:

1. **Data Preprocessing**:
   Cleans text and standardizes corporate suffixes.
   Normalizes addresses for US, India, and France without external APIs.

2. **Candidate Generation (Blocking)**:
   ModernBERT-base creates dense vector embeddings.
   A fast FAISS search finds top candidates for each Source 1 entity.
   The stage writes `output/candidate_pairs.tsv`.

3. **Deep Precision Matching**:
   DeBERTa-v3-large evaluates candidate pairs using the Ditto format.
   The model calculates match probabilities for each candidate pair.

4. **F_0.5 Threshold Filtering**:
   An asymmetric threshold filters out low-confidence pairs.
   This step aggressively prevents false merges.

5. **Verified Merge Clustering**:
   Applies multi-match consistency checks.
   Protects unmatched entities as singletons.
   Writes final matches to `output/matching_results.tsv`.

## 6. How to Validate Outputs

Run the competition validator to verify the format of your output files:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The script prints `PASS` when the submission files are correct.

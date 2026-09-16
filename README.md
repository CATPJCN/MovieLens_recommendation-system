# Hybrid Recommendation Engine

### Project Overview
##### Top-10 Personalized Movie Ranking on MovieLens 1M
  A multi-paradigm recommender combining <b>Content-Based Vectorization</b>, <b>Global Bias Decomposition</b>, and <b>Latent Factor Matrix Factorization (SVD)</b> via a <b>Dirichlet-optimized weighted blend</b>.

---

## Highlights

- **1M Interaction Scale:** Processes 1,000,209 ratings across 6,040 users and 3,883 movies.
- **Tri-Model Ensemble:** Blends orthogonal signals—genre affinity vectors, user/item rating baselines, and low-rank latent user/item embeddings.
- **2x Hit Rate Improvement:** From our testing, Ensemble achieves **185 hits** (0.31% Precision@10) on a 1-item hold-out validation task, nearly doubling the performance of the strongest individual baseline.
- **Zero Data Leakage:** Explicit filtering guarantees users are never re-recommended movies they rated in the training set.

---

## Pipeline Overview

```
                      MovieLens 1M Ingestion
              (train.csv, movies.dat, users.dat)
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
  Content-Based          Global Bias         Latent Factor (SVD)
  Genre Profiles       User/Item Deviations   Matrix Factorization
   (Cosine Sim)          (r = μ + bu + bi)     (Surprise / SGD)
         │                    │                    │
         └─────────────┬──────┴────────────────────┘
                       ▼
            Score Normalization [0, 1]
                       │
                       ▼
          Dirichlet Random Search (30 trials)
             Optimal Weight Blend Formulation
                       │
                       ▼
          Candidate Rejection (Exclude Seen)
                       │
                       ▼
            Top-10 Ranker & Precision@10
```

---

## Model Components

### 1. Content-Based Filtering (Genre Preference Profiles)
- Encodes movie metadata into 18-dimensional binary genre feature vectors $\mathbf{v}_i$.
- Builds user preference profiles as a rating-weighted average across all historically rated titles:

$$\mathbf{u} = \frac{\sum_{i \in I_u} r_{ui} \cdot \mathbf{v}_i}{\sum_{i \in I_u} r_{ui}}$$

- Computes cosine similarity between user vector $\mathbf{u}$ and unseen movie vectors $\mathbf{v}_j$, scaling scores into $[0, 1]$ via $\frac{\text{sim} + 1}{2}$.

### 2. Global Baseline Bias Model
- Decomposes baseline expectations to isolate individual calibration variance:

$$\hat{r}_{ui} = \mu + b_u + b_i$$

  where $\mu$ is overall dataset mean, $b_u = \bar{r}_u - \mu$, and $b_i = \bar{r}_i - \mu$.
- Predictions over unseen user-item pairs are clipped and normalized:

$$\text{norm}_{\text{score}} = \text{clip}\left(\frac{\hat{r}_{ui} - 1.0}{4.0}, 0.0, 1.0\right)$$

### 3. Latent Factor Collaborative Filtering (SVD)
- Factorizes sparse user-item interaction matrix using Stochastic Gradient Descent (SGD).
- **Configuration:**
  - Latent dimensions (`n_factors`): `150`
  - Optimization epochs (`n_epochs`): `10`
  - Learning rate (`lr_all`): `0.01`
  - Regularization (`reg_all`): `0.1`
- Infers dot-product interactions $\hat{r}_{ui} = \mu + b_u + b_i + q_i^T p_u$ and rescales estimates to $[0, 1]$.

### 4. Dirichlet Ensemble Blend
- Merges candidate scores via an inner join across all pipeline predictions on `(user_id, movie_id)`.
- Explores convex weight combinations ($\sum w_m = 1.0$, $w_m \ge 0$) drawn from a Dirichlet prior $\text{Dir}(\alpha)$ across 30 iterations on a 1,000-user validation subsample to maximize Top-10 ranking precision before running inference on the full test split.

---

## Evaluation & Results

### Benchmark Task
- **Validation Set:** Evaluated against `val.csv` (6,037 users), where each user holds exactly **one** target relevant interaction (ground-truth rating $> 3$).
- **Metric Formulation:**

$$\text{Precision@10} = \frac{\text{Total Hits}}{N \times 10}$$

> *Note on scale:* In a single-positive holdout setup, maximum theoretical hit rate per user is $1/10 = 0.10$. A score of 185 hits across 6,037 users with ~3,800 candidate items reflects strong ranking power over random chance ($10 / 3883 \approx 0.0025$).

### Performance Comparison

> *Note:* The reult maybe different because of SVD (Given undefined random_state's value) and Dirichlet distribution

| Model | Hits (out of 6,037) | Precision@10 | Precision@10 (%) | Output CSV |
| :--- | :---: | :---: | :---: | :--- |
| **Global Bias** | 11 | 0.00018 | 0.02% | `recommendations_global.csv` |
| **Latent Factor (SVD)** | 93 | 0.00154 | 0.15% | `recommendations_latent_factor.csv` |
| **Content-Based Filtering** | 95 | 0.00157 | 0.16% | `recommendations_content.csv` |
| **Ensemble (Final Blend)** | **185** | **0.00306** | **0.31%** | `recommendations_ensemble_final.csv` |

```
Precision@10 Hits Comparison (out of 6,037 validation users)
─────────────────────────────────────────────────────────────
Global Bias       [ 11 ] 
Latent Factor SVD [ 93 ] ■■■■■
Content-Based     [ 95 ] ■■■■■
Ensemble Blend    [ 185] ■■■■■■■■■■ (+95% over Content-Based)
```

---

## Repository Structure

```
├── EnsembleRecommendationSystem.ipynb   # End-to-end data prep, modeling & ensemble pipeline
├── train.csv                            # User interaction training split (~988k records)
├── val.csv                              # Validation split (single target item per user)
├── users.dat                            # User metadata (Age, Gender, Occupation, Zip)
├── movies.dat                           # Catalog metadata (MovieID, Title, Genres)
├── recommendations_content.csv          # Content-based Top-10 recommendation rankings
├── recommendations_global.csv           # Global bias Top-10 recommendation rankings
├── recommendations_latent_factor.csv    # SVD Top-10 recommendation rankings
└── recommendations_ensemble_final.csv   # Final weighted ensemble Top-10 recommendations
```

---

## Quickstart

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/CATPJCN/MovieLens_recommendation-system.git
cd <your-ensemble-folder-name>

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Results

Run the notebook sequentially to reproduce preprocessing, model training, Dirichlet weight search, and recommendation exports:

```bash
jupyter notebook EnsembleRecommendationSystem.ipynb
```

---

## Course Context

Built as part of **DES431 — Recommendation Systems**. The objective was to design and evaluate candidate generation and ranking systems capable of scaling to over 1M interactions while balancing collaborative and content features.

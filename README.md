# Hierarchical Bayesian Reanalysis of Buehler et al. (2025)

## Overview

This project reanalyzes the behavioral dataset from:

**Buehler, R., Potocar, L., Mikus, N., & Silani, G. (2025).**  
*Autistic traits relate to reduced reward sensitivity in learning from point-light displays (PLDs).*  
Royal Society Open Science, 12: 241349. https://doi.org/10.1098/rsos.241349

The original study applied maximum likelihood estimation to a static-α Rescorla–Wagner model.  
In this project, we implement a **Hierarchical Bayesian Model (HBM)** that allows for trial-by-trial variation in the learning rate (α) and test whether this dynamic model better captures individual differences in learning behavior—particularly in relation to **Autism Spectrum Quotient (AQ)** scores.

This analysis was conducted as the **final project** for the course *Computational Modeling* (Spring 2025) at **Seoul National University (SNU)**.

---

## Data

The file `data/raw_choices.csv` contains trial-level behavioral data used in this analysis.  
This dataset was retrieved from the Dryad repository associated with the following publication:

> Buehler, R., Potocar, L., Mikus, N., & Silani, G. (2025).  
> *Autistic traits relate to reduced reward sensitivity in learning from point-light displays (PLDs).*  
> Royal Society Open Science, 12: 241349.  
> https://doi.org/10.1098/rsos.241349

📂 Dryad dataset: https://datadryad.org/stash/dataset/doi:10.5061/dryad.4xgxd25k6

This reanalysis uses the behavioral choice data included in the Dryad archive, lightly reformatted for modeling.

> ⚠️ The data are used here **strictly for academic purposes**, and all rights remain with the original authors.

---

## Repository Structure

```bash
├── data/             # data
├── stan/             # Stan model files (HBM, dynamic/static RW)
├── results/          # Posterior estimates
├── scripts/          # R scripts for preprocessing, fitting, and diagnostics
└── README.md
```

---

## Reproducing the Analysis

```r
# Required packages
install.packages(c("tidyverse", "rstan", "loo", "posterior", "cmdstanr"))

# Run the main HBM fitting script
Rscript scripts/fit_HBM.R
```

---

## Key Results

- Dynamic α-HBM yielded improved predictive fit over static model (PSIS-LOO).
- Parameter recovery simulations suggest that the static model may confound reward sensitivity (ρ) and learning speed (α).
- AQ-related effects replicated partially, with reward sensitivity (ρ_win) negatively correlated with AQ scores as in original MLE analysis.

---

## Author
Suyeon Jee  
*Computational Modeling* (Spring 2025)  

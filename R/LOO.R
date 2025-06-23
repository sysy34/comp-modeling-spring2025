# ---------------------------------------------
# LOO.R  – fixed-α vs dynamic-α LOOIC
# ---------------------------------------------
library(rstan)
library(loo)     
setwd("/Users/suyeonjee/Documents/2025-1/modeling/term project/buehler/TPJ/")

# 1) 모델 불러오기 ---------------------------------------------------------------
fit_fix <- readRDS("results/HBM_fix.rds")   # fixed-α
fit_dyn <- readRDS("results/HBM_dyn.rds")   # dynamic-α

# 2) log-lik --------------
loglik_fix <- extract_log_lik(fit_fix, merge_chains = FALSE)  
loglik_dyn <- extract_log_lik(fit_dyn, merge_chains = FALSE)

# 3) LOO 계산 --------------------------------------------------------------------
loo_fix <- loo(loglik_fix)   # Pareto-smoothed importance-sampling LOO
loo_dyn <- loo(loglik_dyn)

# 4) 비교 ------------------------------------------------------------------------
comp <- loo_compare(loo_fix, loo_dyn)
print(comp)                   # ΔLOOIC, weight
# -------------------------------------------------------------------------------

loo_fix_re <- loo(loglik_fix, r_eff = NA, save_psis = TRUE, reloo = TRUE,
                  k_threshold = 0.7, cores = 2)

loo_dyn_re <- loo(loglik_dyn, r_eff = NA, save_psis = TRUE, reloo = TRUE,
                  k_threshold = 0.7, cores = 2)

loo::pareto_k_table(loo_fix_re)   # 모든 k ≤ 0.7 확인
loo::pareto_k_table(loo_dyn_re)


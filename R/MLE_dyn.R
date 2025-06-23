# -----------------------------------------------------------
# MLE_dyn.R  –  per-participant MLE for *dynamic-alpha* RW model
# -----------------------------------------------------------
# 실행:   Rscript MLE_dyn.R   (또는 RStudio Source)
# 출력:   results/MLE_dyn.rds   (리스트: pid → par[ρ, θ, α])
# -----------------------------------------------------------
setwd("/Users/suyeonjee/Documents/2025-1/modeling/term project/buehler/TPJ/")
source("R/model_functions.R")   # negLL_fix, fit_ind_mle
library(tidyverse)
library(glmnet)    # Elastic-Net penalty
library(trust)     # 논문이 쓴 trust-region 방법 (TNC)
library(future.apply)

# --- data -----------------------------------------------------------------
raw <- read_csv("data/raw_choices.csv") %>% filter(!is.na(Reward))

d <- raw %>%
  mutate(Choice  = ParticipantAnswer,
         Outcome = Reward,
         Trial   = TrialNumber) %>%
  arrange(ParticipantID, RunNumber)

# -------- 블록별 MLE 함수 -------------
library(numDeriv)   # → grad(), hessian()

fit_block <- function(df_sub, n_restart = 5,
                      lambda1 = 0.01, lambda2 = 0.01){
  best_val <- Inf; best_par <- NULL; best_conv <- NA_integer_
  
  objfun <- function(par){
    negLL_dyn(par, df_sub, lambda1, lambda2)
  }
  
  lower <- c(-10,-10,  1, -10,-10,  1, -10)
  upper <- c( 10, 10, 50,  10, 10, 50,  10)
  
  for(r in seq_len(n_restart)){
    par0 <- runif(7, lower, upper)
    res  <- optim(par0, objfun, method = "L-BFGS-B",
                  lower = lower, upper = upper,
                  control = list(maxit = 1000))
    
    if(res$value < best_val){
      best_val  <- res$value
      best_par  <- res$par
      best_conv <- res$convergence        # 0 = OK
    }
  }
  
  list(par = best_par,        # numeric(7) 또는 NULL
       conv = best_conv,      # int 또는 NA
       nll = best_val)        # best negative log‑likelihood
}


# -------- 루프 실행 -------------------
plan(multisession, workers = 4)

## (1) 참가자 × BlockType 별 data.frame 리스트
df_list <- split(d, interaction(d$ParticipantID, d$BlockType))

## (2) 병렬 MLE
res_list <- future_lapply(df_list, fit_block, n_restart = 10)

# ── 결과 벡터화 ──────────────────────────────────────────────
est_vec  <- map(res_list, "par")      |> set_names(names(df_list))
conv_vec <- map_int(res_list, "conv") |> set_names(names(df_list))
ll_vec   <- map_dbl(res_list, "nll")  |> set_names(names(df_list))

saveRDS(est_vec,  "results/mle_dyn_estimates.rds")   # 파라미터
saveRDS(conv_vec, "results/mle_dyn_conv.rds")        # convergence
saveRDS(ll_vec,   "results/mle_dyn_loglik.rds")      # -LL

# ── participant × block 테이블 변환 ─────────────────────────
pars_tbl <- imap_dfr(est_vec, ~{
  pid <- str_extract(.y, "^[^.]+")           # "P001.nonsocial" → "P001"
  blk <- str_extract(.y, "(?<=\\.)[^.]+")    # "nonsocial"/"social"
  par <- .x                                  # numeric(7) 또는 NULL
  
  # NULL → NA 대체
  par <- if(length(par) == 7) par else rep(NA_real_, 7)
  
  tibble(
    ParticipantID = pid,
    BlockType     = blk,
    Alpha0_Win    = logistic(par[1]),
    Theta_Win     = par[3],
    Rho_Win       = logistic(par[2]),
    Alpha0_Loss   = logistic(par[4]),
    Theta_Loss    = par[6],
    Rho_Loss      = logistic(par[5]),
    Kappa         = logistic(par[7])
  )
}) |>
  mutate(BlockType = factor(BlockType, levels = c("nonsocial", "social")))

saveRDS(pars_tbl, "results/mle_dyn.rds")

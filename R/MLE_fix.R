# -----------------------------------------------------------
# MLE_fix.R  –  per-participant MLE for *fixed-alpha* RW model
# -----------------------------------------------------------
# 실행:   Rscript fit_mle_fix.R   (또는 RStudio Source)
# 출력:   results/mle_fix_estimates.rds   (리스트: pid → par[ρ, θ, α])
#          results/mle_fix_conv.rds       (named int: pid → convergence)
#          results/mle_fix_loglik.rds     (named dbl: pid → -LL)
#          mle_fix_pars_tbl.rds
# -----------------------------------------------------------
setwd("/Users/suyeonjee/Documents/2025-1/modeling/term project/buehler/TPJ/")
source("R/model_functions.R")   # negLL_fix, fit_ind_mle
library(tidyverse)
library(glmnet)    # Elastic-Net penalty
library(trust)     # TNC
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
  best_val <- Inf
  best_par <- NULL
  best_conv <- NA_integer_
  
  # ── negLL 래퍼 ────────────────────────────────────────────
  objfun <- function(par) {
    negLL_RW(par, df_sub,
                lambda1 = lambda1, lambda2 = lambda2)
  }
  
  # ── 경계: 논문과 동일 (α,ρ = -10~10 → plogis 변환, θ = 1~50) ─
  lower <- c(-10, -10,   1,  -10, -10,   1)
  upper <- c( 10,  10,  50,   10,  10,  50)
  
  # ── random-restart L-BFGS-B ─────────────────────────────
  for(r in seq_len(n_restart)){
    par0 <- runif(6, lower, upper)
    
    res <- optim(par = par0, fn = objfun,
                 method = "L-BFGS-B",
                 lower = lower, upper = upper,
                 control = list(maxit = 1000))
    
    if(res$value < best_val){
      best_val  <- res$value
      best_par  <- res$par
      best_conv <- res$convergence    # 0 → OK
    }
  }
  
  list(par  = best_par,     # numeric(6) 또는 NULL
       conv = best_conv,    # int 또는 NA
       nll  = best_val)     # best negative log-likelihood
}

# -------- 루프 실행 -------------------
plan(multisession, workers = 4)

## (1) 참가자 × BlockType 별 data.frame 리스트 만들기
df_list <- split(d, interaction(d$ParticipantID, d$BlockType))

## (2) 병렬 MLE
res_list <- future_lapply(df_list, fit_block, n_restart = 10)

est_vec  <- map(res_list, "par")      |> set_names(names(df_list))
conv_vec <- map_int(res_list, "conv") |> set_names(names(df_list))
ll_vec   <- map_dbl(res_list, "nll")  |> set_names(names(df_list))

saveRDS(est_vec,  "results/mle_fix_estimates.rds")  # 파라미터
saveRDS(conv_vec, "results/mle_fix_conv.rds")       # convergence
saveRDS(ll_vec,   "results/mle_fix_loglik.rds")     # -LL


pars_tbl <- imap_dfr(est_vec, ~{
  pid <- str_extract(.y, "^[^.]+")            # "S01.social" → "S01"
  blk <- str_extract(.y, "(?<=\\.)[^.]+")     # "S01.social" → "social"
  par <- .x                                   # numeric(6) 또는 NULL
  
  # NULL 이면 NA 채우기
  par <- if(length(par) == 6) par else rep(NA_real_, 6)
  
  tibble(
    ParticipantID = pid,
    BlockType     = blk,
    Alpha_Win  = logistic(par[1]),
    Theta_Win  = par[3],            # 
    Rho_Win    = logistic(par[2]),
    Alpha_Loss = logistic(par[4]),
    Theta_Loss = par[6],            # 
    Rho_Loss   = logistic(par[5])
  )
}) |>
  mutate(BlockType = factor(BlockType, levels = c("nonsocial","social")))

# 3) 파일로 저장───────────────────────────────────────────────
saveRDS(pars_tbl, "results/mle_fix.rds")



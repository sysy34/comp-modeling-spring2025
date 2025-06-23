# -----------------------------------------------------------
# MLE_check.R (sanity check)
# -----------------------------------------------------------
setwd("/Users/suyeonjee/Documents/2025-1/modeling/term project/buehler/TPJ/")
source("R/model_functions.R")
library(tidyverse)

d <- read_csv("data/raw_choices.csv") %>% 
  filter(!is.na(Reward)) %>%           # 
  mutate( Choice  = ParticipantAnswer, # 
          Outcome = Reward )           # 

# # check fixed alpha
# est <- readRDS("results/mle_fix.rds")          # list:  pid.block → num[6]
# conv_vec <- setNames(rep(0, length(est)), names(est))
# ll_wrap   <- negLL_RW
# npar_exp  <- 6 

tol      <- 5

# check dynamic alpha
est <- readRDS("results/mle_dyn.rds")
conv_vec <- setNames(rep(0, length(est)), names(est))
ll_wrap   <- negLL_dyn
npar_exp  <- 7

# sanity check 루프  ────────────────────────────────────
log_vec <- map2_chr(names(est), est, function(pid, par_vec){
  
  df_sub <- d %>% filter(ParticipantID == pid) 
  
  ## 1) 파라미터 상태
  if(length(par_vec) != npar_exp || any(is.na(par_vec)))
    return(paste(pid, "FAIL_NULL"))
  
  ## 2) LL & gradient
  ll_fun <- function(par, df) ll_wrap(par, df, lambda1=.01, lambda2=.01)
  
  chk <- check_solution(par_vec, df_sub,
                        ll_fun   = ll_fun,
                        ll_optim = ll_fun(par_vec, df_sub), tol_grad = tol)
  
  if(chk$pass_ll && chk$pass_grad){
    paste(pid, "OK")
  } else {
    paste(pid, "FAIL",
          "ΔLL",  signif(chk$delta_ll, 3),
          "max∇", signif(chk$max_grad, 3))
  }
})

writeLines(log_vec, "results/sanity_log.txt")


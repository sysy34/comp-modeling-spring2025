# -----------------------------------------------------------
# participant_level_recovery.R
#   – 한 명씩: 가상 데이터 생성 → 두 모형 적합 → κ 회수력 계산
# -----------------------------------------------------------
library(tidyverse)
library(rstan)
options(mc.cores = parallel::detectCores())

### 0) 공통 설정 -------------------------------------------------
proj <- "/Users/suyeonjee/Documents/2025-1/modeling/term project/buehler/TPJ/"
setwd(proj)

model_dyn <- stan_model("stan/HBM_dyn.stan")   # 1명도 HBM 구조 재사용
model_fix <- stan_model("stan/HBM_fix.stan")

# 보상 행렬 (A:0.85 / B:0.15)
rew_prob <- matrix(c(0.85,0.15, 0.15,0.85), nrow = 2, byrow = TRUE)

simulate_trials <- function(par, T_sb){
  Qa <- Qb <- 0.5
  ch <- integer(T_sb); rw <- integer(T_sb)
  for(t in seq_len(T_sb)){
    th <- 1 + 9 * plogis(mean(c(par[2],par[5])))
    pA <- plogis(th*(Qa-Qb)); ch[t] <- rbinom(1,1,1-pA)
    rw[t] <- rbinom(1,1, rew_prob[ch[t]+1,1])
    v <- ifelse(rw[t]==1,1,2)
    alpha <- plogis(par[(v==1)*1+(v==2)*4] + par[7]*(t-1))
    rho   <- par[(v==1)*3+(v==2)*6]
    delta <- rho*rw[t] - ifelse(ch[t]==0,Qa,Qb)
    if(ch[t]==0) Qa <- Qa + alpha*delta else Qb <- Qb + alpha*delta
  }
  list(choice = ch, reward = rw)
}

### 1) 원본 데이터 구조 파악 ------------------------------------
d <- read_csv("data/raw_choices.csv", show_col_types = FALSE) %>% 
  filter(!is.na(Reward)) %>%
  mutate(Choice = ParticipantAnswer, Outcome = Reward) %>%
  arrange(ParticipantID, RunNumber, TrialNumber)

subj_ids <- d |> distinct(ParticipantID) |> pull()
S <- length(subj_ids);  B <- 2L

### 2) posterior draw 중 κ 분산 큰 행 선택 ----------------------
fit_full <- read_rds("results/HBM_dyn.rds")
post_par <- extract(fit_full, pars = "par")$par                # iter × S × B × 7
set.seed(2025)
idx <- which(apply(post_par[,,,7],1,sd) > .10)[1]              # SD(κ)≥0.10
true_par_all <- post_par[idx,,,]                               # S×B×7

### 3) 참가자 루프 --------------------------------------------
result <- tibble(subj = subj_ids,
                 kappa_true_non = NA, kappa_true_soc = NA,
                 kappa_hat_dyn_non = NA, kappa_hat_dyn_soc = NA,
                 rho_hat_fix_non  = NA, rho_hat_fix_soc  = NA)

for(s in seq_len(S)){
  ## 블록별 trial 수
  T_vec <- d |> filter(ParticipantID == subj_ids[s]) |>
    count(BlockType) |> pivot_wider(names_from = BlockType,
                                    values_from = n, values_fill = 0) |>
    select(nonsocial,social) |> as.integer()
  
  ## --- 가상 데이터 생성 --------------------------------------
  ch_all <- rw_all <- integer(sum(T_vec)); pos <- 1
  for(b in 1:B){
    sim <- simulate_trials(true_par_all[s,b,], T_vec[b])
    rng <- pos:(pos+T_vec[b]-1)
    ch_all[rng] <- sim$choice; rw_all[rng] <- sim$reward
    pos <- pos + T_vec[b]
  }
  
  ## Stan 데이터 (하나의 피험자) -------------------------------
  dat_s <- list(S = 1L, B = 2L,
                N = length(ch_all),
                T = matrix(T_vec,1,2),
                choice = ch_all,
                reward = rw_all,
                subj_idx = rep(1L,length(ch_all)),
                blk_idx  = rep(1:2, T_vec))
  
  ## --- 두 모형 적합 (가벼운 세팅) ----------------------------
  fit_d <- sampling(model_dyn, data = dat_s,
                    chains = 2, iter = 1000, warmup = 500,
                    control = list(adapt_delta = 0.995), refresh = 0)
  
  fit_f <- sampling(model_fix, data = dat_s,
                    chains = 2, iter = 1000, warmup = 500,
                    control = list(adapt_delta = 0.995), refresh = 0)
  
  ## --- 파라미터 평균 추출 ------------------------------------
  est_d  <- extract(fit_d,"par")$par |> apply(c(2,3), mean)   # 2×7
  est_f  <- extract(fit_f,"par")$par |> apply(c(2,3), mean)   # 2×6
  
  ## --- 결과 저장 ---------------------------------------------
  result$kappa_true_non[s]     <- true_par_all[s,1,7]
  result$kappa_true_soc[s]     <- true_par_all[s,2,7]
  result$kappa_hat_dyn_non[s]  <- est_d[1,7]
  result$kappa_hat_dyn_soc[s]  <- est_d[2,7]
  result$rho_hat_fix_non[s]    <- est_f[1,3]
  result$rho_hat_fix_soc[s]    <- est_f[2,3]
  cat(sprintf("✓ subj %s done\n", subj_ids[s]))
}

### 4) 참가자 단위 상관계수 ------------------------------
dyn_non  <- cor(result$kappa_true_non, result$kappa_hat_dyn_non)
dyn_soc  <- cor(result$kappa_true_soc, result$kappa_hat_dyn_soc)
sta_non  <- cor(result$kappa_true_non, result$rho_hat_fix_non)
sta_soc  <- cor(result$kappa_true_soc, result$rho_hat_fix_soc)

cat(sprintf(
  "\n[κ recovery – dynamic-α]\n  nonsocial: r = %.3f\n  social   : r = %.3f",
  dyn_non, dyn_soc))
cat(sprintf(
  "\n\n[κ → ρ_win – static-α]\n  nonsocial: r = %.3f\n  social   : r = %.3f\n",
  sta_non, sta_soc))

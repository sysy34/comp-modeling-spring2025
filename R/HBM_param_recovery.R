# -----------------------------------------------------------
# HBM_param_recovery.R  –  data.frame d → parameter recovery
# -----------------------------------------------------------

library(tidyverse)
library(rstan)
options(mc.cores = parallel::detectCores())

# 0) 경로 ----------------------------------------------------
proj <- "/Users/suyeonjee/Documents/2025-1/modeling/term project/buehler/TPJ/"
setwd(proj)

# 1) 원본 데이터 --------------------------------------------
d <- read_csv("data/raw_choices.csv", show_col_types = FALSE) %>%             # ← users line
  filter(!is.na(Reward)) %>%
  mutate(Choice  = ParticipantAnswer,
         Outcome = Reward,
         Trial   = TrialNumber) %>%
  arrange(ParticipantID, RunNumber, Trial)

subj_ids <- d %>% distinct(ParticipantID) %>% pull()
S        <- length(subj_ids)
B        <- 2L                                        # nonsocial=1, social=2

# trial 길이 행렬 T  (S × B) -------------------------------
T_mat <- d %>%
  group_by(ParticipantID, BlockType) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = BlockType, values_from = n, values_fill = 0) %>%
  select(-ParticipantID) %>% 
  as.matrix() %>% 
  `[`( , c("nonsocial","social")) 
stopifnot(nrow(T_mat) == S)                            # sanity‑check

# trial 벡터 및 인덱스 --------------------------------------
choice_vec <- d$Choice
reward_vec <- d$Outcome
subj_idx   <- match(d$ParticipantID, subj_ids)         # 1…S
blk_idx    <- as.integer(d$BlockType == "social") + 1  # 1 or 2
N          <- length(choice_vec)

# 2) Stan 데이터 리스트 (실제 구조) --------------------------
dat_template <- list(
  S        = S,
  B        = B,
  N        = N,
  T        = T_mat,
  choice   = choice_vec,     # 
  reward   = reward_vec,     #  
  subj_idx = subj_idx,
  blk_idx  = blk_idx
)

# 3) ‘진짜’ 파라미터 샘플 -----------------------------------
fit_dyn <- read_rds("results/HBM_dyn.rds")             
par_post <- rstan::extract(fit_dyn, pars = "par")$par  # iter × S × B × 7
set.seed(123)

# κ SD > 0.10 인 행만
big_kappa <- which(apply(par_post[,,,7], 1, sd) > 0.10)

idx       <- big_kappa[1]     # or sample(big_kappa, 1)
true_par  <- par_post[idx, , , ]   #  S × B × 7

# true_par <- par_post[sample(dim(par_post)[1], 1), , , ]   # S × B × 7

# 4) 가상 trial 생성 함수 -----------------------------------
# 보상 확률(0.85/0.15)
rew_prob <- matrix(c(0.85, 0.15,   # choice A (win/loss)
                     0.15, 0.85),  # choice B
                   nrow = 2, byrow = TRUE)

simulate_trials <- function(par, T_sb){
  Qa <- Qb <- 0.5
  choices <- integer(T_sb); rewards <- integer(T_sb)
  
  for(t in seq_len(T_sb)){
    ## 1) 선택 확률 계산
    th <- 1 + 9 * plogis(mean(c(par[2], par[5])))   # 여기서는 θ_w(또는 평균 θ) 하나만 사용
    pA  <- plogis(th * (Qa - Qb))      # 
    choices[t] <- rbinom(1, 1, 1 - pA) # 0=A, 1=B
    
    ## 2) 보상 샘플링 (선택에 따른 0.85 / 0.15 비대칭)
    rewards[t] <- rbinom(1, 1, rew_prob[choices[t] + 1, 1])  # 1열 = win 확률
    ## 3) 이번 trial의 valence 인덱스
    v <- ifelse(rewards[t] == 1, 1, 2)  # 1 = win, 2 = loss
    
    ## 4) 학습률·가치 가중치 선택 후 Q 업데이트
    alpha <- plogis(par[(v==1)*1 + (v==2)*4] + par[7] * (t - 1))
    rho   <- par[(v==1)*3 + (v==2)*6]
    
    delta <- rho * rewards[t] - ifelse(choices[t] == 0, Qa, Qb)
    if (choices[t] == 0) {
      Qa <- Qa + alpha * delta
    } else {
      Qb <- Qb + alpha * delta
    }
  }
  list(choice = choices, reward = rewards)
}

# 5) 전체 가상 데이터 ---------------------------------------
sim_choice <- integer(N);  sim_reward <- integer(N)
pos <- 1
for (s in seq_len(S)) for (b in seq_len(B)) {
  out <- simulate_trials(true_par[s,b,], T_mat[s,b])
  rng <- pos:(pos + T_mat[s,b] - 1)
  sim_choice[rng] <- out$choice
  sim_reward[rng] <- out$reward
  pos <- pos + T_mat[s,b]
}

dat_sim <- dat_template
dat_sim$choice <- sim_choice
dat_sim$reward <- sim_reward
saveRDS(dat_sim, "results/sim_data_dyn.rds")

# 6) 모델 컴파일 & 적합 ------------------------------------
model_dyn <- stan_model("stan/HBM_dyn.stan")
model_fix <- stan_model("stan/HBM_fix.stan")

fit_dyn_sim <- sampling(model_dyn, data = dat_sim,
                        chains = 2, iter = 1500, warmup = 750,
                        control = list(adapt_delta = 0.995))
saveRDS(fit_dyn_sim, "results/recovery_fit_dyn.rds")

fit_fix_sim <- sampling(model_fix, data = dat_sim,
                        chains = 2, iter = 1500, warmup = 750,
                        control = list(adapt_delta = 0.995))
saveRDS(fit_fix_sim, "results/recovery_fit_fix.rds")

# 7) 회수력 계산 -------------------------------------------
est_dyn <- apply(rstan::extract(fit_dyn_sim, pars = "par")$par, c(2,3,4), mean)
est_fix <- apply(rstan::extract(fit_fix_sim, pars = "par")$par, c(2,3,4), mean)

true_kappa <- as.vector(true_par[,,7])
est_kappa  <- as.vector(est_dyn[,,7])
est_rho_w  <- as.vector(est_fix[,,3])

cat(sprintf("\ndynamic‑α  : corr(true κ , est κ)     = %.3f",
            cor(true_kappa, est_kappa)))
cat(sprintf("\nstatic‑α   : corr(true κ , est ρ_win) = %.3f\n",
            cor(true_kappa, est_rho_w)))



## 1. true / estimated 행렬은 이미 S × B × 7 형태
##    → 1열 = nonsocial, 2열 = social

# true kappa
true_kappa_mat <- true_par[,,7]      # S × 2

# dynamic-α 추정 κ̂
est_kappa_mat  <- est_dyn[,,7]

# static-α 추정 ρ̂_win
est_rho_mat    <- est_fix[,,3]

## 2. 블록별 상관계수 계산
cor_non_dyn  <- cor(true_kappa_mat[,1], est_kappa_mat[,1])   # nonsocial
cor_soc_dyn  <- cor(true_kappa_mat[,2], est_kappa_mat[,2])   # social

cor_non_sta  <- cor(true_kappa_mat[,1], est_rho_mat[,1])     # nonsocial
cor_soc_sta  <- cor(true_kappa_mat[,2], est_rho_mat[,2])     # social

## 3. 출력
cat(sprintf(
  "\n[κ recovery – dynamic-α]\n  nonsocial: r = %.3f\n  social   : r = %.3f",
  cor_non_dyn, cor_soc_dyn))

cat(sprintf(
  "\n\n[κ vs ρ_win – static-α]\n  nonsocial: r = %.3f\n  social   : r = %.3f\n",
  cor_non_sta, cor_soc_sta))

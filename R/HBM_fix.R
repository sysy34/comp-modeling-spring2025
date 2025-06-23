# HBM_fix.R – Hierarchical Bayesian RW (no‑κ)
# ----------------------------------------------------
# 목적  : Stan 모델 파일 작성 + 컴파일 + 샘플링
# 실행  : Rscript HBM_fix.R
# 요구  : tidyverse, rstan
# ----------------------------------------------------

library(tidyverse)
library(rstan)
options(mc.cores = 2)

## 1) path ---------------------------------------------------
proj <- "/Users/suyeonjee/Documents/2025-1/modeling/term project/buehler/TPJ/"
setwd(proj)
dir.create("stan",    showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

## 2) Stan code ---------------------------------------------
stan_code <- '
data {
  int<lower=1>  S;
  int<lower=1>  B;
  int<lower=1>  N;              // trial 총개수
  int<lower=1>  T[S,B];         // subj × block
  int<lower=0,upper=1> choice[N];
  int<lower=0,upper=1> reward[N];
  int<lower=1,upper=S> subj_idx[N];
  int<lower=1,upper=B> blk_idx[N];
}

parameters {
  // hyper‑priors (κ 제거 → 6개)
  vector[6]   mu;
  vector<lower=0>[6] sigma;

  // subject‑level raw
  array[S,B,6] real par_raw;    // 1:α0_w 2:θ_w 3:ρ_w 4:α0_l 5:θ_l 6:ρ_l
}

transformed parameters {
  array[S,B,6] real par;
  for (s in 1:S)
    for (b in 1:B)
      for (k in 1:6) {
        real tmp = mu[k] + sigma[k] * par_raw[s,b,k];
        par[s,b,k] =
            (k==2 || k==5) ? 1 + 49 * inv_logit(tmp)   // θ 1‑50
          : (k==3 || k==6) ?       inv_logit(tmp)      // ρ 0‑1
          :                         tmp;               // α0_w, α0_l (logit 공간)
      }
}

model {
  // priors
  to_vector( to_array_1d(par_raw) ) ~ normal(0,1);
  mu     ~ normal(0,2);
  sigma  ~ cauchy(0,2);

  int pos = 1;
  for (s in 1:S)
    for (b in 1:B) {
      real Qa = 0.5;
      real Qb = 0.5;

      for (t in 1:T[s,b]) {
        int win  = reward[pos];
        int v    = win ? 1 : 2;                       // 1=w, 2=l

        real alpha = inv_logit( par[s,b,(v==1)?1:4] );// κ 제거
        real rho   = par[s,b,(v==1)?3:6];
        real th    = par[s,b,(v==1)?2:5];

        real pA = inv_logit( th * (Qa - Qb) );
        choice[pos] ~ bernoulli(1 - pA);

        real delta = rho * reward[pos] - (choice[pos]==0 ? Qa : Qb);
        if (choice[pos]==0) Qa += alpha * delta; else Qb += alpha * delta;

        pos += 1;
      }
    }
}

generated quantities {
  vector[N] log_lik;
  int pos = 1;
  for (s in 1:S)
    for (b in 1:B) {
      real Qa = 0.5;
      real Qb = 0.5;
      vector[6] p = to_vector(par[s,b]);

      for (t in 1:T[s,b]) {
        int win  = reward[pos];
        int v    = win ? 1 : 2;

        real alpha = inv_logit( p[(v==1)?1:4] );      // κ 제거
        real rho   = p[(v==1)?3:6];
        real th    = p[(v==1)?2:5];
        real pA    = inv_logit( th * (Qa - Qb) );

        log_lik[pos] = bernoulli_lpmf(choice[pos] | 1 - pA);

        real delta = rho * reward[pos] - (choice[pos]==0 ? Qa : Qb);
        if (choice[pos]==0) Qa += alpha * delta; else Qb += alpha * delta;

        pos += 1;
      }
    }
}
'
writeLines(stan_code, "stan/HBM_fix.stan")

## 3) data preprocessing ------------------------------------------
raw <- read_csv("data/raw_choices.csv", show_col_types = FALSE) %>%
  filter(!is.na(Reward)) %>%
  mutate(Choice = ParticipantAnswer,
         Outcome = Reward) %>%
  arrange(ParticipantID, RunNumber, TrialNumber)

subj_ids <- raw %>% distinct(ParticipantID) %>% pull()

choice_vec <- raw$Choice
reward_vec <- raw$Outcome
subj_idx   <- match(raw$ParticipantID, subj_ids)
blk_idx    <- as.integer(raw$BlockType == "social") + 1  # 1 or 2

T_mat <- raw %>%
  group_by(ParticipantID, BlockType) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = BlockType,
              values_from = n,
              values_fill = 0) %>%
  select(-ParticipantID) %>% 
  as.matrix() %>% 
  `[`( , c("nonsocial","social"))

stopifnot(sum(T_mat) == length(choice_vec))

dat <- list(
  S        = length(subj_ids),
  B        = 2L,
  N        = length(choice_vec),
  T        = T_mat,
  choice   = choice_vec,
  reward   = reward_vec,
  subj_idx = subj_idx,
  blk_idx  = blk_idx
)

## 4) model complie & sampling -------------------------------
sm_noK <- stan_model("stan/HBM_fix.stan")
fit_noK <- sampling(sm_noK, data = dat,
                    chains = 4, iter = 2000, warmup = 1000,
                    control = list(adapt_delta = 0.995, max_treedepth = 15),
                    refresh = 500)

# sm_fix <- stan_model("stan/HBM_fix.stan")
# fit_test <- sampling(sm_fix, data = dat,
#                      chains = 1, iter = 1500, warmup = 750,
#                      control = list(adapt_delta = 0.995,   
#                                     max_treedepth = 15),
#                      refresh = 300)
# sum(get_sampler_params(fit_test, FALSE)[[1]][,"divergent__"])

saveRDS(fit_noK, "results/HBM_fix.rds")

print(fit_noK, pars = c("mu","sigma"), probs = c(.1,.5,.9), digits = 2)



# ── traceplot ────────────────────────────────────────────────────────────────
library(bayesplot)
library(rstan)

color_scheme_set("orange")

pars_main <- c("mu[1]", "mu[4]",   # α0_w, α0_l
               "mu[3]", "mu[6]",   # ρ_w, ρ_l
               "mu[2]", "mu[5]")   # θ_w, θ_l

posterior_arr <- rstan::extract(fit_noK, pars = pars_main, permuted = FALSE)

bayesplot::mcmc_trace(
  posterior_arr,
  pars = pars_main,
  facet_args = list(ncol = 3, strip.position = "left")
)

## 진단 지표 --------------------------------------------------
sum_d     <- summary(fit_noK)$summary
rhat_bad  <- which(sum_d[,"Rhat"] > 1.01)
ess_bad   <- which(sum_d[,"n_eff"] < 400)

length(rhat_bad)   # Rhat 경고 개수
length(ess_bad)    # ESS 경고 개수

check       <- get_sampler_params(fit_noK, inc_warmup = FALSE)
div_total   <- sum(sapply(check, function(x) sum(x[,"divergent__"])))
print(div_total)   # divergent 총합

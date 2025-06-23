# -----------------------------------------------------------
# HBM_dyn.R – Hierarchical Bayesian dynamic-α
# -----------------------------------------------------------

# 0) packages --------------------------------------------------------------------
library(tidyverse)
library(rstan)
options(mc.cores = 2)

# 1) path -----------------------------------------------------------------
proj <- "/Users/suyeonjee/Documents/2025-1/modeling/term project/buehler/TPJ/"
setwd(proj)
if(!dir.exists("stan"))       dir.create("stan")
if(!dir.exists("results"))    dir.create("results")

# 2) Stan -----------------------------------------------------------
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
  vector[7]   mu;                    // ← 반드시 7
  vector<lower=0>[7] sigma;          // ← 반드시 7
  array[S,B,7] real par_raw;         // ← 3-차원, theta_raw 없어야 함
}

transformed parameters {
  // 결과도 3-D 실수 배열
  array[S,B,7] real par;

  for (s in 1:S)
    for (b in 1:B)
      for (k in 1:7) {
        real tmp = mu[k] + sigma[k] * par_raw[s,b,k];
        par[s,b,k] =
          (k==2 || k==5) ? 1 + 49 * inv_logit(tmp)   // θ 1-50
        : (k==3 || k==6) ?       inv_logit(tmp)      // ρ 0-1
        :                             tmp;           // α0_w, α0_l, κ (logit space)
      }
}

model {
  // priors
  to_vector( to_array_1d(par_raw) ) ~ normal(0, 1); 
  mu    ~ normal(0,2);
  sigma[7] ~ normal(0, 1);
  sigma[1:6] ~ cauchy(0, 2); 

  int pos = 1;
  for (s in 1:S)
    for (b in 1:B) {
      real Qa = 0.5;
      real Qb = 0.5;
  
      // 읽을 때는 인덱스로 바로
      for (t in 1:T[s,b]) {
        int win  = reward[pos];
        int v    = win ? 1 : 2;              // 1=w, 2=l
  
        real alpha = inv_logit(
                       par[s,b,(v==1)?1:4]   // α0_w / α0_l
                       + par[s,b,7] * (t-1)  // κ
                     );
  
        real rho = par[s,b,(v==1)?3:6];
        real th  = par[s,b,(v==1)?2:5];
        real pA = inv_logit(th * (Qa - Qb));   // softmax
        choice[pos] ~ bernoulli(1 - pA);   
        real delta = rho * reward[pos] - (choice[pos]==0 ? Qa : Qb);
        if (choice[pos]==0) Qa += alpha*delta; else Qb += alpha*delta;

        pos += 1;
      }
    }
}

generated quantities {
  // log-lik vector for LOO/WAIC
  vector[N] log_lik;
  int pos = 1;
  for (s in 1:S)
    for (b in 1:B) {
      real Qa = 0.5;
      real Qb = 0.5;
      vector[7] p = to_vector(par[s,b]); 
      for (t in 1:T[s,b]) {
        int win  = reward[pos];
        int v    = win ? 1 : 2;
        real alpha = inv_logit( p[(v==1)?1:4] + p[7]*(t-1) );
        real rho   = p[(v==1)?3:6];
        real th    = p[(v==1)?2:5];
        real pA    = inv_logit(th * (Qa - Qb));

        log_lik[pos] = bernoulli_lpmf(choice[pos] | 1 - pA);

        real delta = rho * reward[pos] - (choice[pos]==0 ? Qa : Qb);
        if (choice[pos]==0) Qa += alpha*delta; else Qb += alpha*delta;

        pos += 1;
      }
    }
}
'

stan_path <- "stan/HBM_dyn.stan"
writeLines(stan_code, stan_path)

# data preprocessing --------------------------------------------------------------
raw <- read_csv("data/raw_choices.csv") %>%
  filter(!is.na(Reward)) %>%
  mutate(Choice = ParticipantAnswer,
         Outcome = Reward) %>%
  arrange(ParticipantID, RunNumber, TrialNumber)

subj_ids <- raw %>% distinct(ParticipantID) %>% pull()

choice_vec <- raw$Choice
reward_vec <- raw$Outcome
subj_idx   <- match(raw$ParticipantID, subj_ids)

blk_idx <- as.integer(raw$BlockType == "social") + 1  # 1 또는 2

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

N <- length(choice_vec)   # trial 

dat <- list(
  S        = length(subj_ids),
  B        = 2L,
  N        = N,           
  T        = T_mat,
  choice   = choice_vec,
  reward   = reward_vec,
  subj_idx = subj_idx,
  blk_idx  = blk_idx
)


# 5) model complie & sampling -------------------------------------------------------
model_path <- "stan/HBM_dyn.stan"
sm_dyn    <- rstan::stan_model(file = model_path) 

fit_dyn <- sampling(sm_dyn, data = dat,
                     chains = 4, iter = 4000,
                     warmup = 2000, cores = 2,
                     control = list(adapt_delta = 0.99, max_treedepth = 15),
                    refresh = 500)

saveRDS(fit_dyn, "results/HBM_dyn.rds")

print(fit_dyn, pars = c("mu","sigma"),
      probs = c(.05,.5,.95), digits = 2, max_pars = 14)


# ── traceplot ────────────────────────────────────────────────────────────────
library(bayesplot)
color_scheme_set("orange")

pars_main <- c("mu[1]","mu[4]",   # α_w, α_l
               "mu[3]","mu[6]",   # ρ_w, ρ_l
               "mu[2]","mu[5]",   # θ_w, θ_l
               "mu[7]")           # κ (원하면 주석 처리)

bayesplot::mcmc_trace(
  fit_dyn,
  pars       = pars_main,
  facet_args = list(ncol = 3, strip.position = "left")
)

# ── 진단 지표 ────────────────────────────────────────────────────────────────
sum_d    <- summary(fit_dyn)$summary
rhat_bad <- which(sum_d[,"Rhat"]  > 1.01 & !is.na(sum_d[,"Rhat"]))
ess_bad  <- which(sum_d[,"n_eff"] < 400  & !is.na(sum_d[,"n_eff"]))

cat("# of parameters with Rhat > 1.01 :", length(rhat_bad), "\n")
cat("# of parameters with n_eff < 400 :", length(ess_bad),  "\n")

# divergent / energy 
sampler <- get_sampler_params(fit_dyn, inc_warmup = FALSE)
div_tot <- sum(vapply(sampler, function(x) sum(x[,"divergent__"]), numeric(1)))
cat("Total divergent transitions :", div_tot, "\n")

# ──────────────────────────────────────────────────────────────────
# model_functions.R  —  Buehler et al. (2025) 고정-α RW MLE
#   ● logistic 변환: α, ρ
#   ● θ 변환: θ = 1 + 49·logistic(xθ)  (→ 1‒50)
#   ● Q 초기값 = 0.5
#   ● Outcome: 1(win), 0(loss)   ← 원본과 동일
#   ● δ  = ρ·outcome  −  Qc      (loss 시 outcome=0 → δ = −Qc)
#   ● Elastic-Net (λ = 0.01) - 원문 파라미터 그대로
#   ● 최적화: trust::trust(TNC)  /   10× random-restart
# ──────────────────────────────────────────────────────────────────

logistic      <- function(x) 1 / (1 + exp(-x))
scale_theta   <- function(x) 1 + 49 * logistic(x)   # 1–50
softmax       <- function(delta, theta) 1 / (1 + exp(-theta * delta))

is_feasible <- function(pu, obj_fn){
  out <- tryCatch(obj_fn(pu), error = function(e) Inf)
  is.finite(out) && !is.nan(out) && out < 1e12
}

# negLL_block <- function(par, df_sub, lambda1 = .01, lambda2 = .01) {
#   
#   # 1) 파라미터 변환 (원본 그대로)
#   alpha_w <- logistic(par[1])        
#   theta_w <- scale_theta(par[2])     
#   rho_w   <- logistic(par[3])        
#   
#   alpha_l <- logistic(par[4])        
#   theta_l <- scale_theta(par[5])     
#   rho_l   <- logistic(par[6])        
#   
#   # 2) 초기 Q (원본: 0.5)
#   Qa <- Qb <- 0.5
#   ll <- 0
#   
#   ## 3) trial 루프 -------------------------------------------------
#   for(i in seq_len(nrow(df_sub))){
#     ## --- softmax -----------------------------------------------
#     win     <- df_sub$Outcome[i] == 1
#     theta <- if (win) theta_w else theta_l 
#     pA    <- 1 / (1 + exp(-theta * (Qa - Qb)))
#     
#     ch <- df_sub$Choice[i]
#     ll <- ll + log( ifelse(ch == 0, pA, 1 - pA) + 1e-12 )
#     
#     ## --- RPE & Q 업데이트 ----------------------------------------
# 
#     rho     <- if (win) rho_w else rho_l
#     outcome <- if (win)  1 else 0
#     Qc    <- if(ch==0) Qa else Qb
#     delta <- rho * outcome - Qc
#     alpha   <- if (win) alpha_w else alpha_l
#     
#     if(ch==0) Qa <- Qa + alpha*delta else Qb <- Qb + alpha*delta
#   }
#   
#   pen <- lambda1 * sum(abs(par)) + lambda2 * sum(par^2)   # elastic‑net
#   -(ll - pen)
# }

# ---------- negative log‑likelihood -------------------------
negLL_RW <- function(par_raw, df_sub,
                           lambda1 = 0.01, lambda2 = 0.01)
{
  ## 1) transform
  alpha_w <- logistic(par_raw[1]);  rho_w <- logistic(par_raw[2]);  theta_w <- par_raw[3]
  alpha_l <- logistic(par_raw[4]);  rho_l <- logistic(par_raw[5]);  theta_l <- par_raw[6]
  
  ## 2) initialise
  Qa <- Qb <- 0.5
  ll <- 0
  
  ## 3) trial loop
  for(i in seq_len(nrow(df_sub))){
    win   <- df_sub$Outcome[i] == 1
    theta <- if(win) theta_w else theta_l
    pA    <- 1 / (1 + exp(-theta * (Qa - Qb)))
    
    ch <- df_sub$Choice[i]
    ll <- ll + log(ifelse(ch == 0, pA, 1 - pA) + 1e-12)
    
    rho   <- if(win) rho_w else rho_l
    out   <- if(win) 1 else 0
    Qc    <- if(ch == 0) Qa else Qb
    delta <- rho * out - Qc
    alpha <- if(win) alpha_w else alpha_l
    
    if(ch == 0) Qa <- Qa + alpha * delta else Qb <- Qb + alpha * delta
  }
  
  ## 4) elastic‑net (논문 스케일)
  pen <- lambda1 * sum(abs(par_raw)) + lambda2 * sum(par_raw^2)
  
  -(ll - pen)          # return penalised −log‑likelihood
}



# ──────────────────────────────────────────────────────────────────
#dynamic alpha 
# ──────────────────────────────────────────────────────────────────
# negLL_block_dyn <- function(par, df_sub, lambda1 = .01, lambda2 = .01){
#   
#   a0_w <- logistic(par[1]); th_w <- scale_theta(par[2]); rho_w <- logistic(par[3])
#   a0_l <- logistic(par[4]); th_l <- scale_theta(par[5]); rho_l <- logistic(par[6])
#   k    <- exp(par[7])          # shared decay
#   
#   Qa <- Qb <- 0.5
#   ll <- 0
#   
#   for(t in seq_len(nrow(df_sub))){
#     
#     win   <- df_sub$Outcome[t] == 1
#     alpha_t0 <- if(win) a0_w else a0_l
#     alpha_t  <- alpha_t0 * exp(-k * (t - 1))
#     
#     theta <- if(win) th_w else th_l
#     rho   <- if(win) rho_w else rho_l
#     pA    <- 1 / (1 + exp(-theta * (Qa - Qb)))
#     
#     ch <- df_sub$Choice[t]                      # ▲
#     ll <- ll + log(ifelse(ch == 0, pA, 1 - pA) + 1e-12)
#     
#     reward <- df_sub$Outcome[t]
#     Qc     <- if(ch == 0) Qa else Qb
#     delta  <- rho * reward - Qc
#     
#     if(ch == 0) Qa <- Qa + alpha_t * delta else Qb <- Qb + alpha_t * delta
#   }
#   
#   pen <- lambda1 * sum(abs(par)) + lambda2 * sum(par^2)
#   -(ll - pen)
# }

# ---------- negative log‑likelihood (dynamic‑α) -------------

## dynamic‑alpha (win/lose 별 baseline α, ρ, θ + κappa) ----------
negLL_dyn <- function(par_raw, df,
                            lambda1 = 0.01, lambda2 = 0.01)
{
  ## 0) unpack & transform ------------------------------------------
  alpha_w0 <- logistic(par_raw[1])          # baseline α_win (t = 0)
  rho_w    <- logistic(par_raw[2])          # ρ_win
  theta_w  <- par_raw[3]                    # θ_win  (1–50)
  
  alpha_l0 <- logistic(par_raw[4])          # baseline α_loss
  rho_l    <- logistic(par_raw[5])          # ρ_loss
  theta_l  <- par_raw[6]                    # θ_loss (1–50)
  
  kappa    <- logistic(par_raw[7])          # 0–1  mixing weight
  
  ## 1) initialise ---------------------------------------------------
  Qa <- Qb <- 0.5
  alpha_w <- alpha_w0
  alpha_l <- alpha_l0
  delta_prev_w <- delta_prev_l <- 0
  ll <- 0
  
  ## 2) trial loop ---------------------------------------------------
  for(i in seq_len(nrow(df))){
    win   <- df$Outcome[i] == 1
    theta <- if(win) theta_w else theta_l
    pA    <- 1 / (1 + exp(-theta * (Qa - Qb)))
    
    ch <- df$Choice[i]
    ll <- ll + log( ifelse(ch == 0, pA, 1 - pA) + 1e-12 )
    
    ## ----- RPE & Q update -----------------------------------------
    rho     <- if(win) rho_w else rho_l
    outcome <- if(win) 1 else 0
    Qc      <- if(ch == 0) Qa else Qb
    delta   <- rho * outcome - Qc
    alpha   <- if(win) alpha_w else alpha_l
    
    if(ch == 0) Qa <- Qa + alpha * delta else Qb <- Qb + alpha * delta
    
    ## ----- dynamic α update (domain‑specific) ---------------------
    if(win){
      alpha_w     <- kappa * abs(delta_prev_w) + (1 - kappa) * alpha_w
      delta_prev_w<- delta
    } else {
      alpha_l     <- kappa * abs(delta_prev_l) + (1 - kappa) * alpha_l
      delta_prev_l<- delta
    }
    
    ## clamp α to (1e‑6, 1‑1e‑6) for numerical stability
    alpha_w <- pmin(pmax(alpha_w, 1e-6), 1 - 1e-6)
    alpha_l <- pmin(pmax(alpha_l, 1e-6), 1 - 1e-6)
  }
  
  ## 3) elastic‑net (raw‑scale, 논문 방식) ---------------------------
  pen <- lambda1 * sum(abs(par_raw)) + lambda2 * sum(par_raw^2)
  
  -(ll - pen)
}



# ── helper: check_solution ──────────────────────────────────────────
#  • ll_fun      :  (par, df) → -log-likelihood
#  • ll_optim    :  optim() 이 리턴한 best_negLL (optional; 차이 검증용)
#  • tol_ll      :  ΔLL 허용 오차
#  • tol_grad    :  ∥gradient∥_∞ 허용 오차
library(numDeriv)   # grad() 사용

check_solution <- function(par, df, ll_fun,
                           ll_optim = NULL,
                           tol_ll  = 1e-6,
                           tol_grad = 1e-3){
  
  if(length(par) != 6 || any(!is.finite(par)))
    return(list(pass_ll = FALSE, pass_grad = FALSE,
                delta_ll = Inf, max_grad = Inf))
  
  # (1) LL 재계산
  ll_recalc <- ll_fun(par, df)
  delta_ll  <- if(!is.null(ll_optim)) ll_recalc - ll_optim else 0
  pass_ll   <- abs(delta_ll) < tol_ll
  
  # (2) gradient 계산  ※ df= 명시!
  g <- numDeriv::grad(ll_fun, par, df = df)
  max_grad  <- max(abs(g))
  pass_grad <- max_grad < tol_grad
  
  list(pass_ll   = pass_ll,
       pass_grad = pass_grad,
       delta_ll  = delta_ll,
       max_grad  = max_grad)
}
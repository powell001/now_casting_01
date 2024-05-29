library(dlm)
library(tidyverse)

y = read_csv('data/a0_combinedQuarterly.csv')
y = ts(y['gdp_total'])
print(class(y))
print(y)

t_max <- length(y)

# Function performing Kalman filtering for one time point
Kalman_filtering <- function(m_t_minus_1, C_t_minus_1, t){
  # One-step-ahead predictive distribution
  a_t <- G_t %*% m_t_minus_1
  R_t <- G_t %*% C_t_minus_1 %*% t(G_t) + W_t

  # One-step-ahead predictive likelihood
  f_t <- F_t %*% a_t
  Q_t <- F_t %*% R_t %*% t(F_t) + V_t

  # Kalman gain
  K_t <- R_t %*% t(F_t) %*% solve(Q_t)

  # State update
  m_t <- a_t + K_t %*% (y[t] - f_t)
  C_t <- (diag(nrow(R_t)) - K_t %*% F_t) %*% R_t

  # Return the mean and variance of the filtering distribution (and also one-step-ahead predictive distribution)
  return(list(m = m_t, C = C_t,
              a = a_t, R = R_t))
}

# # Set parameters of the linear Gaussian state space (all 1 x 1 matrices)
G_t <- matrix(1, ncol = 1, nrow = 1)
W_t <- matrix(exp(7.29), ncol = 1, nrow = 1)
F_t <- matrix(1, ncol = 1, nrow = 1)
V_t <- matrix(exp(9.62), ncol = 1, nrow = 1)
m0  <- matrix(0, ncol = 1, nrow = 1)
C0  <- matrix(1e+7, ncol = 1, nrow = 1)

# Calculate the mean and variance of the filtering distribution (and also one-step-ahead predictive distribution)

# Allocate memory for state (mean and covariance)
m <- rep(NA_real_, t_max); C <- rep(NA_real_, t_max)
a <- rep(NA_real_, t_max); R <- rep(NA_real_, t_max)

# Time point: t = 1
KF <- Kalman_filtering(m0, C0, t = 1)
m[1] <- KF$m; C[1] <- KF$C
a[1] <- KF$a; R[1] <- KF$R

# Time point: t = 2 to t_max
for (t in 2:t_max){
  KF <- Kalman_filtering(m[t-1], C[t-1], t = t)
  m[t] <- KF$m; C[t] <- KF$C
  a[t] <- KF$a; R[t] <- KF$R
}

# Ignore the display of following codes

# Find 2.5% and 97.5% values for 95% intervals of the filtering distribution
m_sdev <- sqrt(C)
m_quant <- list(m + qnorm(0.025, sd = m_sdev), m + qnorm(0.975, sd = m_sdev))

# Plot results
ts.plot(cbind(y, m, do.call("cbind", m_quant)),
        col = c("lightgray", "black", "black", "black"),
        lty = c("solid", "solid", "dashed", "dashed"))

# Legend
legend(legend = c("Observations", "Mean (filtering distribution)", "95% intervals (filtering distribution)"),
       lty = c("solid", "solid", "dashed"),
       col = c("lightgray", "black", "black"),
       x = "topright", cex = 0.6)
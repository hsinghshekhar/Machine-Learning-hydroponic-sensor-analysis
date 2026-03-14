"""
============================================================
 Shekhar Agriculture Laboratory
 Hydroponic Farming Research & Data Science Division
 ML Step-by-Step Applied on N=10 Sensor Dataset
 
 Date: March 14, 2026
 Report ID: SAL-ML-2026-N10
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Dark Theme Setup ─────────────────────────────────────────
plt.rcParams['figure.facecolor'] = '#0b1426'
plt.rcParams['axes.facecolor']   = '#0d1e36'
plt.rcParams['text.color']       = '#e2e8f0'
plt.rcParams['axes.labelcolor']  = '#7ea0c8'
plt.rcParams['xtick.color']      = '#7ea0c8'
plt.rcParams['ytick.color']      = '#7ea0c8'
plt.rcParams['grid.color']       = '#1a3050'

# ════════════════════════════════════════════════════════════
# SECTION 0: Dataset D (N=10)
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 0: Dataset D — N=10 Hydroponic Sensor Samples")
print("=" * 60)

ec    = np.array([1.621, 1.535, 1.668, 1.689, 1.495,
                  1.548, 1.656, 1.632, 1.661, 1.610])
ph    = np.array([6.50, 6.48, 6.49, 6.52, 6.50,
                  6.50, 6.51, 6.48, 6.51, 6.50])
hum   = np.array([59.5, 59.5, 61.1, 61.1, 59.4,
                  62.7, 61.5, 60.6, 63.6, 63.2])
temp  = np.array([26.31, 26.16, 26.46, 25.57, 25.87,
                  25.81, 25.41, 26.16, 25.81, 26.05])
light = np.array([709, 738, 722, 797, 775,
                  721, 727, 760, 721, 797])

# Labels
y_bin = (ec > 1.63).astype(int)   # Binary: High Growth vs Normal
y_k   = np.where(ec < 1.55, 0,
         np.where(ec < 1.65, 1, 2))  # 3-class

# Regression target
np.random.seed(42)
y_reg = 2*ec**2 - 4*ec + 2 + np.random.normal(0, 0.03, 10)

print(f"{'#':<4} {'EC':<8} {'pH':<6} {'Hum%':<7} {'Temp':<7} {'Light':<7} {'y_bin':<7} {'y_k'}")
for i in range(10):
    print(f"{i+1:<4} {ec[i]:<8.3f} {ph[i]:<6.2f} {hum[i]:<7.1f} {temp[i]:<7.2f} {light[i]:<7} {y_bin[i]:<7} {['Low','Med','High'][y_k[i]]}")


# ════════════════════════════════════════════════════════════
# SECTION 1: Binary Linear Classifier (Logistic Regression)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 1: Binary Linear Classifier")
print("=" * 60)

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def bce_loss(th0, th1, X, Y):
    p = sigmoid(th0 + th1 * X)
    return -np.mean(Y * np.log(p + 1e-15) + (1 - Y) * np.log(1 - p + 1e-15))

# Gradient Descent
th0, th1 = 0.0, 0.0
lr = 0.05
losses = []

for epoch in range(5000):
    z   = th0 + th1 * ec
    p   = sigmoid(z)
    err = p - y_bin
    g0  = np.mean(err)
    g1  = np.mean(err * ec)
    th0 -= lr * g0
    th1 -= lr * g1
    losses.append(bce_loss(th0, th1, ec, y_bin))

print(f"theta0 = {th0:.6f}")
print(f"theta1 = {th1:.6f}")
print(f"Decision Boundary: EC = {-th0/th1:.6f} ms/cm")
print(f"ERM Loss = {losses[-1]:.6f}")

print("\nPer-Sample Predictions:")
for i in range(10):
    z_i = th0 + th1 * ec[i]
    p_i = sigmoid(z_i)
    y_hat = int(p_i >= 0.5)
    ok = "OK" if y_hat == y_bin[i] else "WRONG"
    print(f"  #{i+1}: EC={ec[i]:.3f} z={z_i:.4f} p={p_i:.4f} y_hat={y_hat} y={y_bin[i]} {ok}")


# ════════════════════════════════════════════════════════════
# SECTION 2: K-Class Softmax (K=3)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: Softmax K=3 Classification")
print("=" * 60)

def softmax3(ec_val):
    v = np.array([-5.0 * ec_val, 3.0 * ec_val, 9.0 * ec_val])
    v_exp = np.exp(v - v.max())
    return v_exp / v_exp.sum()

print(f"\n{'#':<4} {'EC':<8} {'P(Low)':<10} {'P(Med)':<10} {'P(High)':<10} {'Pred':<8} {'True'}")
for i in range(10):
    probs = softmax3(ec[i])
    pred  = ['Low', 'Med', 'High'][np.argmax(probs)]
    true  = ['Low', 'Med', 'High'][y_k[i]]
    ok    = "OK" if np.argmax(probs) == y_k[i] else "WRONG"
    print(f"{i+1:<4} {ec[i]:<8.3f} {probs[0]:<10.4f} {probs[1]:<10.4f} {probs[2]:<10.4f} {pred:<8} {ok}")


# ════════════════════════════════════════════════════════════
# SECTION 3: ERM Loss
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: Logistic ERM Average Loss")
print("=" * 60)

p_vals = sigmoid(th0 + th1 * ec)
erm_losses = [-(y_bin[i]*np.log(p_vals[i]+1e-15) +
                (1-y_bin[i])*np.log(1-p_vals[i]+1e-15))
              for i in range(10)]
print(f"ERM Average Loss = {np.mean(erm_losses):.6f}")


# ════════════════════════════════════════════════════════════
# SECTION 4: Generalization
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4: Generalization (Train/Val/Test)")
print("=" * 60)

train_ec=ec[:7]; train_y=y_bin[:7]
val_ec=ec[7:9];  val_y=y_bin[7:9]
test_ec=ec[9:];  test_y=y_bin[9:]

# Retrain on train only
th0_t, th1_t = 0.0, 0.0
for _ in range(5000):
    err = sigmoid(th0_t + th1_t * train_ec) - train_y
    th0_t -= 0.05 * np.mean(err)
    th1_t -= 0.05 * np.mean(err * train_ec)

r_train = bce_loss(th0_t, th1_t, train_ec, train_y)
r_val   = bce_loss(th0_t, th1_t, val_ec, val_y)
r_test  = bce_loss(th0_t, th1_t, test_ec, test_y)
print(f"R_train = {r_train:.6f}")
print(f"R_val   = {r_val:.6f}")
print(f"R_test  = {r_test:.6f}")
print(f"Gen Gap = {abs(r_test - r_train):.6f}")


# ════════════════════════════════════════════════════════════
# SECTION 5: Square Error Loss
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: Square Error Loss (Regression)")
print("=" * 60)

h_const = ec.mean()
mse     = np.mean((h_const - ec)**2)
rmse    = np.sqrt(mse)
print(f"Optimal constant h*(x) = mean(EC) = {h_const:.4f}")
print(f"MSE  = {mse:.6f}")
print(f"RMSE = {rmse:.6f}")


# ════════════════════════════════════════════════════════════
# SECTION 6: Bias-Variance Decomposition
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 6: Bias-Variance Decomposition")
print("=" * 60)

x_s    = np.argsort(ec); ec_s=ec[x_s]; yr_s=y_reg[x_s]
DA_idx = [0,2,4,6,8]; DB_idx=[1,3,5,7,9]; DC_idx=[0,1,2,3,4]

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    hA = np.polyfit(ec_s[DA_idx], yr_s[DA_idx], 2)
    hB = np.polyfit(ec_s[DB_idx], yr_s[DB_idx], 2)
    hC = np.polyfit(ec_s[DC_idx], yr_s[DC_idx], 2)

pA=np.polyval(hA,ec_s); pB=np.polyval(hB,ec_s); pC=np.polyval(hC,ec_s)
hbar=np.mean([pA,pB,pC],axis=0)
hstar=2*ec_s**2-4*ec_s+2

variance = float(np.mean([(p-hbar)**2 for p in [pA,pB,pC]]))
bias2    = float(np.mean((hbar-hstar)**2))
noise    = float(np.mean((hstar-yr_s)**2))
total    = variance + bias2 + noise

print(f"Variance = {variance:.6f} ({variance/total*100:.1f}%)")
print(f"Bias^2   = {bias2:.6f}   ({bias2/total*100:.1f}%)")
print(f"Noise    = {noise:.6f}   ({noise/total*100:.1f}%)")
print(f"Total    = {total:.6f}")


# ════════════════════════════════════════════════════════════
# REGULARIZATION: Ridge & Lasso
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("REGULARIZATION: Ridge (L2) & Lasso (L1)")
print("=" * 60)

def gd_reg(X, Y, lr=0.05, epochs=5000, lam=0.0, reg='none'):
    t0, t1 = 0.0, 0.0; N = len(X)
    for _ in range(epochs):
        err = sigmoid(t0 + t1*X) - Y
        g0 = np.mean(err); g1 = np.mean(err*X)
        if reg == 'ridge': g1 += (lam/N)*t1
        elif reg == 'lasso': g1 += (lam/N)*np.sign(t1)
        t0 -= lr*g0; t1 -= lr*g1
    return t0, t1

th0_r, th1_r = gd_reg(ec, y_bin, lam=0.5, reg='ridge')
th0_l, th1_l = gd_reg(ec, y_bin, lam=0.5, reg='lasso')

print(f"\nERM:         theta0={th0:.4f}  theta1={th1:.4f}  ||theta||^2={th0**2+th1**2:.4f}")
print(f"Ridge(0.5):  theta0={th0_r:.4f}  theta1={th1_r:.4f}  ||theta||^2={th0_r**2+th1_r**2:.4f}")
print(f"Lasso(0.5):  theta0={th0_l:.4f}  theta1={th1_l:.4f}  ||theta||^2={th0_l**2+th1_l**2:.6f}")

print("\n✅ Complete ML Pipeline Done! Check /results folder for all figures.")


'pricer swaption con 4 elementi'

import numpy as np
import scipy.stats as spss
import scipy.optimize as spopt
import seaborn as sns
import QuantLib as ql
import matplotlib.pyplot as plt
import math


'prima scrivo il pricing di una swaption con 4 elementi nel basket sotto la simulazione del montecarlo (quindi 8 componenti)'

def BlackBasketPayoffMC(f0, k, alpha, sigma, expiry, n_sample=1000000, seed=42, mc_error=False):
    # Simulazione delle variabili casuali per i quattro componenti di ciascun basket
    eps = spss.norm.rvs(loc=0, scale=1, size=(4, n_sample), random_state=seed)
    
    
    # Drift per i quattro componenti di ciascun basket
    drift1 = [-0.5 * sigma1[i] * sigma1[i] * expiry for i in range(4)]
    drift2 = [-0.5 * sigma2[i] * sigma2[i] * expiry for i in range(4)]
    
    # Diffusione (parte stocastica) per i quattro componenti di ciascun basket
    diff1 = [sigma1[i] * math.sqrt(expiry) * eps[i] for i in range(4)]
    diff2 = [sigma2[i] * math.sqrt(expiry) * eps[i] for i in range(4)]
    
    # Calcolo del valore di ciascun componente di ogni basket
    basket1_components = [alpha1[i] * (np.exp(drift1[i] + diff1[i]) - 1) for i in range(4)]
    basket2_components = [alpha2[i] * (np.exp(drift2[i] + diff2[i]) - 1) for i in range(4)]
    
    # Somma di tutti i componenti del basket
    basket1 = sum(basket1_components)
    basket2 = sum(basket2_components)
    
    # Calcolo del valore del tasso alla scadenza
    Rt = f0 + basket1 + basket2
    
    # Payoff della swaption
    payoff_sample = np.clip(Rt - k, 0, np.inf)
    payoff_mean = payoff_sample.mean()
    
    if mc_error:
        # Calcolo della varianza della stima e dell'errore Monte Carlo
        estimate_variance = payoff_sample.var(ddof=1) / n_sample
        mc_error_value = math.sqrt(estimate_variance)
        return payoff_mean, mc_error_value
    
    return payoff_mean

# Esempio di parametri con 4 alfa e 4 sigma per ciascun basket
f0 = 0.02  # Tasso iniziale
k = 0.025  # Strike

# Quattro pesi e volatilità per ogni basket
alpha1 = [0.5, 0.3, 0.2, 0.1]  # Pesi per il primo elemento del basket
alpha2 = [0.4, 0.3, 0.2, 0.1]  # Pesi per il secondo elemento del basket
sigma1 = [0.2, 0.15, 0.1, 0.05]  # Volatilità per il primo elemento del basket
sigma2 = [0.25, 0.2, 0.15, 0.1]  # Volatilità per il secondo elemento del basket

expiry = 1  # Tempo alla scadenza in anni
n_sample = 1000000  # Numero di simulazioni

# Calcolo del prezzo con Monte Carlo
price, error = BlackBasketPayoffMC(f0, k, alpha1, alpha2, sigma1, sigma2, expiry, n_sample=n_sample, mc_error=True)
print(f"Prezzo stimato della swaption con 4 componenti per basket: {price:.6f}")
print(f"Errore Monte Carlo: {error:.6f}")


def _black_basket_option_price(a1, a2, g1, g2, B):
        return a1*spss.norm.cdf(g1-B)+ a2*spss.norm.cdf(g2-B)
    
def BlackBasketApproximatePayoffMaximized(f0, k, alpha1, alpha2, sigma1, sigma2, expiry, convergence = False):
    stdev1 = sigma1*math.sqrt(expiry)
    stdev2 = sigma2*math.sqrt(expiry)
    var1 = stdev1*stdev1
    var2 = stdev2*stdev2
    
    boundary_constant = k - f0 + 0.5*(alpha1*var1 + alpha2*var2)
    total_stdev = math.sqrt(alpha1*alpha1*var1+alpha2*alpha2*var2)
    B = boundary_constant/total_stdev
    
    gamma1= alpha1*var1 / total_stdev
    gamma2= alpha2*var2 / total_stdev
    
    'implementazione3: con minimizzazione con moltiplicatore di Lagrange'
    'calcolo con il segno meno perchè posso fare solo la minimizzazione, quindi minimizzo una funzione negativa'
    
    opt_price_func = lambda x: -_black_basket_option_price(alpha1, alpha2, x[0], x[1],x[2])
    opt_price_constr = lambda x: x[0]*x[0]/var1 + x[1]*x[1]/var2 -1.0
    
    minimization_constrains = ({"type": "eq", "fun": opt_price_constr}, )
    
    opt = spopt.minimize(fun = opt_price_func, x0 = [gamma1,gamma2,B], method= "SLSQP", constraints= minimization_constrains)
    
    if convergence:
        print(opt)
        
    gamma1, gamma2, B = opt.x
    

    return _black_basket_option_price(alpha1, alpha2, gamma1, gamma2, B)


    'adesso ottimizziamo B'
    
def _B_root(B_candidate, a1, a2, g1, g2, m):
    basket1 = a1*math.exp(g1*B_candidate-0.5*g1*g1)
    basket2 = a2*math.exp(g2*B_candidate-0.5*g2*g2)
    f = basket1 + basket2 + m
    fprime = g1*basket1 + g2*basket2
    return f, fprime


def BlackBasketPayoffMC(f0,k, alpha1, alpha2, sigma1,sigma2, expiry, n_sample= 1000000, seed=42, mc_error=False):
    eps1, eps2 = spss.norm.rvs(loc=0, scale=1, size=(2,n_sample), random_state = seed)
    
    drift1 = -0.5*sigma1*sigma1*T0
    drift2 = -0.5*sigma2*sigma2*T0
    diff1 = sigma1*math.sqrt(expiry)*eps1
    diff2 = sigma2*math.sqrt(expiry)*eps2
    
    basket1= np.exp(drift1 + diff1) -1
    basket2= np.exp(drift2 + diff2) -1
    
    Rt = f0 + alpha1*basket1 + alpha2*basket2
    payoff_sample = np.clip(Rt-k,0, np.inf)
    payoff_mean = payoff_sample.mean()
    
    if  mc_error:
        estimate_variance = payoff_sample.var(ddof=1)/n_sample
        mc_error = math.sqrt(estimate_variance)
        return payoff_mean, mc_error
    return payoff_mean

  
    
    
q99= spss.norm.ppf(0.99)
mc_forward_payoff, mc_stdev = BlackBasketPayoffMC(f0, strike, a1, a2, s1, s2, T0, n_sample = 1000000, mc_error = True)
mc_payoff_ub = mc_forward_payoff + q99*mc_stdev    
mc_payoff_lb = mc_forward_payoff - q99*mc_stdev

mc_ivol_ub = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_payoff_ub, discount = 1) *1e4   
mc_ivol_lb = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_payoff_lb, discount = 1) *1e4
mc_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_forward_payoff, discount = 1) *1e4    

maximization_forward_payoff = BlackBasketApproximatePayoffMaximized(f0, strike, a1, a2, s1, s2, T0)
maximization_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, maximization_forward_payoff, discount=1) * 1e4 


print(f"mc payoff ({T0}y{T}y @ {strike*100:.2f}%):\t\t{mc_forward_payoff*1e4:.2f}bps / [{mc_payoff_lb*1e4:.2f}, {mc_payoff_ub*1e4:.2f}]bps")
print(f"mc ivol:\t\t\t\t{mc_ivol:.2f}bps / [{mc_ivol_lb:.2f}, {mc_ivol_ub:.2f}]bps")
print()
print(f"maximization payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{maximization_forward_payoff*1e4:.2f}bps")
print(f"maximization ivol:\t\t\t{maximization_ivol:.2f}bps")
print()


def _black_basket_option_price(a1,a2,a3,a4,g1,g2,g3,g4,B):
    return a1*spss.norm.cdf(g1-B)+ a2*spss.norm.cdf(g2-B)+a3*spss.norm.cdf(g3-B)+ a4*spss.norm.cdf(g4-B)


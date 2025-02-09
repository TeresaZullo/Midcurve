'pricer swaption con 4 elementi'

import numpy as np
import scipy.stats as spss
from scipy.stats import multivariate_normal
import scipy.optimize as spopt
from scipy.optimize import minimize
import seaborn as sns
import QuantLib as ql
from QuantLib import *
import xlwings as xw
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm


percorso = r'C:\Users\T004697\Desktop\oswap_price\DATI_TESI.xlsm'
file = xw.Book(percorso)
curve_sheet = file.sheets['curves']
vols_sheet = file.sheets['vol']

f0_1 = -0.0019
f0_2 = 0.00022  
strike = 0

alpha_1 = [0.00624, -0.00441]
alpha_2 = [0.00787, -0.00714]  
alpha_3 = f0_1 + f0_2 - strike - sum(alpha_1) - sum(alpha_2)
alpha = alpha_1 + alpha_2 + [alpha_3]


sigma = [0.5132, 0.5132]
eta =  [0.5246 , 0.5246]  
#creo la matrice delle standard deviation data da sigma+ eta + 0 
stdev_array = np.array(sigma + eta)

T0 = 1  # expiry swaption
T_1 = 5 # tenor swap
T_2 = 10 # tenor swap

n_sample = 1000000  # Numero di simulazioni

# Esempio di input per gli angoli
theta_11 = -1.3
theta_12 = 0.0
theta_21 = 0.0
theta_22 = 0.00
theta = [theta_11, theta_12, theta_21, theta_22]
rho_matrix_1 = np.eye(len(alpha_1))
rho_matrix_2 = np.eye(len(alpha_2))
    
def cov_matrix(sigma,theta):
    
    theta_11, theta_12, theta_21, theta_22 = theta

    C_Z = np.array([
            [np.sin(theta_11), np.cos(theta_11) * np.sin(theta_12)],
            [np.cos(theta_11)*np.sin(theta_21), np.cos(theta_21)*np.sin(theta_22) * np.cos(theta_12) - np.sin(theta_21)*np.sin(theta_11) * np.sin(theta_12)]
        ])
    
    C_Z_T = C_Z.T

    #adesso creo le matrici identità
    identity_matrix_1 = np.eye(2)
    identity_matrix_2 = np.eye(2)
    
    #creo la matrice di correlazione 
    rho_matrix = np.block([
        [identity_matrix_1, C_Z],
        [C_Z_T, identity_matrix_2]
    ])
    
    stdev_matrix = np.diag(stdev_array)
    #creo la matrice delle covarianze 
    var_cov_matrix = stdev_matrix @ rho_matrix @ stdev_matrix
    
    n_elements = var_cov_matrix.shape[0]
    new_row = np.zeros((1, n_elements))  # Nuova riga di zeri
    new_col = np.zeros((n_elements + 1, 1))  # Nuova colonna di zeri

    # Aggiunta della nuova riga
    var_cov_matrix = np.vstack((var_cov_matrix, new_row))
    # Aggiunta della nuova colonna
    var_cov_matrix = np.hstack((var_cov_matrix, new_col))

    return var_cov_matrix, rho_matrix

var_cov_matrix, rho_matrix = cov_matrix(sigma, theta)

eigenvalues = np.linalg.eigvals(var_cov_matrix)
print("Autovalori:", eigenvalues)


print("Matrice di cross-correlation (C_Z):")
print(var_cov_matrix[:2, 2])  

print("Trasposta della matrice di cross-correlation (C_Z_T):")
print(var_cov_matrix[2:, :2]) 
print("Covariance matrix:")
print(rho_matrix) 

print("Matrice di varianza-covarianza:")
print(var_cov_matrix)

#pricer con metodo Montecarlo generico per swaption 

def BlackBasketPayoffMC(f0, strike, alpha, sigma, rho_matrix, T0, n_sample=1000000, seed=42, mc_error=True):
    
    n_elements = len(alpha)

    mean = np.zeros(n_elements)
    eps = spss.multivariate_normal.rvs(mean=mean, cov= rho_matrix, size=n_sample, random_state=seed)

     
    # Drift e diffusione per ogni componente del basket
    drift = -0.5 * np.array(sigma) ** 2 * T0
    diff = np.array([sigma[i] * np.sqrt(T0) * eps[:, i] for i in range(n_elements)])
    
    basket = np.exp(drift[:, None] + diff) - 1
    
    # Somma di tutti i 4 componenti per formare il basket finale
    basket = np.sum(np.array(alpha)[:, None] * basket, axis=0)
    
    Rt = f0 + basket
    
    # Payoff della swaption
    payoff_sample = np.clip(Rt - strike, 0, np.inf)
    payoff_mean = payoff_sample.mean()
    
    if mc_error:
         estimate_variance = payoff_sample.var(ddof=1) / n_sample
         mc_error_value = math.sqrt(estimate_variance)
         return payoff_mean, mc_error_value
     
    return payoff_mean 

#per swaption 1
payoff_1, stdev_1 = BlackBasketPayoffMC(f0_1, strike, alpha_1, sigma, rho_matrix_1, T0, n_sample=n_sample, seed = 42, mc_error=True)
    
#per swaption 2
payoff_2, stdev_2 = BlackBasketPayoffMC(f0_2, strike, alpha_2, eta, rho_matrix_2, T0, n_sample=n_sample, seed = 42, mc_error=True)

#Calcolo degli intervalli di confidenza
q99 = spss.norm.ppf(0.99)
#per swaption 1
mc_payoff_ub_1 = payoff_1 + q99 * stdev_1
mc_payoff_lb_1 = payoff_1 - q99 * stdev_1

#Calcolo delle volatilità implicite con Bachelier
mc_ivol_ub_1 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_1, T0, mc_payoff_ub_1, discount=1) * 1e4
mc_ivol_lb_1 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_1, T0, mc_payoff_lb_1, discount=1) * 1e4
mc_ivol_1 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_1, T0, payoff_1, discount=1) * 1e4
    
#per swaption 2 
mc_payoff_ub_2 = payoff_2 + q99 * stdev_2
mc_payoff_lb_2 = payoff_2 - q99 * stdev_2

# Calcolo delle volatilità implicite con Bachelier
mc_ivol_ub_2 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_2, T0, mc_payoff_ub_2, discount=1) * 1e4
mc_ivol_lb_2 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_2, T0, mc_payoff_lb_2, discount=1) * 1e4
mc_ivol_2 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_2, T0, payoff_2, discount=1) * 1e4

# Stampa risultati
print("Swaption 1:")
print(f"Payoff mean: {payoff_1*1e4:.2f}bps, LB: {mc_payoff_lb_1*1e4:.2f}bps, UB: {mc_payoff_ub_1*1e4:.2f}bps")
print(f"Implied Volatility: {mc_ivol_1:.2f}bps")

print("\nSwaption 2:")
print(f"Payoff mean: {payoff_2*1e4:.2f}bps, LB: {mc_payoff_lb_2*1e4:.2f}bps, UB: {mc_payoff_ub_2*1e4:.2f}bps")
print(f"Implied Volatility: {mc_ivol_2:.2f}bps")
    

def build_var_matrix(std_dev):
    #Crea una matrice 2x2 di varianze a partire da standard deviation.
    std_dev_matrix = np.diag(std_dev)  
    variance_matrix = std_dev_matrix @ std_dev_matrix  
    return variance_matrix

def calculate_B(strike, f0, alpha, sigma, var_cov_matrix, expiry):
    
    n_elements = len(alpha)
    
    # Converte alpha in un array NumPy e lo trasforma in un vettore colonna
    alpha = np.array(alpha).reshape((n_elements, 1))  # Vettore colonna
    alpha_T = alpha.T  # Vettore riga (trasposto)
    
    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
    var = [stdev[i]*stdev[i] for i in range(n_elements)]
    numerator = strike - f0 + 0.5 * sum(alpha[i] * var[i] for i in range(n_elements))
        
    denominator_B = np.dot(np.dot(alpha_T,var_cov_matrix), alpha)
    denominator_B = denominator_B.item()
            
    total_stdev = math.sqrt(denominator_B)
     
    B = numerator / total_stdev
    
    return B, total_stdev

#costruisco le matrici var_cov che dovranno essere utilizzare per il calcolo di B in caso di signole Swaption
var_cov_matrix_1 = build_var_matrix(sigma)
var_cov_matrix_2 = build_var_matrix(eta)

B_1, total_stdev_1 = calculate_B(strike, f0_1, alpha_1, sigma, var_cov_matrix_1, T0)
B_2, total_stdev_2 = calculate_B(strike, f0_2, alpha_2, eta, var_cov_matrix_2, T0)


#Creo B per la midcurve swaption 
def calculate_B_MC(f0_1, f0_2, strike, alpha, var_cov_matrix, expiry):
    
    n_elements = len(alpha)
    
    # Converte alpha in un array NumPy e lo trasforma in un vettore colonna
    alpha = np.array(alpha).reshape((n_elements, 1)) 
    alpha_T = alpha.T  
    
    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
    var = [stdev[i]*stdev[i] for i in range(n_elements)]
    numerator = strike - (f0_1 + f0_2) + 0.5 * sum(alpha[i] * var[i] for i in range(n_elements))
        
    denominator_B_MC = np.dot(np.dot(alpha_T,var_cov_matrix), alpha)
    denominator_B_MC = denominator_B_MC.item()
            
    total_stdev = math.sqrt(denominator_B_MC)
     
    B_MC = numerator / total_stdev
    
    return B_MC, total_stdev

def calculate_Gamma(beta, var_cov_matrix, expiry):
    
    beta = np.array(beta).reshape((-1, 1))
    # Calcolo di gamma utilizzando la matrice di varianza-covarianza
    gamma = np.dot(var_cov_matrix, beta) * expiry

    return gamma.flatten()

def BlackBasketApprossimativePayoff(f0, strike, alpha, sigma, theta, var_cov_matrix, expiry):
    n_elements = len(alpha)
    
    B, total_stdev = calculate_B(strike, f0, alpha, sigma, var_cov_matrix, expiry)
    beta = np.array(alpha)/ total_stdev
    gamma = calculate_Gamma(beta,var_cov_matrix, expiry)
    
    cdf_values = [norm.cdf(gamma[i] - B) for i in range(n_elements)]

    payoff = np.dot(alpha, cdf_values) + (f0 - strike - sum(alpha)) * norm.cdf(-B)
   
    return payoff, B, gamma

analytical_forward_payoff_1, B_1, gamma_1 = BlackBasketApprossimativePayoff(f0_1, strike, alpha_1,sigma, theta, var_cov_matrix_1, T0)
analytical_forward_payoff_2, B_2, gamma_2 = BlackBasketApprossimativePayoff(f0_2, strike, alpha_2, eta, theta, var_cov_matrix_2, T0)

# Verifica e conversione in scalare per payoff_1
if isinstance(analytical_forward_payoff_1, np.ndarray):
    analytical_forward_payoff_1 = analytical_forward_payoff_1.item()

# Verifica e conversione in scalare per payoff_2
if isinstance(analytical_forward_payoff_2, np.ndarray):
    analytical_forward_payoff_2 = analytical_forward_payoff_2.item()

    
analytical_ivol_1 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_1, T0, analytical_forward_payoff_1, discount=1) * 1e4
analytical_ivol_2 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_2, T0, analytical_forward_payoff_2, discount=1) * 1e4
    
print("Swaption 1:")
print(f"Analytical Payoff ({T0}y @ {strike*100:.2f}%):\t{analytical_forward_payoff_1*1e4:.2f}bps")
print(f"Analytical Implied Volatility:\t\t\t{analytical_ivol_1:.2f}bps")

print("\nSwaption 2:")
print(f"Analytical Payoff ({T0}y @ {strike*100:.2f}%):\t{analytical_forward_payoff_2*1e4:.2f}bps")
print(f"Analytical Implied Volatility:\t\t\t{analytical_ivol_2:.2f}bps")






class AnnuityApproximation:
    def __init__(self, T0, maturities, rates):
       
        self.T0 = T0
        self.maturities = maturities
        self.rates = rates  
        if len(maturities) != len(rates):
            raise ValueError("the maturity and rates must have the same array's length")

    def calculate_annuity(self):
        annuities = []

        for i in range(len(self.maturities)):
            T_i = self.maturities[i]
            delta_T = T_i - self.T0

            # Se il tasso è una funzione, valutalo in T0, altrimenti usa il valore fisso
            if callable(self.rates[i]):
                R_value = self.rates[i](self.T0)
            else:
                R_value = self.rates[i]

            annuity = delta_T - 0.5 * R_value * (delta_T ** 2)
            annuities.append(annuity)

        return annuities
    
    def calculate_mid_curve_annuity(self, annuities):

        #Calcolo l'annuity della mid-curve come differenza tra le due annuity
        if len(annuities) < 2:
            raise ValueError("At least two annuities are needed to calculate the mid-curve annuity.")
        
        mid_curve_annuity = annuities[1] - annuities[0]
        return mid_curve_annuity

maturities = [T_1 + T0, T_2 + T0]  
rates = [f0_1, f0_2] 
T0 = T0

annuity_rates = AnnuityApproximation(T0=T0, maturities=maturities, rates=rates)

annuities = annuity_rates.calculate_annuity()

for i, annuity in enumerate(annuities):
     print(f"Annuity_{maturities[i]}Y: {annuity:.6f}")

mid_curve_annuity = annuity_rates.calculate_mid_curve_annuity(annuities)
print(f"MC Annuity: {mid_curve_annuity:.6f}")

class MeasureApproximation:
    def __init__(self, annuities, maturities, T0, rates):

        self.annuities = annuities
        self.maturities = maturities
        self.T0 = T0
        self.rates = rates
        
    def calculate_lambda(self):
  
        lambdas = []

        for i in range(len(self.annuities)):
            A_i_0 = self.annuities[i]
            T_i = self.maturities[i]

            for j in range(i+1, len(self.annuities)):
                A_j_0 = self.annuities[j]
                T_j = self.maturities[j]

                # Calcolo del denominatore
                denom = A_j_0 - A_i_0
                if denom == 0:
                    raise ValueError("A_j_0 and A_i_0 cannot be egual")

                delta_T_squared = (T_i - self.T0)**2 - (T_j - self.T0)**2
                lambda_i = 0.5 * (delta_T_squared / denom + (T_i - self.T0)**2 / A_i_0)
                lambda_j = 0.5 * (delta_T_squared / denom + (T_j - self.T0)**2 / A_j_0)
                lambdas.append((lambda_i,lambda_j))

        return lambdas

measure_approx = MeasureApproximation(annuities=annuities, maturities=maturities, T0=T0, rates=rates)


lambdas = measure_approx.calculate_lambda()

for i, (lambda_i,lambda_j) in enumerate(lambdas):
    print(f"λ_i: {lambda_i:.6f}")
    print(f"λ_j: {lambda_j:.6f}")


'devo travere R hlat = prezzo swaption sotto misura midcurve (utilizzando lambda)'

def BlackBasketApprossimativeSigmaHat(f0, strike, alpha, sigma, theta, expiry, T0, midcurve_annuity_measure):
    
    annuity_approx = AnnuityApproximation(T0=T0, maturities=maturities, rates=rates)
    annuities = annuity_approx.calculate_annuity()
    
    # Calcola il valore di lambda (fattore di cambio misura) usando la classe MeasureApproximation
    measure_approx = MeasureApproximation(annuities=annuities, maturities=maturities, T0=T0, rates=rates)
    lambdas = measure_approx.calculate_lambda()
    
    # Usa il primo valore di lambda calcolato come fattore per adattare la misura
    midcurve_annuity_measure = lambdas[0][0]  # Scegli il valore di lambda appropriato
    
    
    adj_alhpa = [a * midcurve_annuity_measure for a in alpha]
    adj_strike = strike * midcurve_annuity_measure 
    adj_forward = f0 * midcurve_annuity_measure
    
    model_price = BlackBasketApprossimativePayoff(adj_forward, adj_strike, adj_alpha, sigma, theta, var_cov_matrix, expiry)

    sigmahat = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, expiry, model_price, discount = 1.0)
    print("Implied Vol SigmaHat:", sigmahat)
    print("Payoff swaption under midcurve annuity measure:", model_price)


def BlackBasketObjectiveFunctionSigma(sigmahat, f0, strikes, alpha, theta, expiry, european_options_market_prices):
    errors = []
    for k, mkt_price in zip(strikes, european_options_market_prices):
        model_price = BlackBasketApprossimativePayoff(f0, k, alpha, sigmahat, theta, expiry)
        error = (model_price - mkt_price) / mkt_price
        errors.append(error)

    errors = np.array(errors)

    return errors @ errors


'swaption price con 4 elementi con approssimazione del black basket con lagrange'

def _black_basket_option_price(f0, strike, alpha, sigma, var_cov_matrix, gamma, B, expiry):
    
    n_elements = len(alpha)
    
    cdf_values = [norm.cdf(gamma[i] - B) for i in range(n_elements)]

    payoff = np.dot(alpha, cdf_values)
    
    return payoff


def BlackBasketApproximatePayoffMaximized(f0, strike, alpha, sigma, var_cov_matrix, expiry, convergence = False):
    
    n_elements = len(alpha)
    
    B, total_stdev = calculate_B(strike, f0, alpha, sigma, var_cov_matrix, expiry)
    
    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
    var = [stdev[i]*stdev[i] for i in range(n_elements)]
    
    
    beta = [alpha[i]/total_stdev for i in range(n_elements)]
    gamma = calculate_Gamma(beta, var_cov_matrix, expiry)
    gamma = gamma.flatten()
    
    B = B.item()
    x0 = np.append(gamma,B)
    
    #calcolo matrice var-cov inversa
    # Considero la sottomatrice 4x4 di var_cov_matrix, ignorando l'ultima riga e colonna
    var_cov_4x4 = var_cov_matrix[:4, :4]
    # Calcola l'inversa della sottomatrice 4x4
    var_cov_inv_4x4 = np.linalg.inv(var_cov_4x4)
    # Creo una nuova matrice 5x5 e inserisco l'inversa 4x4 al suo interno
    var_cov_inv = np.zeros((5, 5))
    var_cov_inv[:4, :4] = var_cov_inv_4x4
    
    'implementazione3 con minimizzazione con moltiplicatore di Lagrange'
    'calcolo con il segno meno perchè posso fare solo la minimizzazione, minimizzo una funzione negativa'
    opt_price_func = lambda x: -_black_basket_option_price(f0, strike, alpha, sigma, var_cov_matrix, x[:n_elements], x[n_elements], expiry)
    opt_price_constr = lambda x:  np.dot(np.dot(x[:n_elements].T, var_cov_inv), x[:n_elements]) - 1.0

    minimization_constrains = ({"type": "eq", "fun": opt_price_constr}, )
    
    opt = spopt.minimize(fun = opt_price_func, x0 = x0, method= "SLSQP", constraints= minimization_constrains)
    
    if convergence:
        print(opt)
        
    gamma = opt.x[:n_elements]
    B = opt.x[n_elements]
    print(f"stdev: {stdev}")
    print(f"var: {var}")
    print(f"Gamma ottimizzato: {gamma}")
    print(f"B ottimizzato: {B}")
    return _black_basket_option_price(f0, strike, alpha, sigma, var_cov_matrix, gamma, B, expiry)


maximization_forward_payoff = BlackBasketApproximatePayoffMaximized(f0, strike, alpha, sigma, var_cov_matrix, T0)

if isinstance(maximization_forward_payoff, np.ndarray):
    maximization_forward_payoff = maximization_forward_payoff.item()
    
maximization_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, maximization_forward_payoff, discount=1) * 1e4 


print(f"maximization payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{maximization_forward_payoff*1e4:.2f}bps")
print(f"maximization ivol:\t\t\t{maximization_ivol:.2f}bps")

#proviamo a trovare gamma e B con il metodo dell'iterazione


    
def _B_root(B_candidate, a, g, m):
    
    n_elements = len(alpha)
    
    basket = 0
    fprime = 0
    
    for i in range(n_elements):
        basket_i = a[i]*np.exp(g[i]*B_candidate-0.5*g[i]*g[i])
        basket += basket_i 
        f = basket + m
        fprime += basket_i * g[i]
   
    return f, fprime

 
def BlackBasketPayoffIterative(f0, k, alpha, sigma, var_cov_matrix, expiry, N=2, tolerance = 1e-8, convergence = False):

    n_elements = len(alpha)

    B, total_stdev = calculate_B(strike, f0, alpha, sigma, var_cov_matrix, expiry)

    stdev = [sigma[i]*np.sqrt(expiry) for i in range(n_elements)]     
    var = [stdev[i]*stdev[i] for i in range(n_elements)]


    beta = [alpha[i]/total_stdev for i in range(n_elements)]
    gamma = calculate_Gamma(beta, var_cov_matrix, expiry)
    gamma = gamma.flatten()



   # Inizializzazione del payoff
    p0 = -10000
    p1 = _black_basket_option_price(f0, k, alpha, sigma, var_cov_matrix, gamma, B, expiry)

   # Iterazione per il calcolo di gamma e B
    while abs(p1 - p0) > tolerance:
       p0 = p1

       # Trova i gamma ottimali per ogni elemento
       w = np.array([alpha[i] * spss.norm.pdf(gamma[i] - B) for i in range(n_elements)])
       w = np.array(w).reshape((-1, 1))
       
       gamma_denom = np.sqrt(np.dot(np.dot(w.T, var_cov_matrix), w).item())
       gamma = np.dot(var_cov_matrix, w) / gamma_denom
       gamma = gamma.flatten()
       
       # Ottimizzazione per trovare B utilizzando la funzione _B_root generalizzata
       opt = spopt.root_scalar(f=_B_root, method="newton", x0=B, fprime=True,
                               args=(alpha, gamma, f0 - k - sum(alpha)))
       B = opt.root

       if convergence:
           print(opt)
       
       p1 = _black_basket_option_price(f0, k, alpha, sigma, var_cov_matrix, gamma, B, expiry)
   
    return p1
    

iterative_forward_payoff = BlackBasketPayoffIterative(f0, strike, alpha, sigma, var_cov_matrix, T0, tolerance=1e-8)

if isinstance(iterative_forward_payoff, np.ndarray):
    iterative_forward_payoff = iterative_forward_payoff.item()
    
iterative_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, iterative_forward_payoff, discount=1) * 1e4

print(f"iterative payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{iterative_forward_payoff*1e4:.2f}bps")
print(f"iterative ivol:\t\t\t\t{iterative_ivol:.2f}bps")




'pricer swaption con 4 elementi'

import numpy as np
import scipy.stats as spss
from scipy.stats import multivariate_normal
import scipy.optimize as spopt
import seaborn as sns
import QuantLib as ql
from QuantLib import *
import xlwings as xw
import pandas as pd
import matplotlib.pyplot as plt
import math


f0 = -0.0019  
strike = 0

# Quattro pesi e volatilità per ogni basket
alpha = [0.00624, -0.00441, 0.00787, -0.00714]  
sigma = [0.5132, 0.5132, 0.5246 , 0.5246,0]  
new_alpha = f0 - strike - sum(alpha)
alpha = np.append(alpha, new_alpha)


T0 = 1  # expiry swaption
T = 5 # tenor swap

n_sample = 1000000  # Numero di simulazioni

# Esempio di input per gli angoli
theta_11 = 0.01
theta_12 = 0.01
theta_21 = 0.01
theta_22 = 0.00

'prima scrivo il pricing di una swaption con n elementi nel basket sotto la simulazione del montecarlo (quindi nx2 componenti)'

def var_cov_matrix(sigma,theta_11,theta_12,theta_21,theta_22):

    def Cross_correlation_matrix(theta_11, theta_12, theta_21, theta_22):
    # Costruzione della matrice 
        C_Z = np.array([
            [np.sin(theta_11), np.cos(theta_11) * np.sin(theta_12)],
            [np.cos(theta_11)*np.sin(theta_21), np.cos(theta_21)*np.sin(theta_22) * np.cos(theta_12) - np.sin(theta_21)*np.sin(theta_11) * np.sin(theta_12)]
        ])
    
        return C_Z

    def transpose_Cross_Correlation_matrix(matrix):
        return np.transpose(matrix)


    # Creazione della matrice C_Z e la sua trasposta
    C_Z = Cross_correlation_matrix(theta_11, theta_12, theta_21,theta_22)
    C_Z_T = transpose_Cross_Correlation_matrix(C_Z)

    #adesso creo le matrici identità
    identity_matrix_1 = np.eye(2)
    identity_matrix_2 = np.eye(2)

    #creo 2 matrici 2x2 con le standard dev
    stdev_matrix_1 = identity_matrix_1 @ np.diag(sigma[:2]) 
    stdev_matrix_2 = identity_matrix_2 @ np.diag(sigma[2:4]) 

    #creo la matrice a blocchi 

    block_matrix_1 = np.block([
        [stdev_matrix_1 @ stdev_matrix_1, np.zeros((2, 2))], 
        [np.zeros((2, 2)), stdev_matrix_2 @ stdev_matrix_2]   
    ])


    block_matrix_2 = np.block([
        [np.zeros((2, 2)), C_Z],    # Parte alta a dx con C_Z
        [C_Z_T, np.zeros((2, 2))]   # Parte bassa a sx con C_Z_T
    ])

    # Sommiamo le due matrici a blocchi per ottenere la matrice di varianza-covarianza
    var_cov_total = block_matrix_1 + block_matrix_2
    n_elements = var_cov_total.shape[0]
    new_row = np.zeros((1, n_elements))  # Nuova riga di zeri
    new_col = np.zeros((n_elements + 1, 1))  # Nuova colonna di zeri

    # Aggiunta della nuova riga
    var_cov_total = np.vstack((var_cov_total, new_row))
    # Aggiunta della nuova colonna
    var_cov_total = np.hstack((var_cov_total, new_col))

    return var_cov_total 

var_cov_total = var_cov_matrix(sigma, theta_11, theta_12, theta_21, theta_22)

eigenvalues = np.linalg.eigvals(var_cov_total)
print("Autovalori:", eigenvalues)

print("Matrice di cross-correlation (C_Z):")
print(var_cov_total[:2, 2:])  # Stampa della parte C_Z

print("Trasposta della matrice di cross-correlation (C_Z_T):")
print(var_cov_total[2:, :2])  # Stampa della parte C_Z_T

print("Matrice delle deviazioni standard 1:")
print(var_cov_total[:2, :2])  # Stampa della matrice quadrata delle deviazioni standard 1

print("Matrice delle deviazioni standard 2:")
print(var_cov_total[2:, 2:])  # Stampa della matrice quadrata delle deviazioni standard 2

print("Matrice di varianza-covarianza totale:")
print(var_cov_total)


def BlackBasketPayoffMC(f0, strike, alpha, sigma, var_cov_total, T0, n_sample=1000000, seed=42, mc_error=False):
    
     n_elements = len(alpha)
        
     #generazione di random variables indipendenti
     mean = np.zeros(n_elements)
     eps = spss.multivariate_normal.rvs(mean=mean, cov=var_cov_total, size=n_sample, random_state=seed)
    
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



'questa matrice mi serve per applicare la correzione tra gli elementi del black basket'


q99= spss.norm.ppf(0.99)
mc_forward_payoff, mc_stdev = BlackBasketPayoffMC(f0, strike, alpha, sigma, var_cov_total, T0 , n_sample = 1000000, mc_error = True)
mc_payoff_ub = mc_forward_payoff + q99*mc_stdev    
mc_payoff_lb = mc_forward_payoff - q99*mc_stdev

mc_ivol_ub = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_payoff_ub, discount = 1) *1e4   
mc_ivol_lb = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_payoff_lb, discount = 1) *1e4
mc_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_forward_payoff, discount = 1) *1e4 

print(f"mc payoff ({T0}y{T}y @ {strike*100:.2f}%):\t\t{mc_forward_payoff*1e4:.2f}bps / [{mc_payoff_lb*1e4:.2f}, {mc_payoff_ub*1e4:.2f}]bps")
print(f"mc ivol:\t\t\t\t{mc_ivol:.2f}bps / [{mc_ivol_lb:.2f}, {mc_ivol_ub:.2f}]bps")
print()

#plot da rivedere 

# def plot_volatility():
    
#     strikes = np.arange(-0.015, 0.015 + 0.005, 0.005)
#     black_vols = []


#     for strike in strikes:
#         Rsample = BlackBasketPayoffMC(f0, strike, alpha, sigma, rho, T0 , n_sample = 1000000, mc_error = True)
#         atm_price = np.clip(Rsample - strike, 0, np.inf).mean()
#         black_vol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, atm_price, discount=1)
#         black_vols.append(black_vol)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(strikes, black_vols, marker='o', linestyle='-', color='b', label='Volatilità')
#     plt.xlabel('Strike')
#     plt.ylabel('Black Volatility')
#     plt.title('Black Volatility vs Strike')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# plot_volatility()


# prima di implementare il metodo di approssimanzione con il moltiplicatore di Lagrange, controlliamo se con la formula chiusa base senza iterazioni otteniamo lo stesso risultato del mecaaa

def calculate_B(strike, f0, alpha, sigma, var_cov_total, expiry):
    
    n_elements = len(alpha)
    
    # Converte alpha in un array NumPy e lo trasforma in un vettore colonna
    alpha = np.array(alpha).reshape((n_elements, 1))  # Vettore colonna
    alpha_T = alpha.T  # Vettore riga (trasposto)
    
    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
    var = [stdev[i]*stdev[i] for i in range(n_elements)]
    numerator = strike - f0 + 0.5 * sum(alpha[i] * var[i] for i in range(n_elements))
        
    denominator_B = np.dot(np.dot(alpha_T,var_cov_total), alpha)
    denominator_B = denominator_B.item()
            
    total_stdev = math.sqrt(denominator_B)
     
    B = numerator / total_stdev
    
    return B, total_stdev


def calculate_Gamma(beta, var_cov_total,expiry):
    
    beta = np.array(beta).reshape((-1, 1))  # Vettore colonna
    # Calcolo di gamma utilizzando la matrice di varianza-covarianza
    gamma = np.dot(var_cov_total, beta) * expiry

    return gamma.flatten()
    

def BlackBasketApprossimativePayoff(f0, strike, alpha, sigma, var_cov_total, expiry):
    
    n_elements = len(alpha)
    
    #aggiungo un nuovo alpha = f0 - strike - sum(alpha) e un sigma a 0 
  
 
    #stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
    #var = [stdev[i]*stdev[i] for i in range(n_elements)]
    
    B, total_stdev = calculate_B(strike, f0, alpha, sigma, var_cov_total, expiry)
    beta = [alpha[i] / total_stdev for i in range(n_elements)]
    gamma = calculate_Gamma(beta, var_cov_total, expiry)
    gamma = gamma.flatten()
    
    gamma_minus_B = np.array(gamma - B)
    #gamma_minus_B_matrix = np.diag(gamma_minus_B)
   
   # Calcolare la CDF multivariata per il vettore gamma - B
    cdf_values = multivariate_normal.cdf(gamma_minus_B, mean=np.zeros(n_elements), cov=var_cov_total, allow_singular=True)
    payoff = np.dot(alpha, cdf_values)
   
    return payoff
    

analytical_forward_payoff = BlackBasketApprossimativePayoff(f0, strike, alpha, sigma, var_cov_total, T0)

if isinstance(analytical_forward_payoff, np.ndarray):
    analytical_forward_payoff = analytical_forward_payoff.item()
    
analytical_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, analytical_forward_payoff, discount=1) * 1e4

    
print(f"analytical payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{analytical_forward_payoff*1e4:.2f}bps")
print(f"analytical ivol:\t\t\t{analytical_ivol:.2f}bps")
print()


'swaption price con 4 elementi con approssimazione del black basket con lagrange'

def _black_basket_option_price(f0, strike, alpha, sigma, var_cov_total, gamma, B, expiry):
    
    n_elements = len(alpha)
    
    if len(sigma) != n_elements:
        raise ValueError("alhpa and sigma array must have same length")
        
    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
    var = [stdev[i]*stdev[i] for i in range(n_elements)] 
    
    gamma_minus_B = gamma - B
    gamma_minus_B_matrix = np.diag(gamma_minus_B)
    
    cdf_values = multivariate_normal.cdf(gamma_minus_B_matrix, mean=np.zeros(n_elements), cov=var_cov_total, allow_singular=True)
    payoff = np.dot(alpha, cdf_values)
    
    return payoff


def BlackBasketApproximatePayoffMaximized(f0, strike, alpha, sigma, var_cov_total, expiry, convergence = False):
    
    n_elements = len(alpha)
    
    B, total_stdev = calculate_B(strike, f0, alpha, sigma, var_cov_total, expiry)
    
    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
    var = [stdev[i]*stdev[i] for i in range(n_elements)]
    
    
    beta = [alpha[i]/total_stdev for i in range(n_elements)]
    gamma = calculate_Gamma(beta, var_cov_total, expiry)
    gamma = gamma.flatten()
    
    B = B.item()
    x0 = np.append(gamma,B)
    
    #calcolo matrice var-cov inversa
    var_cov_inv = np.linalg.inv(var_cov_total)
    'implementazione3 con minimizzazione con moltiplicatore di Lagrange'
    'calcolo con il segno meno perchè posso fare solo la minimizzazione, di minimizzo una funzione negativa'
    opt_price_func = lambda x: -_black_basket_option_price(f0, strike, alpha, sigma, var_cov_total, x[:n_elements], x[n_elements], expiry)
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
    return _black_basket_option_price(f0, strike, alpha, sigma, var_cov_total, gamma, B, expiry)


maximization_forward_payoff = BlackBasketApproximatePayoffMaximized(f0, strike, alpha, sigma, var_cov_total, T0)

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

 
def BlackBasketPayoffIterative(f0, k, alpha, sigma, var_cov_total, expiry, N=2, tolerance = 1e-8, convergence = False):

    n_elements = len(alpha)

    B, total_stdev = calculate_B(strike, f0, alpha, sigma, var_cov_total, expiry)

    stdev = [sigma[i]*np.sqrt(expiry) for i in range(n_elements)]     
    var = [stdev[i]*stdev[i] for i in range(n_elements)]


    beta = [alpha[i]/total_stdev for i in range(n_elements)]
    gamma = calculate_Gamma(beta, var_cov_total, expiry)
    gamma = gamma.flatten()



   # Inizializzazione del payoff
    p0 = -10000
    p1 = _black_basket_option_price(f0, k, alpha, sigma, var_cov_total, gamma, B, expiry)

   # Iterazione per il calcolo di gamma e B
    while abs(p1 - p0) > tolerance:
       p0 = p1

       # Trova i gamma ottimali per ogni elemento
       w = np.array([alpha[i] * spss.norm.pdf(gamma[i] - B) for i in range(n_elements)])
       w = np.array(w).reshape((-1, 1))
       
       gamma_denom = np.sqrt(np.dot(np.dot(w.T, var_cov_total), w).item())
       gamma = np.dot(var_cov_total, w) / gamma_denom
       gamma = gamma.flatten()
       
       # for i in range(n_elements):
       #     gamma[i] = w[i] * var[i] / gamma_denom

       # Ottimizzazione per trovare B utilizzando la funzione _B_root generalizzata
       opt = spopt.root_scalar(f=_B_root, method="newton", x0=B, fprime=True,
                               args=(alpha, gamma, f0 - k - sum(alpha)))
       B = opt.root

       if convergence:
           print(opt)
       
       p1 = _black_basket_option_price(f0, k, alpha, sigma, var_cov_total, gamma, B, expiry)
   
    return p1
    

iterative_forward_payoff = BlackBasketPayoffIterative(f0, strike, alpha, sigma, var_cov_total, T0, tolerance=1e-8)

if isinstance(iterative_forward_payoff, np.ndarray):
    iterative_forward_payoff = iterative_forward_payoff.item()
    
iterative_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, iterative_forward_payoff, discount=1) * 1e4

print(f"iterative payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{iterative_forward_payoff*1e4:.2f}bps")
print(f"iterative ivol:\t\t\t\t{iterative_ivol:.2f}bps")


def GetData():
    wb = xw.Book(r'C:\Users\T004697\Desktop\TESI\DATI _TESI.xlsx')
    sh1 = wb.sheets['Eur6m']
    sh1 = wb.sheets['ESTR']
    sh2 = wb.sheets['Vol']
  
    #curve
    df1 = sh1.range('A1').options(pd.DataFrame, expand='table').value
    df2 = sh2.range('A1').options(pd.DataFrame, expand='table').value


    
    Eur6m = DiscountCurve([Date.from_date(d) for d in df1.index.to_list()],
                                                         df1['CURVE_QUOTE'].to_list(),
                                                         Actual360())
    ESTR = DiscountCurve([Date.from_date(d) for d in df2.index.to_list()],
                                                         df2['CURVE_QUOTE'].to_list(),
                                                         Actual360())

   # abilito estrapolazione per entrambe le curve
    eur6m_curve.enableExtrapolation()
    estr_curve.enableExtrapolation()   
    
    # Crea gli handle delle curve
    eur6m_handle = RelinkableYieldTermStructureHandle(eur6m_curve)
    estr_handle = RelinkableYieldTermStructureHandle(estr_curve)
    
    return eur6m_handle, estr_handle

    # Utilizzo della funzione GetData
    eur6m_handle, estr_handle = GetData()

'generalizzo formula Annuity'
'Sulla base dell eq 28 è necessario annuity di R1, annuity di R2, annuity midcurve ' 



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

maturities = [2+T0, 5+T0]  
rates = [-0.003, -0.0019] 
T0 = 1

annuity_rates = AnnuityApproximation(T0=T0, maturities=maturities, rates=rates)

annuities = annuity_rates.calculate_annuity()

for i, annuity in enumerate(annuities):
    print(f"Annuity_{maturities[i]}Y: {annuity:.6f}")


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

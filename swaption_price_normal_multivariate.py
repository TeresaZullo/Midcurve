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
from scipy.stats import norm


f0 = -0.0019  
strike = 0

# Quattro pesi e volatilità per ogni basket
alpha = [0.00624, -0.00441,0.00787, -0.00714]  
sigma = [0.5132, 0.5132,  0.5246 , 0.5246 ,0]  
new_alpha = f0 - strike - sum(alpha)
alpha = np.append(alpha, new_alpha)


T0 = 1  # expiry swaption
T = 5 # tenor swap

n_sample = 1000000  # Numero di simulazioni

# Esempio di input per gli angoli
theta_11 = -1.3
theta_12 = 0.0
theta_21 = 0.0
theta_22 = 0.00
theta = [theta_11, theta_12, theta_21, theta_22]

'prima scrivo il pricing di una swaption con n elementi nel basket sotto la simulazione del montecarlo (quindi nx2 componenti)'

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
    
    #creo la matrice delle standard deviation 
    stdev_matrix = np.diag(sigma[:-1])

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

print("Matrice di varianza-covarianza:")
print(var_cov_matrix)


## da rivedeere simulazione montecarlo. non coincide con l'implied vol della swaption descritta dal paper

def BlackBasketPayoffMC(f0, strike, alpha, sigma, rho_matrix, T0, n_sample=1000000, seed=42, mc_error=False):
    
     n_elements = len(alpha) - 1
    
     # escludo l'ultimo valore di alpha che corrisponde a (f0 - K -alpha[i])
     alpha = alpha[:n_elements]
     sigma = sigma[:n_elements]
        
     #generazione di random variables indipendenti
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



'questa matrice mi serve per applicare la correzione tra gli elementi del black basket'


q99= spss.norm.ppf(0.99)
mc_forward_payoff, mc_stdev = BlackBasketPayoffMC(f0, strike, alpha, sigma, rho_matrix, T0 , n_sample = 1000000, mc_error = True)
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


def calculate_Gamma(beta, var_cov_matrix,expiry):
    
    beta = np.array(beta).reshape((-1, 1))  # Vettore colonna
    # Calcolo di gamma utilizzando la matrice di varianza-covarianza
    gamma = np.dot(var_cov_matrix, beta) * expiry

    return gamma.flatten()
    

def BlackBasketApprossimativePayoff(f0, strike, alpha, sigma, theta, var_cov_matrix, expiry):
    
    n_elements = len(alpha)
    
    B, total_stdev = calculate_B(strike, f0, alpha, sigma, var_cov_matrix, expiry)
    beta = [alpha[i] / total_stdev for i in range(n_elements)]
    gamma = calculate_Gamma(beta, var_cov_matrix, expiry)
    gamma = gamma.flatten()
    
    cdf_values = [norm.cdf(gamma[i] - B) for i in range(n_elements)]

    payoff = np.dot(alpha, cdf_values)
   
    return payoff

def BlackBasketObjectiveFunctionTheta(theta11, f0, strike, alpha, sigma, expiry, midcurve_market_price):
    theta = [theta11, 0.0, 0.0, 0.0]
    return (BlackBasketApprossimativePayoff(f0, strike, alpha, sigma, theta, expiry) - midcurve_market_price)**2


analytical_forward_payoff = BlackBasketApprossimativePayoff(f0, strike, alpha, sigma, theta, var_cov_matrix, T0)

if isinstance(analytical_forward_payoff, np.ndarray):
    analytical_forward_payoff = analytical_forward_payoff.item()
    
analytical_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, analytical_forward_payoff, discount=1) * 1e4

    
print(f"analytical payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{analytical_forward_payoff*1e4:.2f}bps")
print(f"analytical ivol:\t\t\t{analytical_ivol:.2f}bps")
print()



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





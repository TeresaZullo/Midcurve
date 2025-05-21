'pricer swaption con 4 elementi'

import numpy as np
import scipy.stats as spss
from scipy.stats import multivariate_normal
import scipy.optimize as opt
from scipy.optimize import minimize
import seaborn as sns
import QuantLib as ql
from QuantLib import *
import xlwings as xw
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy.stats import norm


percorso = r'D:\anaconda\Midcurve\DATI_TESI_pc_teresa.xlsm'
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

expiry = 5  # expiry swaption
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

def BlackBasketPayoffMC(f0, strike, alpha, sigma, rho_matrix, expiry, n_sample=1000000, seed=42, mc_error=True):
    
    n_elements = len(alpha)

    mean = np.zeros(n_elements)
    eps = spss.multivariate_normal.rvs(mean=mean, cov= rho_matrix, size=n_sample, random_state=seed)

     
    # Drift e diffusione per ogni componente del basket
    drift = -0.5 * np.array(sigma) ** 2 * expiry
    diff = np.array([sigma[i] * np.sqrt(expiry) * eps[:, i] for i in range(n_elements)])
    
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
payoff_1, stdev_1 = BlackBasketPayoffMC(f0_1, strike, alpha_1, sigma, rho_matrix_1, expiry, n_sample=n_sample, seed = 42, mc_error=True)
    
#per swaption 2
payoff_2, stdev_2 = BlackBasketPayoffMC(f0_2, strike, alpha_2, eta, rho_matrix_2, expiry, n_sample=n_sample, seed = 42, mc_error=True)

#Calcolo degli intervalli di confidenza
q99 = spss.norm.ppf(0.99)
#per swaption 1
mc_payoff_ub_1 = payoff_1 + q99 * stdev_1
mc_payoff_lb_1 = payoff_1 - q99 * stdev_1

#Calcolo delle volatilità implicite con Bachelier
mc_ivol_ub_1 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_1, expiry, mc_payoff_ub_1, discount=1) * 1e4
mc_ivol_lb_1 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_1, expiry, mc_payoff_lb_1, discount=1) * 1e4
mc_ivol_1 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_1, expiry, payoff_1, discount=1) * 1e4
    
#per swaption 2 
mc_payoff_ub_2 = payoff_2 + q99 * stdev_2
mc_payoff_lb_2 = payoff_2 - q99 * stdev_2

# Calcolo delle volatilità implicite con Bachelier
mc_ivol_ub_2 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_2, expiry, mc_payoff_ub_2, discount=1) * 1e4
mc_ivol_lb_2 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_2, expiry, mc_payoff_lb_2, discount=1) * 1e4
mc_ivol_2 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_2, expiry , payoff_2, discount=1) * 1e4

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

B_1, total_stdev_1 = calculate_B(strike, f0_1, alpha_1, sigma, var_cov_matrix_1, expiry)
B_2, total_stdev_2 = calculate_B(strike, f0_2, alpha_2, eta, var_cov_matrix_2, expiry)


#Creo B per la midcurve swaption 
def calculate_B_MC(R_mc, strike, alpha, sigma, var_cov_matrix, expiry):
    
    
    n_elements = len(alpha)
    
    # Converte alpha in un array NumPy e lo trasforma in un vettore colonna
    alpha = np.array(alpha).reshape((n_elements, 1)) 
    alpha_T = alpha.T  
    
    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
    var = [stdev[i]*stdev[i] for i in range(n_elements)]
    numerator = strike - R_mc + 0.5 * sum(alpha[i] * var[i] for i in range(n_elements))
        
    denominator_B_MC = np.dot(np.dot(alpha_T,var_cov_matrix), alpha)
    denominator_B_MC = denominator_B_MC.item()
            
    total_stdev = math.sqrt(expiry) * math.sqrt(denominator_B_MC)
     
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
   
    return payoff

analytical_forward_payoff_1 = BlackBasketApprossimativePayoff(f0_1, strike, alpha_1,sigma, theta, var_cov_matrix_1, expiry)
analytical_forward_payoff_2 = BlackBasketApprossimativePayoff(f0_2, strike, alpha_2, eta, theta, var_cov_matrix_2, expiry)

# Verifica e conversione in scalare per payoff_1
if isinstance(analytical_forward_payoff_1, np.ndarray):
    analytical_forward_payoff_1 = analytical_forward_payoff_1.item()

# Verifica e conversione in scalare per payoff_2
if isinstance(analytical_forward_payoff_2, np.ndarray):
    analytical_forward_payoff_2 = analytical_forward_payoff_2.item()

    
analytical_ivol_1 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_1, expiry, analytical_forward_payoff_1, discount=1) * 1e4
analytical_ivol_2 = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0_2, expiry, analytical_forward_payoff_2, discount=1) * 1e4
    
print("Swaption 1:")
print(f"Analytical Payoff ({expiry}y @ {strike*100:.2f}%):\t{analytical_forward_payoff_1*1e4:.2f}bps")
print(f"Analytical Implied Volatility:\t\t\t{analytical_ivol_1:.2f}bps")

print("\nSwaption 2:")
print(f"Analytical Payoff ({expiry}y @ {strike*100:.2f}%):\t{analytical_forward_payoff_2*1e4:.2f}bps")
print(f"Analytical Implied Volatility:\t\t\t{analytical_ivol_2:.2f}bps")


def BlackBasketApprossimativePayoff_Adj(f0, strike, alpha, sigma, theta, var_cov_matrix, expiry):
    B, total_stdev = calculate_B(strike, f0, alpha, sigma, var_cov_matrix, expiry)
    beta = np.array(alpha) / total_stdev
    gamma = calculate_Gamma(beta, var_cov_matrix, expiry)

    cdf_values = [norm.cdf(gamma[i] - B) for i in range(len(alpha))]
    payoff = np.dot(alpha, cdf_values) + (f0 - strike - sum(alpha)) * norm.cdf(-B)

    return payoff    

def BlackBasketApprossimativePayoffMidCurve(R_mc_0, strike, alpha_1, alpha_2, sigma_calibrated_1, sigma_calibrated_2, theta, expiry, annuity_1, annuity_2):
    
    annuity_MC = vols_sheet.range('K249').value
    
    delta_1 = -annuity_1 / annuity_MC
    delta_2 = annuity_2 / annuity_MC

    alpha_1_adj = [a * delta_1 for a in alpha_1]
    alpha_2_adj = [a * delta_2 for a in alpha_2]
    
    alpha_3 = (R_mc_0 - strike - delta_1 * sum(alpha_1) - delta_2 * sum(alpha_2))
    
    alpha_total = np.array(alpha_1_adj + alpha_2_adj + [alpha_3])
    sigma_total = list(sigma_calibrated_1) + list(sigma_calibrated_2) + [0]
    
    var_cov_matrix_total, _ = cov_matrix(sigma_total, theta)
    R_mc_0 = R_mc_0
    f0_adj = strike

    B, total_stdev = calculate_B_MC(R_mc_0, f0_adj, alpha_total, sigma_total, var_cov_matrix_total, expiry)
    beta = np.array(alpha_total) / total_stdev
    gamma = calculate_Gamma(beta, var_cov_matrix_total, expiry)

    cdf_values = norm.cdf(gamma - B)
    payoff = np.dot(alpha_total, cdf_values)

    return payoff*annuity_MC

def objective_function_swaption_1(params, market_swaption_1):
    alpha_1 = params[:2]  
    sigma = params[2:4] 
    var_cov_matrix = build_var_matrix(sigma)
    
    errors = []
    for i, strike in enumerate(market_swaption_1['strike']):
        model_price = BlackBasketApprossimativePayoff(market_swaption_1['f0'], market_swaption_1['strike'][i], alpha_1, sigma, theta, var_cov_matrix, market_swaption_1['expiry']).item()

        # print(f"Strike: {strike}, Model Price: {model_price}")
        
        error = (model_price - market_swaption_1['market_prices'][i]) / (market_swaption_1['black_vegas'][i] * market_swaption_1["black_vols"][1])
        errors.append(error ** 2)
    return np.sqrt(np.sum(errors))

def objective_function_swaption_2(params, market_swaption_2):
    alpha_2 = params[:2] 
    eta = params[2:4]  
    
    var_cov_matrix = build_var_matrix(eta)
    
    errors = []
    for i, strike in enumerate(market_swaption_2['strike']):
        model_price = BlackBasketApprossimativePayoff(market_swaption_2['f0'], market_swaption_2['strike'][i], alpha_2, eta, theta, var_cov_matrix, market_swaption_2['expiry']).item()
        
        # print(f"Strike: {strike}, Model Price: {model_price}")

        error = (model_price - market_swaption_2['market_prices'][i]) / (market_swaption_2['black_vegas'][i] * market_swaption_2["black_vols"][1])
        errors.append(error ** 2)
    return np.sqrt(np.sum(errors))

def calibrate_black_basket_swaption_1(market_swaption_1, objective_function_swaption_1):
    x_0 = [0.008813103, -0.006769068, 0.656223135, 0.764578644]
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0.0,np.inf), (0.0,np.inf)]
    
    result = opt.minimize(objective_function_swaption_1, x_0, args=(market_swaption_1), bounds=bounds, method='SLSQP', options={'disp': True, 'maxiter': 450})
    
    if result.success:
        return result.x[:2], result.x[2:4]
    else:
        return None

def calibrate_black_basket_swaption_2(market_swaption_2, objective_function_swaption_2):
    x_0 = [0.008944806,	-0.00623156, 0.648173091, 0.789885495]
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0.0,np.inf), (0.0,np.inf)]
    
    result = opt.minimize(objective_function_swaption_2, x_0, args=(market_swaption_2), bounds=bounds, method='SLSQP', options={'disp': True, 'maxiter': 450})
    
    if result.success:
        return result.x[:2], result.x[2:4]
    else:
        return None


def objective_function_ivol_1(params, market_swaption_1):  
    alpha_1 = params[:2] 
    sigma = params[2:4] 
        
    var_cov_matrix = build_var_matrix(sigma)
                                              
    errors = []
    for i, strike in enumerate(market_swaption_1['strike']):
        model_price = BlackBasketApprossimativePayoff(market_swaption_1['f0'], strike, alpha_1, sigma, theta, var_cov_matrix, market_swaption_1['expiry']).item()
                
        model_ivol = ql.blackFormulaImpliedStdDev(ql.Option.Call, strike, market_swaption_1['f0'], model_price,1, 0.03)
        market_ivol = market_swaption_1['black_vols'][i]
                
        error = (model_ivol - market_ivol)
        errors.append(error ** 2)
                
    return np.sqrt(np.sum(errors))
            
def calibrate_black_basket_swaption_ivol_1(market_swaption_1, objective_function_ivol_1,initial_params_1):       
    x_0 = list(initial_params_1[0]) + list(initial_params_1[1]) 
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0.0, np.inf), (0.0, np.inf)]
        
    result = opt.minimize(objective_function_ivol_1, x_0, args=(market_swaption_1), bounds=bounds, method='SLSQP', options={'disp': True})
        
    if result.success:
        return result.x[:2], result.x[2:4]
    else:
        return None 


def objective_function_ivol_2(params, market_swaption_2):
    alpha_2 = params[:2] 
    eta = params[2:4] 
        
    var_cov_matrix = build_var_matrix(eta)
    errors = []
        
    for i, strike in enumerate(market_swaption_2['strike']):
        model_price = BlackBasketApprossimativePayoff(market_swaption_2['f0'], strike, alpha_2, eta, theta, var_cov_matrix, market_swaption_2['expiry']).item()
            
        model_ivol = ql.blackFormulaImpliedStdDev(ql.Option.Call, strike, market_swaption_2['f0'], model_price, 1, .03)
        market_ivol = market_swaption_2['black_vols'][i]
                
        error = (model_ivol - market_ivol)
        errors.append(error ** 2)
            
    return np.sqrt(np.sum(errors))

def calibrate_black_basket_swaption_ivol_2(market_swaption_2, objective_function_ivol_2, initial_params_2):
    x_0 = list(initial_params_2[0]) + list(initial_params_2[1]) 
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0.0, np.inf), (0.0, np.inf)]
         
    result = opt.minimize(objective_function_ivol_2, x_0, args=(market_swaption_2), bounds=bounds, method='SLSQP', options={'disp': True})
         
    if result.success:
        return result.x[:2], result.x[2:4]
    else:
        return None 


def objective_function_swaption_sigma_1(params, market_swaption_1, alpha_fixed_list, f0_fixed_list, strike_adj):
    sigma = params[:2]  
    var_cov_matrix = build_var_matrix(sigma)

    errors = []
    for i, strike in enumerate(strike_adj):  
        model_price = BlackBasketApprossimativePayoff_Adj(f0_fixed_list[i], strike, alpha_fixed_list[i], sigma, theta, var_cov_matrix, market_swaption_1['expiry']).item()
        error = (model_price - market_swaption_1['market_prices'][i]) / (market_swaption_1['black_vegas'][i] * market_swaption_1["black_vols"][1])
        errors.append(error ** 2)

    return np.sqrt(np.sum(errors))

def objective_function_swaption_sigma_2(params, market_swaption_2, alpha_fixed_list, f0_fixed_list, strike_adj):
    eta = params[:2]
    var_cov_matrix = build_var_matrix(eta)

    errors = []
    for i, strike in enumerate(strike_adj):  
        model_price = BlackBasketApprossimativePayoff_Adj(f0_fixed_list[i], strike, alpha_fixed_list[i],eta, theta, var_cov_matrix, market_swaption_2['expiry']).item()
        error = (model_price - market_swaption_2['market_prices'][i]) / (market_swaption_2['black_vegas'][i] * market_swaption_2["black_vols"][1])
        errors.append(error ** 2)

    return np.sqrt(np.sum(errors))


def calibrate_black_basket_swaption_sigma_1(market_swaption, alpha_fixed_list, f0_fixed_list, strike_adj):
    x_0 = [0.656223135,	0.764578644]
    bounds = [(0.0, np.inf), (0.0, np.inf)] 
    
    result = opt.minimize(objective_function_swaption_sigma_1, x_0, args=(market_swaption, alpha_fixed_list, f0_fixed_list, strike_adj), 
                          bounds=bounds, method='SLSQP', options={'disp': True, 'maxiter': 450})
    
    if result.success:
        return result.x  
    else:
        return None


def calibrate_black_basket_swaption_sigma_2(market_swaption, alpha_fixed_list, f0_fixed_list, strike_adj):
    x_0 = [0.648173091, 0.789885495]
    bounds = [(0.0, np.inf), (0.0, np.inf)]  
    
    result = opt.minimize(objective_function_swaption_sigma_2, x_0, args=(market_swaption, alpha_fixed_list, f0_fixed_list, strike_adj), 
                          bounds=bounds, method='SLSQP', options={'disp': True, 'maxiter': 450})
    
    if result.success:
        return result.x 
    else:
        return None
    
def objective_function_theta(theta_param, market_price, alpha_1, alpha_2, sigma_calibrated_1, sigma_calibrated_2, expiry, annuity_1, annuity_2, annuity_MC):
    theta = [theta_param[0], 0.0, 0.0, 0.0]  
    R_mc = vols_sheet.range('J257').value                     
    strike= vols_sheet.range('J258').value
    
    model_price = BlackBasketApprossimativePayoffMidCurve(R_mc, strike, alpha_1, alpha_2, sigma_calibrated_1, sigma_calibrated_2, theta, expiry, annuity_1, annuity_2)
    error = (model_price - market_price)/market_price

    return np.sqrt(error**2)

def calibrate_theta(market_price, alpha_1, alpha_2, sigma_calibrated_1, sigma_calibrated_2, expiry, annuity_1, annuity_2):
    annuity_MC = vols_sheet.range('K249').value
    theta_initial_guess = [-0.5] 
    bounds = [(-np.inf, np.inf)]
    result = opt.minimize(
        objective_function_theta,
        theta_initial_guess,
        args=(market_price, alpha_1, alpha_2, sigma_calibrated_1, sigma_calibrated_2, expiry, annuity_1, annuity_2, annuity_MC),
        bounds=bounds,
        method='SLSQP',
        options={'disp': True, 'maxiter': 300}
    )

    if result.success:
        calibrated_theta = [result.x[0], 0.0, 0.0, 0.0]
        return calibrated_theta
    else:
        raise ValueError("Calibrazione theta non riuscita")
        
        
def plot_ivol_smile(strike_spread_1, model_ivols_1, market_ivols_1, market_swaption_1,
                    strike_spread_2, model_ivols_2, market_ivols_2, market_swaption_2):
    
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    expiry_1 = market_swaption_1["expiry"]
    tenor_1 = market_swaption_1["tenor"]

    expiry_2 = market_swaption_2["expiry"]
    tenor_2 = market_swaption_2["tenor"]
    
    title_1 = f"Swaption {expiry_1}Y{tenor_1}Y"
    title_2 = f"Swaption {expiry_2}Y{tenor_2}Y"
    
    sns.lineplot(x=strike_spread_1, y=model_ivols_1, label="Model", color="blue", ax=axes[0])
    sns.scatterplot(x=strike_spread_1, y=market_ivols_1, label="Market", color="orange", ax=axes[0])

    axes[0].set_title(title_1)
    axes[0].set_xlabel("Strike (%)")
    axes[0].set_ylabel("Implied Volatility (bps)")
    axes[0].set_xticks(np.arange(-2,2.5,0.5))  
    axes[0].legend()
    axes[0].grid(True)

    
    sns.lineplot(x=strike_spread_2, y=model_ivols_2, label="Model", color="blue", ax=axes[1])
    sns.scatterplot(x=strike_spread_2, y=market_ivols_2, label="Market", color="orange", ax=axes[1])

    axes[1].set_title(title_2)
    axes[1].set_xlabel("Strike Spread (%)")
    axes[1].set_ylabel("Implied Volatility (bps)")
    axes[1].set_xticks(np.arange(-2,2.5,0.5)) 
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


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

            if callable(self.rates[i]):
                R_value = self.rates[i](self.T0)
            else:
                R_value = self.rates[i]

            annuity = delta_T - 0.5 * R_value * (delta_T ** 2)
            annuities.append(annuity)

        return annuities

class MeasureApproximation:
    def __init__(self, annuities, maturities, T0, rates):

        self.annuities = annuities
        self.maturities = maturities
        self.expiry = expiry
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

                delta_T_squared = (T_i - self.expiry)**2 - (T_j - self.expiry)**2
                lambda_i = 0.5 * (delta_T_squared / denom + (T_i - self.expiry)**2 / A_i_0)
                lambda_j = 0.5 * (delta_T_squared / denom + (T_j - self.expiry)**2 / A_j_0)
                lambdas.append((lambda_i,lambda_j))

        return lambdas

# def R_MC_0(T0, f0_1_adj, f0_2_adj, strike_1_adj, strike_2_adj, 
#                                   alpha_1_adj_list, alpha_2_adj_list, 
#                                   sigma_calibrated_1, sigma_calibrated_2, 
#                                   theta, var_cov_matrix_1, var_cov_matrix_2, expiry, annuity_rates, annuity_MC):    
#     R_mc_0_list = []
    
#     annuities = annuity_rates.calculate_annuity()
    
#     annuity_1 = annuities[0]
#     annuity_2 = annuities[1]
    
#     for i in range(len(strike_1_adj)):  
#         R1_hat = BlackBasketApprossimativePayoff_Adj(f0_1_adj[i], strike_1_adj[i], alpha_1_adj_list[i], sigma_calibrated_1, theta, var_cov_matrix_1, expiry).item()
#         R2_hat = BlackBasketApprossimativePayoff_Adj(f0_2_adj[i], strike_2_adj[i], alpha_2_adj_list[i], sigma_calibrated_2, theta, var_cov_matrix_2, expiry).item()
#         R_mc_0 = (R2_hat * annuity_2 / annuity_MC) - (R1_hat * annuity_1 / annuity_MC)
#         R_mc_0_list.append(R_mc_0) 

#     return R_mc_0_list


def R_MC_0(vols_sheet, annuity_rates):
    annuity_MC = vols_sheet.range('K249').value
    R1_hat = vols_sheet.range('J255').value
    R2_hat = vols_sheet.range('J256').value

    annuities = annuity_rates.calculate_annuity()
    annuity_1 = annuities[0]
    annuity_2 = annuities[1]

    R_mc_0 = (R2_hat * annuity_2 - R1_hat * annuity_1) / annuity_MC

    return R_mc_0
   
def extract_numeric_value(cell_value):
    # Estrae solo la parte numerica da un valore letto da Excel 
    if isinstance(cell_value, str): 
        return int(''.join(filter(str.isdigit, cell_value))) 
    return int(cell_value) 


def plot_model_price_vs_theta(market_price, alpha_1, alpha_2, sigma_calibrated_1, sigma_calibrated_2,
                                  expiry, annuity_1, annuity_2):
    theta_range = np.linspace(-4 * np.pi, 4 * np.pi, 500)
    model_prices = []

    for theta_val in theta_range:
        theta_test = [theta_val, 0.0, 0.0, 0.0]

        model_price = BlackBasketApprossimativePayoffMidCurve(
            R_mc_0= vols_sheet.range('J257').value,  
            strike= vols_sheet.range('J258').value,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            sigma_calibrated_1=sigma_calibrated_1,
            sigma_calibrated_2=sigma_calibrated_2,
            theta=theta_test,
            expiry=expiry,
            annuity_1=annuity_1,
            annuity_2=annuity_2,
        )

        model_prices.append(model_price)

    plt.figure(figsize=(10, 6))
    plt.plot(theta_range, model_prices, label="Model Price", color='blue')
    plt.axhline(y=market_price, color='red', linestyle='--', label="Market Price")
    plt.title("Prezzo modello vs. θ₁₁")
    plt.xlabel("theta_11 (radianti)")
    plt.ylabel("Prezzo della Mid-Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.show()

def main():
    market_swaption_1 = {
        "f0": vols_sheet.range('J255').value,
        "strike": vols_sheet.range('M255:Y255').value,
        "market_prices": list(vols_sheet.range('AB255:AN255').value),
        "black_vols": list(vols_sheet.range('AQ255:BC255').value),
        "black_vegas": list(vols_sheet.range('BF255:BR255').value),
        "bachelier_vegas": list(vols_sheet.range('CJ255:CV255').value),
        "market_ivols": list(vols_sheet.range('BU255:CF255').value),
        "expiry": extract_numeric_value(vols_sheet.range('H255').value),
        "tenor": extract_numeric_value(vols_sheet.range('I255').value),
    }
    
    market_swaption_2 = {
        "f0": vols_sheet.range('J256').value,
        "strike": list(vols_sheet.range('M256:Y256').value),
        "market_prices": list(vols_sheet.range('AB256:AN256').value),
        "black_vols": list(vols_sheet.range('AQ256:BC256').value),
        "black_vegas":list(vols_sheet.range('BF256:BR256').value),
        "bachelier_vegas": list(vols_sheet.range('CJ256:CV256').value),
        "market_ivols": list(vols_sheet.range('BU256:CF256').value),
        "expiry": extract_numeric_value(vols_sheet.range('H256').value),
        "tenor": extract_numeric_value(vols_sheet.range('I256').value)
    }
    
    #prima calibrazione
    
    calibrated_params_1 = calibrate_black_basket_swaption_1(market_swaption_1, objective_function_swaption_1)
    calibrated_params_2 = calibrate_black_basket_swaption_2(market_swaption_2, objective_function_swaption_2)
    
    
    wb = xw.Book(percorso)
    params_sheet = wb.sheets["params"]

    expiry_1 = market_swaption_1["expiry"]
    tenor_1 = market_swaption_1["tenor"]

    expiry_2 = market_swaption_2["expiry"]
    tenor_2 = market_swaption_2["tenor"]

    params_sheet.range("K3").value = f"Parametri {expiry_1}Y{tenor_1}Y"
    params_sheet.range("L2").value = ["alpha1", "alpha2", "sigma1", "sigma2"]
    params_sheet.range("L3").value = list(calibrated_params_1[0]) + list(calibrated_params_1[1])


    params_sheet.range("K4").value = f"Parametri {expiry_2}Y{tenor_2}Y"
    params_sheet.range("L4").value = list(calibrated_params_2[0]) + list(calibrated_params_2[1])

    
    print("Parametri prima calibrazione per Swaption 1:", calibrated_params_1)
    print("Parametri prima calibrazione per Swaption 2:", calibrated_params_2)
    
    
    # seconda calibrazione
    
    # calibrated_ivol_params_1 = calibrate_black_basket_swaption_ivol_1(market_swaption_1, objective_function_ivol_1, calibrated_params_1)
    # calibrated_ivol_params_2 = calibrate_black_basket_swaption_ivol_2(market_swaption_2, objective_function_ivol_2, calibrated_params_2)

    # print("Parametri seconda calibrazione per Swaption 1:", calibrated_ivol_params_1)
    # print("Parametri seconda calibrazione per Swaption 2:", calibrated_ivol_params_2)
    
    ## plot normal ivols - swaption 1 e 2
    alpha_1 = calibrated_params_1[0] 
    sigma = calibrated_params_1[1]  # Array che include 2 sigma per swaption
    var_cov_matrix_sigma = build_var_matrix(sigma)

    model_prices_1 = []
    model_ivols_1 = []
    market_ivols_1 = []
    
    for i, strike in enumerate(market_swaption_1['strike']):
        model_price = BlackBasketApprossimativePayoff(market_swaption_1['f0'], strike, alpha_1, sigma, theta, var_cov_matrix_sigma, market_swaption_1['expiry']).item()
        model_prices_1.append(model_price)

        model_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, market_swaption_1['f0'], market_swaption_1["expiry"], model_price, discount=1) * 1e4
        model_ivols_1.append(model_ivol)

        market_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, market_swaption_1['f0'], market_swaption_1["expiry"], market_swaption_1['market_prices'][i], discount=1) * 1e4
        market_ivols_1.append(market_ivol)

    alpha_2 = calibrated_params_2[0] 
    eta = calibrated_params_2[1] 
    var_cov_matrix_eta = build_var_matrix(eta)

    model_prices_2 = []
    model_ivols_2 = []
    market_ivols_2 = []
    
    for i, strike in enumerate(market_swaption_2['strike']):
        model_price = BlackBasketApprossimativePayoff(market_swaption_2['f0'], strike, alpha_2, eta, theta, var_cov_matrix_eta, market_swaption_2['expiry']).item()
        model_prices_2.append(model_price)

        model_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, market_swaption_2['f0'], market_swaption_2["expiry"], model_price, discount=1)* 1e4 
        model_ivols_2.append(model_ivol)

        market_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, market_swaption_2['f0'], market_swaption_2["expiry"], market_swaption_2['market_prices'][i], discount=1)* 1e4
        market_ivols_2.append(market_ivol)

        pd.DataFrame(
            [model_ivols_1,
             market_ivols_1,
             model_ivols_2,
             market_ivols_2],
            index=["model_1", "market_1", "model_2", "market_2"]
        ).to_csv(r'D:\anaconda\Midcurve\output_calibration.csv')

    strike_spread_1 = np.linspace(-0.75, 0.75, len(market_swaption_1['strike'])) 
    strike_spread_2 = np.linspace(-0.75, 0.75, len(market_swaption_2['strike']))  
    
    plot_ivol_smile(strike_spread_1, model_ivols_1, market_ivols_1, market_swaption_1, 
                    strike_spread_2, model_ivols_2, market_ivols_2, market_swaption_2)



    rates= [market_swaption_1["f0"], market_swaption_2["f0"] ]
    
    maturities = [market_swaption_1["expiry"] + market_swaption_1["tenor"], 
              market_swaption_2["expiry"] + market_swaption_2["tenor"]]
    
    T0 = market_swaption_1["expiry"]

    annuity_rates = AnnuityApproximation(T0=T0, maturities=maturities, rates=rates)
    annuities = annuity_rates.calculate_annuity()
    # mid_curve_annuity = annuity_rates.calculate_mid_curve_annuity(annuities)
    
    measure_approx= MeasureApproximation(annuities=annuities, maturities=maturities, T0=T0, rates=rates)

    # print(f"🔹 Annuities: {annuities}")
    # print(f"🔹 Mid-Curve Annuity: {mid_curve_annuity:.6f}")
    # print(f"🔹 λ per Swaption 1 ({market_swaption_1['expiry']}Y{market_swaption_1['tenor']}Y): {lambdas[0][0]:.6f}")
    # print(f"🔹 λ per Swaption 2 ({market_swaption_2['expiry']}Y{market_swaption_2['tenor']}Y): {lambdas[0][1]:.6f}")
    
    lambdas_1 = measure_approx.calculate_lambda()[0][0]  
    lambdas_2 = measure_approx.calculate_lambda()[0][1]    
    
    lambdas_k_1 = [1 - lambdas_1 * K for K in market_swaption_1["strike"]]
    lambdas_k_2 = [1 - lambdas_2 * K for K in market_swaption_2["strike"]]
    
    f0_1_adj = [market_swaption_1["f0"] * lambda_k for lambda_k in lambdas_k_1]
    alpha_1_adj =[[a * lambda_k for a in calibrated_params_1[0]] for lambda_k in lambdas_k_1] 
    strike_1_adj =[K * (1 -lambdas_1*market_swaption_1["f0"]) for K in market_swaption_1["strike"]]    
    
    f0_2_adj = [market_swaption_2["f0"] * lambda_k for lambda_k in lambdas_k_2]
    alpha_2_adj = [[a * lambda_k for a in calibrated_params_2[0]] for lambda_k in lambdas_k_2]
    strike_2_adj =[K * (1 -lambdas_2* market_swaption_2["f0"] ) for K in market_swaption_2["strike"]] 
    
    sigma_calibrated_1 = calibrate_black_basket_swaption_sigma_1(market_swaption_1, alpha_1_adj, f0_1_adj, strike_1_adj)
    sigma_calibrated_2 = calibrate_black_basket_swaption_sigma_2(market_swaption_2, alpha_2_adj, f0_2_adj, strike_2_adj)
    
    print(f"✅ Nuovi sigma calibrati per Swaption 1: {sigma_calibrated_1}")
    print(f"✅ Nuovi sigma calibrati per Swaption 2: {sigma_calibrated_2}")
 
#plot seconda calibrazione
    
    alpha_1_adj_list = []
    alpha_2_adj_list = []
    strike_1_adj_list = []
    strike_2_adj_list = []

    for K in market_swaption_1["strike"]:
        lambda_k_1 = 1 - lambdas_1 * K  
        alpha_1_adj_k =[alpha_1[0] * lambda_k_1, alpha_1[1] * lambda_k_1]  
        alpha_1_adj_list.append(tuple(alpha_1_adj_k)) 
        strike_1_adj_list.append(K * lambda_k_1) 

    for K in market_swaption_2["strike"]:
        lambda_k_2 = 1 - lambdas_2 * K
        alpha_2_adj_k =[alpha_2[0] * lambda_k_2, alpha_2[1] * lambda_k_2]
        alpha_2_adj_list.append(tuple(alpha_2_adj_k)) 
        strike_2_adj_list.append(K * lambda_k_2)

    
    model_prices_1_adj = []
    model_ivols_1_adj = []
    
    for i, (K_adj, alpha_1_k) in enumerate(zip(strike_1_adj_list, alpha_1_adj_list)): 
        model_price = BlackBasketApprossimativePayoff_Adj(
            f0_1_adj[i], K_adj,alpha_1_k, sigma_calibrated_1, theta,
            build_var_matrix(sigma_calibrated_1), market_swaption_1['expiry']
            ).item()
        
        model_prices_1_adj.append(model_price)

        model_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, K_adj, f0_1_adj[i], market_swaption_1["expiry"], model_price, discount=1) * 1e4
        model_ivols_1_adj.append(model_ivol)
        
        model_prices_2_adj = []
        model_ivols_2_adj = []
        
    for i, (K_adj, alpha_2_k) in enumerate(zip(strike_2_adj_list, alpha_2_adj_list)):
        alpha_2_k = np.array(alpha_2_k) 
        model_price = BlackBasketApprossimativePayoff_Adj(
            f0_2_adj[i], K_adj, alpha_2_k, sigma_calibrated_2, theta,
            build_var_matrix(sigma_calibrated_2), market_swaption_2['expiry']
            ).item()
    
        model_prices_2_adj.append(model_price)
        
        model_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, K_adj, f0_2_adj[i], market_swaption_2["expiry"], model_price, discount=1) * 1e4
        model_ivols_2_adj.append(model_ivol)

    strike_spread_1_adj = np.linspace(-0.75, 0.75, len(market_swaption_1['strike']))
    strike_spread_2_adj = np.linspace(-0.75, 0.75, len(market_swaption_2['strike']))

    plot_ivol_smile(strike_spread_1_adj, model_ivols_1_adj, market_ivols_1, market_swaption_1,
                strike_spread_2_adj, model_ivols_2_adj, market_ivols_2, market_swaption_2)   
    
    
    market_price_midcurve = vols_sheet.range('J249').value
    annuity_1 = annuities[0]
    annuity_2 = annuities[1]

    calibrated_theta = calibrate_theta(
    market_price_midcurve,
    alpha_1,
    alpha_2,
    sigma_calibrated_1,
    sigma_calibrated_2,
    expiry,
    annuity_1,
    annuity_2,
    )

    print(f"Theta calibrato: {calibrated_theta}")
    
    R_mc_0 = R_MC_0(vols_sheet, annuity_rates)
    
    output_start_cell = 'J257'
    vols_sheet.range(output_start_cell).value = R_mc_0
    
    Market_price = vols_sheet.range('J249').value

    Model_price = BlackBasketApprossimativePayoffMidCurve(
    R_mc_0 = vols_sheet.range('J257').value,                     
    strike= vols_sheet.range('J258').value,  
    alpha_1=alpha_1,
    alpha_2=alpha_2,
    sigma_calibrated_1=sigma_calibrated_1,
    sigma_calibrated_2=sigma_calibrated_2,
    theta=calibrated_theta,
    expiry=expiry,
    annuity_1=annuity_1,
    annuity_2=annuity_2,
    # annuity_MC= vols_sheet.range('K249').value
)
    
    print("\n📊 Confronto Prezzi Mid-Curve:")
    print(f"▪️ Prezzo di mercato  : {Market_price:.6f}")
    print(f"▪️ Prezzo da modello  : {Model_price:.6f}")
    print(f"▪️ Errore assoluto    : {abs(Model_price - Market_price):.6f}")
    print(f"▪️ Errore relativo (%) : {(Model_price - Market_price) / Market_price * 100:.2f}%")
    
    

        
    plot_model_price_vs_theta(market_price_midcurve, alpha_1, alpha_2, sigma_calibrated_1, sigma_calibrated_2, expiry, annuity_1, annuity_2)
        
            
if __name__ == "__main__":
    main()





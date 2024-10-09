import numpy as np
import scipy.stats as spss
import scipy.optimize as spopt
import seaborn as sns
import QuantLib as ql
import xlwings as xw
import matplotlib.pyplot as plt
import math


'prima scrivo il pricing di una swaption con n elementi nel basket sotto la simulazione del montecarlo (quindi nx2 componenti)'

def BlackBasketPayoffMC(f0, strike, alpha, sigma, rho, T0, n_sample=1000000, seed=42, mc_error=False):
    
    n_elements = len(alpha)
    
    if len(sigma) != n_elements:
        raise ValueError("alhpa and sigma array must have same length")
        
    if len(rho) != n_elements or any(len(rho[i]) != n_elements for i in range(n_elements)):
        raise ValueError("Correlation matrix must be quadratic and must have same alpha and sigma's length")
        
    #generazione di random variables indipendenti
    eps = np.array([spss.norm.rvs(loc=0, scale=1, size=n_sample, random_state=seed+i) for i in range(n_elements)])
    
    
    #applico la Cholesky decomposition per creare eps correlati tra di loro (vedi appendice D)
    # creo una sqare root of a matrix p dove p = AA^T
    
    A = np.linalg.cholesky(rho)
    eps_corr= np.dot(A,eps)
    
    # Drift e diffusione per ogni componente del basket
    drift = [-0.5 * sigma[i] * sigma[i] * T0 for i in range(n_elements)]
    diff = [sigma[i] * math.sqrt(T0) * eps_corr[i] for i in range(n_elements)]
    
    basket_components = [alpha[i] * (np.exp(drift[i] + diff[i]) - 1) for i in range(n_elements)]
    
    # Somma di tutti i 4 componenti per formare il basket finale
    basket = sum(basket_components)
    
    Rt = f0 + basket
    
    # Payoff della swaption
    payoff_sample = np.clip(Rt - strike, 0, np.inf)
    payoff_mean = payoff_sample.mean()
    
    if mc_error:
        estimate_variance = payoff_sample.var(ddof=1) / n_sample
        mc_error_value = math.sqrt(estimate_variance)
        return payoff_mean, mc_error_value
    
    return payoff_mean


f0 = -0.0019  
strike = 0

# Quattro pesi e volatilità per ogni basket
alpha = [0.00624, -0.00441, 0.1, 0.1]  
sigma = [0.5132, 0.5132, 0.1 , 0.1]  

T0 = 1  # expiry swaption
T = 5 # tenor swap

n_sample = 1000000  # Numero di simulazioni

'questa matrice mi serve per applicare la correzione tra gli elementi del black basket'
#prima correliamo con 0.5 

rho = [
    [1, 0.5, 0.0, 0.0],  # Correlazione tra 1° e 2° elemento
    [0.5, 1, 0.0, 0.0],  # Correlazione tra 2° e 1° elemento
    [0.0, 0.0, 1, 0.0],  # Nessuna correlazione con altri
    [0.0, 0.0, 0.0, 1] 
]
  
q99= spss.norm.ppf(0.99)
mc_forward_payoff, mc_stdev = BlackBasketPayoffMC(f0, strike, alpha, sigma, rho, T0 , n_sample = 1000000, mc_error = True)
mc_payoff_ub = mc_forward_payoff + q99*mc_stdev    
mc_payoff_lb = mc_forward_payoff - q99*mc_stdev

mc_ivol_ub = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_payoff_ub, discount = 1) *1e4   
mc_ivol_lb = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_payoff_lb, discount = 1) *1e4
mc_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_forward_payoff, discount = 1) *1e4 

print(f"mc payoff ({T0}y{T}y @ {strike*100:.2f}%):\t\t{mc_forward_payoff*1e4:.2f}bps / [{mc_payoff_lb*1e4:.2f}, {mc_payoff_ub*1e4:.2f}]bps")
print(f"mc ivol:\t\t\t\t{mc_ivol:.2f}bps / [{mc_ivol_lb:.2f}, {mc_ivol_ub:.2f}]bps")
print()

# plot da rivedere 

def plot_volatility():
    
    strikes = np.arange(-0.015, 0.015 + 0.005, 0.005)
    black_vols = []


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


def BlackBasketApprossimativePayoff(f0, strike, alpha, sigma, rho, expiry):
    
    n_elements = len(alpha)
        
    if len(sigma) != n_elements:
        raise ValueError("alhpa and sigma array must have same length")
            
    if len(rho) != n_elements or any(len(rho[i]) != n_elements for i in range(n_elements)):
        raise ValueError("Correlation matrix must be quadratic and must has same alpha and sigma's length")
    
           
    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)]
    var = [stdev[i]*stdev[i] for i in range(n_elements)]

    
    numerator_B = strike - f0 + 1/2 * sum(alpha[i] * var[i] for i in range(n_elements))
    
    denominator_B = 0
    for i in range(n_elements):
        for j in range(n_elements):
               if i == j:
            # Termini diagonali: alpha_i^2 * sigma_i^2 * t
                  denominator_B += alpha[i]**2 * var[i] 
               else:
            # Termini fuori diagonale: 2 * alpha_i * alpha_j * sigma_i * sigma_j * rho[i,j] * t
                  denominator_B += 2 * alpha[i] * alpha[j] * sigma[i] * sigma[j] * rho[i][j] * expiry
        
    total_stdev = math.sqrt(denominator_B)
    
    B = numerator_B/total_stdev
       
    beta = [alpha[i]/total_stdev for i in range(n_elements)]
        
    gamma = []
    for i in range(n_elements):
        gamma_i = beta[i] * var[i]        
        for j in range(n_elements):
            if i != j:
                gamma_i += beta[j] * sigma[i] * sigma[j] * rho[i][j] * expiry
        
        gamma.append(gamma_i)  

    payoff = 0
    for i in range(n_elements):
        payoff += alpha[i] * spss.norm.cdf(gamma[i] - B)

    payoff += (f0 - strike - sum(alpha)) * spss.norm.cdf(-B)
     
    return payoff

analytical_forward_payoff = BlackBasketApprossimativePayoff(f0, strike, alpha, sigma, rho, T0)
analytical_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, analytical_forward_payoff, discount=1) * 1e4

print(f"analytical payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{analytical_forward_payoff*1e4:.2f}bps")
print(f"analytical ivol:\t\t\t{analytical_ivol:.2f}bps")
print()


'swaption price con 4 elementi con approssimazione del black basket con lagrange'

def _black_basket_option_price(f0, strike, alpha, sigma, rho, gammas, B, expiry):
    
    n_elements = len(alpha)
    
    if len(sigma) != n_elements:
        raise ValueError("alhpa and sigma array must have same length")
        
    if len(rho) != n_elements or any(len(rho[i]) != n_elements for i in range(n_elements)):
        raise ValueError("Correlation matrix must be quadratic and must has same alpha and sigma's length")
        
    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
    var = [stdev[i]*stdev[i] for i in range(n_elements)] 
    
    payoff = 0
    for i in range(n_elements):
        payoff += alpha[i]*spss.norm.cdf(gammas[i]-B)
    payoff += f0-strike-sum(alpha)*spss.norm.cdf(-B)
    
    return payoff


def calculate_B(strike, f0, alpha, sigma, rho, expiry):
    
    n_elements = len(alpha)
    
    
    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
    var = [stdev[i]*stdev[i] for i in range(n_elements)]
    
    numerator = (strike - f0)
    for i in range(n_elements):
        numerator += 0.5*(alpha[i] * var[i])
        
    denominator_B = 0
    for i in range(n_elements):
        denominator_B += alpha[i]**2 * var[i]
        for j in range(i + 1, n_elements):
            denominator_B += 2 * alpha[i] * alpha[j] * sigma[i] * sigma[j] * rho[i][j] * expiry
            
    total_stdev = math.sqrt(denominator_B)
     
    B = numerator / total_stdev
    
    return B, total_stdev


def BlackBasketApproximatePayoffMaximized(f0, strike, alpha, sigma, rho, expiry, convergence = False):
    
    n_elements = len(alpha)
    
    B, total_stdev = calculate_B(strike, f0, alpha, sigma, rho, expiry)
    
    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
    var = [stdev[i]*stdev[i] for i in range(n_elements)]
    
    
    beta = [alpha[i]/total_stdev for i in range(n_elements)]
            
    gammas = []
    for i in range(n_elements):
        gamma_i = beta[i] * var[i]        
        for j in range(n_elements):
            if i != j:
                gamma_i += beta[j] * sigma[i] * sigma[j] * rho[i][j] * expiry
            
        gammas.append(gamma_i) 

    'implementazione3 con minimizzazione con moltiplicatore di Lagrange'
    'calcolo con il segno meno perchè posso fare solo la minimizzazione, di minimizzo una funzione negativa'
    opt_price_func = lambda x: -_black_basket_option_price(f0, strike, alpha, sigma, rho, x[:n_elements], x[n_elements], expiry)
    opt_price_constr = lambda x: sum(x[i]**2/var[i] if var[i] > 0 else 0 for i in range(n_elements)) - 1.0

    minimization_constrains = ({"type": "eq", "fun": opt_price_constr}, )
    
    opt = spopt.minimize(fun = opt_price_func, x0 = gammas + [B], method= "SLSQP", constraints= minimization_constrains)
    
    if convergence:
        print(opt)
        
    gammas = opt.x[:n_elements]
    B = opt.x[n_elements]
    print(f"stdev: {stdev}")
    print(f"var: {var}")

    return _black_basket_option_price(f0, strike, alpha, sigma, rho, gammas, B, expiry)


maximization_forward_payoff = BlackBasketApproximatePayoffMaximized(f0, strike, alpha, sigma, rho, T0)
maximization_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, maximization_forward_payoff, discount=1) * 1e4 


print(f"maximization payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{maximization_forward_payoff*1e4:.2f}bps")
print(f"maximization ivol:\t\t\t{maximization_ivol:.2f}bps")

#proviamo a trovare gamma e B con il metodo dell'iterazione


    
def _B_root(B_candidate, a, g, m):
    
    n_elements = len(alpha)
    
    basket = 0
    fprime = 0
    
    for i in range(n_elements):
        basket_i = a[i]*math.exp(g[i]*B_candidate-0.5*g[i]*g[i])
        basket += basket_i 
        f = basket + m
        fprime += basket_i * g[i]
   
    return f, fprime


    
def calculate_B(strike, f0, alpha, sigma, rho, expiry):
     
      n_elements = len(alpha)
     
     
      stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)] 
      var = [stdev[i]*stdev[i] for i in range(n_elements)]
     
      numerator = (strike - f0)
      for i in range(n_elements):
          numerator += 0.5*(alpha[i] * var[i])
         
      denominator_B = 0
      for i in range(n_elements):
          denominator_B += alpha[i]**2 * var[i]
          for j in range(i + 1, n_elements):
              denominator_B += 2 * alpha[i] * alpha[j] * sigma[i] * sigma[j] * rho[i][j] * expiry
             
      total_stdev = math.sqrt(denominator_B)
      
      B = numerator / total_stdev
     
      return B, total_stdev   
 
def BlackBasketPayoffIterative(f0, k, alpha, sigma, rho, expiry, N=2, tolerance = 1e-8, convergence = False):

    n_elements = len(alpha)

    B, total_stdev = calculate_B(strike, f0, alpha, sigma, rho, expiry)

    stdev = [sigma[i]*math.sqrt(expiry) for i in range(n_elements)]     
    var = [stdev[i]*stdev[i] for i in range(n_elements)]


    beta = [alpha[i]/total_stdev for i in range(n_elements)]
        
    gammas = []
    for i in range(n_elements):
        gamma_i = beta[i] * var[i]        
        for j in range(n_elements):
            if i != j:
                gamma_i += beta[j] * sigma[i] * sigma[j] * rho[i][j] * expiry
        
        gammas.append(gamma_i) 


   # Inizializzazione del payoff
    p0 = -10000
    p1 = _black_basket_option_price(f0, k, alpha, sigma, rho, gammas, B, expiry)

   # Iterazione per il calcolo di gamma e B
    while abs(p1 - p0) > tolerance:
       p0 = p1

       # Trova i gamma ottimali per ogni elemento
       w = [alpha[i] * spss.norm.pdf(gammas[i] - B) for i in range(n_elements)]
       gamma_denom = math.sqrt(sum(w[i]**2 * var[i] for i in range(n_elements)))
       
       for i in range(n_elements):
           gammas[i] = w[i] * var[i] / gamma_denom

       # Ottimizzazione per trovare B utilizzando la funzione _B_root generalizzata
       opt = spopt.root_scalar(f=_B_root, method="newton", x0=B, fprime=True,
                               args=(alpha, gammas, f0 - k - sum(alpha)))
       B = opt.root

       if convergence:
           print(opt)
       
       p1 = _black_basket_option_price(f0, k, alpha, sigma, rho, gammas, B, expiry)
   
    return p1
    

iterative_forward_payoff = BlackBasketPayoffIterative(f0, strike, alpha, sigma, rho, T0, tolerance=1e-8)
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

maturities = [2, 5]  
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


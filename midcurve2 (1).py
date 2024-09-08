# -*- coding: utf-8 -*-'
"""
Created on Mon May 13 17:13:29 2024

@author: S541998
"""

import numpy as np
import scipy.stats as spss
import scipy.optimize as spopt
import seaborn as sns
import QuantLib as ql
import matplotlib.pyplot as plt
import math


# 'implementiamo black basket prima definendo l equazione 25 del paper' 
# 'un solo basket con due elementi'
#  def R(t,r0, alpha1, sigma1, alpha2, sigma2, w1, w2):
#      drift1 = -0.5*sigma1*sigma1*t
#      drift2 = -0.5*sigma2*sigma2*t
#      diff1 = sigma1*w1
#      diff2 = sigma2*w2
#      basket1 = np.exp(drift1 + diff1) -1
#      basket2 = np.exp(drift2 -diff2) -1 
#      return r0 + alpha1*basket1 + alpha2*basket2
    
    
#  ' genero due sets di random variable da una distribuzione normale con mean=0 e dev std=1'
#  eps1,eps2= spss.norm.rvs(loc=0,scale=1,size=(2,10000))
#  print(eps1,eps2)

'prendo i valori dei paramentri della swaption dal paper a pagina 8'

'1y2y'

# f0= -0.003
# s1= 0.4814
# s2= 0.4814
# a1= 0.00466
# a2= -0.00311
# T0=1
# T=2
# N = 1e7
# strike = 0

'1y5y'

f0= -0.0019
s1= 0.5132
s2= 0.5132
a1= 0.00624
a2= -0.00441
T0=1
T=5
N = 1e7
strike = 0

'1y10y'

# f0= 0.00022
# s1= 0.5246
# s2= 0.5246
# a1= 0.00787
# a2= -0.00714
# T0=1
# T=10
# N = 1e7
# strike = 0
# 'creo il sample di ogni R a partire da una lista di epsilon'
# 'prima inizializzo la lista vuota di Rsample e poi inserisco un ciclo for degli elementi'

# Rsample = [] 
# for e1,e2 in zip(eps1,eps2):
#     Rt= R(T0,f0,a1,sigma_1,a2,sigma_2,e1,e2)
#     Rsample.append(Rt)
    
# Rsample = np.array(Rsample)

# 'calcolo annuity'
# annuity = (T1-T0)-0.5*Rsample*((T1-T0)**2)
# annuity_mean= np.mean(annuity)
# print(annuity_mean)

# 'trovo il prezzo della swaption con un solo basket di due elementi'

# payoff = np.clip(Rsample - strike, 0, 10000).mean()
# result = annuity_mean*payoff*N
# print(result) 
# atm_price = np.clip(Rsample - strike, 0, 10000).mean()
# mc_stdev = np.clip(Rsample - strike, 0, 10000).std()
# print(atm_price,mc_stdev)

# black_vol= ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0,T0, atm_price, discount = 1)

# strikes = np.linspace(-0.01, 0.01, 50)
# black_vols = []

# for strike in strikes:
#     payoff = np.clip(Rsample - strike, 0, 10000).mean()
#     atm_price = np.clip(Rsample - strike, 0, 10000).mean()
#     black_vol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, atm_price, discount=1)
#     black_vols.append(black_vol)
    
# plt.figure(figsize=(10, 6))
# plt.plot(strikes, black_vols, label='Black Volatility')
# plt.xlabel('Strike')
# plt.ylabel('Black Volatility')
# plt.title('Black Volatility vs Strike')
# plt.legend()
# plt.grid(True)
# plt.show()

'implementazione1: senza minimizzazione'
'implementazione2:con minimizzazione della eq 40'
'implementazione3: con minimizzazione con moltiplicatore di Lagrange'




'implementazione1'
'a partire dall eq 38, la nostra approssimazione di gamme e beta come descritto a pag.17 nell eq 40 senza minimizzazione'  

def BlackBasketOptionIterative(f0,k,alpha1,aplha2,sigma1,sigma2,expiry,tolerance=1e-8):
    stdev1= sigma1*math.sqrt(expiry)
    stdev2= sigma2*math.sqrt(expiry)
    A1= alpha1*stdev1
    A2 = alpha2*stdev2
    
    def option_price(gamma1,gamma2,B):
        return A1*spss.norm.cdf(gamma1-B)+ A2*spss.norm.cdf(gamma2-B)

    'inizializzo B e i gamma'   

    'dall approssimazione svolta ottengo che B può essere definito come:'

    num_B= k - f0 + 0.5*(A1*stdev1*math.sqrt(expiry) + A2*stdev2*math.sqrt(expiry))
    denom_B= A1+A2
    B = num_B/denom_B
    gamma1 = (alpha1*sigma1*sigma1) /denom_B
    gamma2 = (alpha2*sigma2*sigma2) /denom_B  

    p0 = 10000
    p1 = option_price(gamma1, gamma2, B)

    while abs(p1-p0) > tolerance:
          p0 = p1
          'find optimal gamma'
          w1 = A1*spss.norm.pdf(gamma1 - B)
          w2 = A2*spss.norm.pdf(gamma2 - B)
          'gamma_denom = math.sqrt(w1*w1*stdev1 + w2*w2*stdev2)'
          gamma1 = w1*sigma2*sigma2/denom_B
          gamma2 = w2*sigma2*sigma2/denom_B
        
          'solve for B'
          
          def B_root(B_candidate, g1, g2, a1, a2, m):
              basket1 = a1*math.exp(g1*B_candidate-0.5*g1*g1)
              basket2 = a2*math.exp(g2*B_candidate-0.5*g2*g2)
              f = basket1 + basket2 + m
              fprime = g1*basket1 + g2*basket2
              return f, fprime
     
          opt = spopt.root_scalar(B_root, args=(gamma1, gamma2, A1, A2, f0 - K - alpha1 - alpha2), method="newton", x0=B, fprime=True)
          B = opt.root
          print(opt)
          print()
          p1 = option_price(gamma1, gamma2, B)
    return p1


'iterazione 2'

def _black_basket_option_price(a1,a2,g1,g2,B):
        return a1*spss.norm.cdf(g1-B)+ a2*spss.norm.cdf(g2-B)
    
def _B_root(B_candidate,a1,a2,g1,g2,m):
    basket1 = a1*math.exp(g1*B_candidate-0.5*g1*g1)
    basket2 = a2*math.exp(g2*B_candidate-0.5*g2*g2)
    f= basket1 + basket2 + m
    fprime = g1*basket1 + g2*basket2
    return f, fprime


def BlackBasketOptionIterative2(f0, k, alpha1, alpha2, sigma1, sigma2, expiry, tolerance =1e-8):
    stdev1= sigma1*math.sqrt(expiry)
    stdev2= sigma2*math.sqrt(expiry)
    A1= alpha1*stdev1
    A2 = alpha2*stdev2

    'inizializzo B e gamma'    
    num_B= k - f0 + 0.5*(A1*stdev1*math.sqrt(expiry) + A2*stdev2*math.sqrt(expiry))
    denom_B= A1+A2
    B = num_B/denom_B
    gamma1 = (alpha1*sigma1*sigma1) /denom_B
    gamma2 = (alpha2*sigma2*sigma2) /denom_B 



    p0 = 10000
    p1 = _black_basket_option_price(A1, A2, gamma1, gamma2, B)

    while abs(p1-p0) > tolerance:
          p0 = p1
          'find optimal gamma'
          w1 = A1*spss.norm.pdf(gamma1 - B)
          w2 = A2*spss.norm.pdf(gamma2 - B)
          'gamma_denom = math.sqrt(w1*w1*stdev1 + w2*w2*stdev2)'
          gamma1 = w1*sigma2*sigma2/denom_B
          gamma2 = w2*sigma2*sigma2/denom_B
        
          'solve for B'
         
          opt = spopt.root_scalar(f= _B_root, args=(alpha1, alpha2, gamma1, gamma2, f0 - K - alpha1 - alpha2), method="newton", x0=B, fprime=True)
          B = opt.root
          print(opt)
          print()
          p1 = _black_basket_option_price(alpha1, alpha2, gamma1, gamma2, B)
    return p1


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


def BlackBasketApprossimativePayoff(f0, k, alpha1, alpha2, sigma1, sigma2, expiry):
    stdev1 = sigma1*math.sqrt(expiry)
    stdev2 = sigma2*math.sqrt(expiry)
    var1 = stdev1*stdev1
    var2 = stdev2*stdev2
    boundary_constant = k - f0 + 0.5*(alpha1*var1 + alpha2*var2)
    total_stdev = math.sqrt(alpha1*alpha1*var1+alpha2*alpha2*var2)
    B = boundary_constant/total_stdev
    
    gamma1= alpha1*var1 / total_stdev
    gamma2= alpha2*var2 / total_stdev
    return alpha1*spss.norm.cdf(gamma1-B)+ alpha2*spss.norm.cdf(gamma2-B)


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

def BlackBasketPayoffIterative(f0, k, alpha1, alpha2, sigma1, sigma2, expiry, N=2, tolerance = 1e-8, convergence = False):
    stdev1 = sigma1*math.sqrt(expiry)
    stdev2 = sigma2*math.sqrt(expiry)
    var1 = stdev1*stdev1
    var2 = stdev2*stdev2

    'definiamo i limiti della funzione da massimizzare'
    boundary_constant = k - f0 + 0.5*(alpha1*var1 + alpha2*var2)
    total_stdev = math.sqrt(alpha1*alpha1*var1+alpha2*alpha2*var2)
    B = boundary_constant/total_stdev
    
    gamma1= alpha1*var1 / total_stdev
    gamma2= alpha2*var2 / total_stdev   
    
    p0 = -10000
    p1 = _black_basket_option_price(alpha1, alpha2, gamma1, gamma2, B)

    while abs(p1-p0) > tolerance:
        p0 = p1
        'trova i gamma ottimali e poi dopo averli trovati trovi B'
        w1 = alpha1 *spss.norm.pdf(gamma1 - B)
        w2 = alpha2 *spss.norm.pdf(gamma2 - B)
        
        gamma_denom = math.sqrt(w1*w1*stdev1*stdev1 + w2*w2*stdev2*stdev2)
        gamma1 = w1*stdev1*stdev1/gamma_denom
        gamma2 = w2*stdev2*stdev2/gamma_denom

        opt = spopt.root_scalar(f=_B_root, method="newton", x0=B, fprime=True,
                                   args=(alpha1, alpha2, gamma1, gamma2, f0 - k - alpha1 - alpha2))
        B = opt.root
        if convergence:
           print(opt)
           print()
           p1 = _black_basket_option_price(alpha1, alpha2, gamma1, gamma2, B)
    return p1    
     

    
q99= spss.norm.ppf(0.99)
mc_forward_payoff, mc_stdev = BlackBasketPayoffMC(f0, strike, a1, a2, s1, s2, T0, n_sample = 1000000, mc_error = True)
mc_payoff_ub = mc_forward_payoff + q99*mc_stdev    
mc_payoff_lb = mc_forward_payoff - q99*mc_stdev

mc_ivol_ub = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_payoff_ub, discount = 1) *1e4   
mc_ivol_lb = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_payoff_lb, discount = 1) *1e4
mc_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, mc_forward_payoff, discount = 1) *1e4    

analytical_forward_payoff = BlackBasketApprossimativePayoff(f0, strike, a1, a2, s1, s2, T0)
analytical_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, analytical_forward_payoff, discount=1) * 1e4

maximization_forward_payoff = BlackBasketApproximatePayoffMaximized(f0, strike, a1, a2, s1, s2, T0)
maximization_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, maximization_forward_payoff, discount=1) * 1e4 

iterative_forward_payoff = BlackBasketPayoffIterative(f0, strike, a1, a2, s1, s2, T0, tolerance=1e-8)
iterative_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, iterative_forward_payoff, discount=1) * 1e4

print(f"mc payoff ({T0}y{T}y @ {strike*100:.2f}%):\t\t{mc_forward_payoff*1e4:.2f}bps / [{mc_payoff_lb*1e4:.2f}, {mc_payoff_ub*1e4:.2f}]bps")
print(f"mc ivol:\t\t\t\t{mc_ivol:.2f}bps / [{mc_ivol_lb:.2f}, {mc_ivol_ub:.2f}]bps")
print()
print(f"analytical payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{analytical_forward_payoff*1e4:.2f}bps")
print(f"analytical ivol:\t\t\t{analytical_ivol:.2f}bps")
print()
print(f"maximization payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{maximization_forward_payoff*1e4:.2f}bps")
print(f"maximization ivol:\t\t\t{maximization_ivol:.2f}bps")
print()
print(f"iterative payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{iterative_forward_payoff*1e4:.2f}bps")
print(f"iterative ivol:\t\t\t\t{iterative_ivol:.2f}bps")


# def BlackBasketOption(alpha1,alpha2,sigma1,sigma2,beta1,beta2,B):
#     gamma1= sigma1*sigma1*beta1
#     gamma2= sigma2*sigma1*beta2


    
    # Ngamma1_B = norm.cdf(gamma1 -B)
    # Ngamma2_B = norm.cdf(gamma2 -B)
    
    # return alpha1*Ngamma1_B + alpha2*Ngamma2_B

# 'equazione 37'

# beta1^2*sigma1*sigma1 + beta2^2*sigma2*sigma2 =1

# 'per ottenere il vincolo di 1, b_n può essere definito nel seguente modo:'



# beta_n = alpha_n/(math.sqrt(alpha1**2*sigma1**2 + alpha2**2*sigma2**2))



# 'equazione 39'

# gamma1^2/(sigma1**2) + gamma2^2/(sigma2**2) = 1

# A_n = alpha_n * sigma_n* math.sqrt(t)

# gamma_n = beta_n*sigma_n*sigma_n = (alpha_n*sigma_n*sigma_n) / (math.sqrt(alpha1**2*sigma1**2 + alpha2**2*sigma2**2))




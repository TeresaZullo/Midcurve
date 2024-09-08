import numpy as np
import scipy.stats as spss
import scipy.optimize as spopt
import seaborn as sns
import QuantLib as ql
import matplotlib.pyplot as plt
import math

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
a2= 0.00441
T0=1
T=5
N = 1e7
strike = f0

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

    # plt.figure(figsize=(10, 6))
# plt.plot(strikes, black_vols, label='Black Volatility')
# plt.xlabel('Strike')
# plt.ylabel('Black Volatility')
# plt.title('Black Volatility vs Strike')
# plt.legend()
# plt.grid(True)
# plt.show()


def BlackBasketPayoffMC(f0,k, alpha1, alpha2, sigma1,sigma2, expiry, n_sample= 1000000, seed=42, mc_error=False):
    eps1, eps2 = spss.norm.rvs(loc=0, scale=1, size=(2,n_sample), random_state = seed)
    
    
    diff1 = sigma1*math.sqrt(expiry)
    diff2 = sigma2*math.sqrt(expiry)
    A1 = alpha1*diff1
    A2 = alpha2*diff2
  
    Rt = A1*eps1 - A2*eps2 + f0 - 0.5*(A1*diff1 + A2*diff2)
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


print(f"mc payoff ({T0}y{T}y @ {strike*100:.2f}%):\t\t{mc_forward_payoff*1e4:.2f}bps / [{mc_payoff_lb*1e4:.2f}, {mc_payoff_ub*1e4:.2f}]bps")
print(f"mc ivol:\t\t\t\t{mc_ivol:.2f}bps / [{mc_ivol_lb:.2f}, {mc_ivol_ub:.2f}]bps")
print()


# def _black_basket_option_price(a1, a2, g1, g2, B):
#     return a1 * spss.norm.cdf(g1 - B) + a2 * spss.norm.cdf(g2 - B)

# def BlackBasketApproximatePayoffMaximized(f0, k, alpha1, alpha2, sigma1, sigma2, expiry, convergence=False):
#     stdev1 = sigma1 * math.sqrt(expiry)
#     stdev2 = sigma2 * math.sqrt(expiry)
#     var1 = stdev1 * stdev1
#     var2 = stdev2 * stdev2
    
#     boundary_constant = k - f0 + 0.5 * (alpha1 * var1 + alpha2 * var2)
#     total_stdev = math.sqrt(alpha1 * alpha1 * var1 + alpha2 * alpha2 * var2)
#     B = boundary_constant / total_stdev
    
#     gamma1 = alpha1 * var1 / total_stdev
#     gamma2 = alpha2 * var2 / total_stdev
    
#     print(f"Initial parameters: gamma1={gamma1}, gamma2={gamma2}, B={B}")
    
#     # Define the optimization function and constraints
#     opt_price_func = lambda x: -_black_basket_option_price(alpha1, alpha2, x[0], x[1], x[2])
#     opt_price_constr = lambda x: x[0] * x[0] / (stdev1 * stdev1) + x[1] * x[1] / (stdev2 * stdev2) - 1.0
#     minimization_constraints = ({"type": "eq", "fun": opt_price_constr}, )
    
#     opt = spopt.minimize(fun=opt_price_func, x0=[gamma1, gamma2, B], method="SLSQP", constraints=minimization_constraints)
    
#     if convergence:
#         print(opt)
    
#     if opt.success:
#         gamma1, gamma2, B = opt.x
#         print(f"Optimized parameters: gamma1={gamma1}, gamma2={gamma2}, B={B}")
#         return _black_basket_option_price(alpha1, alpha2, gamma1, gamma2, B)
#     else:
#         print(f"Optimization failed: {opt.message}")
#         return None



# # Prezzo con l'approssimazione Black Basket
# maximization_forward_payoff = BlackBasketApproximatePayoffMaximized(f0, strike, a1, a2, s1, s2, T0)
# maximization_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, maximization_forward_payoff, discount=1) * 1e4 
# print(f"maximization payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{maximization_forward_payoff*1e4:.2f}bps")
# print(f"maximization ivol:\t\t\t{maximization_ivol:.2f}bps")



# def _black_basket_option_price(a1, a2, g1, g2, B):
#     return a1 *np.exp(g1*B-0.5*g1*g1)+a2*np.exp(g2*B-0.5*g2*g2)

# def BlackBasketApproximatePayoffMaximized(f0, k, alpha1, alpha2, sigma1, sigma2, expiry, convergence=False):
#     stdev1 = sigma1 * math.sqrt(expiry)
#     stdev2 = sigma2 * math.sqrt(expiry)
#     var1 = stdev1 * stdev1
#     var2 = stdev2 * stdev2
    
#     boundary_constant = k - f0 + 0.5 * (alpha1 * var1 + alpha2 * var2)
#     total_stdev = math.sqrt(alpha1 * alpha1 * var1 + alpha2 * alpha2 * var2)
#     B = boundary_constant / total_stdev
    
#     gamma1 = alpha1 * var1 / total_stdev
#     gamma2 = alpha2 * var2 / total_stdev
    
#     print(f"Initial parameters: gamma1={gamma1}, gamma2={gamma2}, B={B}")
    
#     # Define the optimization function and constraints
#     opt_price_func = lambda x: -_black_basket_option_price(alpha1, alpha2, x[0], x[1], x[2])
#     opt_price_constr = lambda x: x[0] * x[0] / (stdev1 * stdev1) + x[1] * x[1] / (stdev2 * stdev2) - 1.0
#     minimization_constraints = ({"type": "eq", "fun": opt_price_constr}, )
    
#     opt = spopt.minimize(fun=opt_price_func, x0=[gamma1, gamma2, B], method="SLSQP", constraints=minimization_constraints)
    
#     if convergence:
#         print(opt)
    
#     if opt.success:
#         gamma1, gamma2, B = opt.x
#         print(f"Optimized parameters: gamma1={gamma1}, gamma2={gamma2}, B={B}")
#         return _black_basket_option_price(alpha1, alpha2, gamma1, gamma2, B)
#     else:
#         print(f"Optimization failed: {opt.message}")
#         return None



# # Prezzo con l'approssimazione Black Basket
# maximization_forward_payoff = BlackBasketApproximatePayoffMaximized(f0, strike, a1, a2, s1, s2, T0)
# # maximization_ivol = ql.bachelierBlackFormulaImpliedVol(ql.Option.Call, strike, f0, T0, maximization_forward_payoff, discount=1) * 1e4 
# print(f"maximization payoff ({T0}y{T}y @ {strike*100:.2f}%):\t{maximization_forward_payoff*1e4:.2f}bps")
# # print(f"maximization ivol:\t\t\t{maximization_ivol:.2f}bps")

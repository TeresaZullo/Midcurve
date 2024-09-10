
'pricer swaption con 4 elementi'

import numpy as np
import scipy.stats as spss
import scipy.optimize as spopt
import seaborn as sns
import QuantLib as ql
import matplotlib.pyplot as plt
import math

def _black_basket_option_price(a1,a2,a3,a4,g1,g2,g3,g4,B):
    return a1*spss.norm.cdf(g1-B)+ a2*spss.norm.cdf(g2-B)+a3*spss.norm.cdf(g3-B)+ a4*spss.norm.cdf(g4-B)


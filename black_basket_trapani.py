import numpy as np
import pandas as pd
import QuantLib as ql
import matplotlib.pyplot as plt
import scipy.stats as spss
import scipy.optimize as spopt

from dataclasses import dataclass

from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from numpy.typing import NDArray


@dataclass
class CalibrationData:
    expiry_period: ql.Period
    tenor_period: ql.Period
    expiry: float
    forward_swap: float
    strikes: NDArray[np.float64]
    market_prices: NDArray[np.float64]
    vegas: NDArray[np.float64]
    annuity: float

    def get_total_maturity(self) -> float:
        total_period: ql.Period = self.expiry_period + self.tenor_period
        if total_period.units() == 3:
            return total_period.length()
        elif total_period.units() == 2:
            return total_period.length() / 12.0
        else:
            raise ValueError("Expiries in days are not supported.")


def BlackBasketPayoffAnalyticIter(
    forward_swap: float,
    expiry: float,
    strike: float,
    alpha: NDArray[np.float64],
    sigma: NDArray[np.float64],
    lambda_corrector: float = 0.0,
    tol: float = 1e-8,
) -> float:

    lk = 1.0 - lambda_corrector * strike
    strike = strike * (1.0 - lambda_corrector * forward_swap)
    alpha = alpha * lk
    forward_swap = forward_swap * lk

    rho = np.eye(len(alpha))
    alpha_row = alpha[np.newaxis]
    cov = np.diag(sigma * np.sqrt(expiry)) @ rho @ np.diag(sigma * np.sqrt(expiry))
    stdev = np.sqrt(alpha_row @ cov @ alpha_row.T)

    B = (strike - forward_swap) + 0.5 * np.sum(alpha @ cov)
    B /= stdev

    beta = alpha / stdev
    gamma = cov @ beta.T

    payoff0 = (alpha_row.T * spss.norm.cdf(gamma - B)).sum(axis=0) + spss.norm.cdf(
        -B
    ) * (forward_swap - strike - alpha.sum())

    payoff = -1
    while abs(payoff - payoff0) >= tol:
        payoff0 = payoff

        w = alpha_row.T * spss.norm.pdf(gamma - B)
        gamma = (cov @ w) / np.sqrt(w.T @ cov @ w)
        opt_B = lambda b: alpha @ np.exp(gamma * b - 0.5 * gamma * gamma) + (
            forward_swap - strike - alpha.sum()
        )
        B = spopt.root_scalar(opt_B, x0=B.flatten()).root
        payoff = (alpha_row.T * spss.norm.cdf(gamma - B.flatten())).sum(axis=0) + (
            forward_swap - strike - alpha.sum()
        ) * spss.norm.cdf(-B)

    payoff = (alpha_row.T * spss.norm.cdf(gamma - B)).sum(axis=0) + spss.norm.cdf(
        -B
    ) * (forward_swap - strike - alpha.sum())
    return float(np.squeeze(payoff))


def BlackBasketPayoffMC(
    f0: float,
    expiry: float,
    strike: float | NDArray[np.float64],
    alpha: NDArray[np.float64],
    sigma: NDArray[np.float64],
    rho: float,
    lambda_corrector: float = 0.0,
    n_sample: int = 100000,
    seed: int = 42,
) -> float | NDArray[np.float64]:

    rgen = np.random.default_rng(seed=seed)

    eps0 = rgen.standard_normal(size=n_sample)
    eps0 = (eps0 - np.mean(eps0)) / np.std(eps0)

    eps1 = rgen.standard_normal(size=n_sample)
    eps1 = (eps1 - np.mean(eps1)) / np.std(eps1)

    eps2 = rho * eps0 + np.sqrt(1.0 - rho * rho) * eps1

    eps = np.column_stack((eps0, eps2))
    # Drift e diffusione per ogni componente del basket
    drift = -0.5 * (sigma**2.0) * expiry
    diff = sigma * eps * np.sqrt(expiry)

    # Somma di tutti i 4 componenti per formare il basket finale
    basket = np.sum(alpha * (np.exp(drift + diff) - 1), axis=1, keepdims=True)

    if type(strike) == np.ndarray:
        Rt = (f0 + basket) * ((1.0 - lambda_corrector * strike)[np.newaxis])
    elif type(strike) == float:
        Rt = f0 + basket * (1.0 - lambda_corrector * strike)
    else:
        raise ValueError("typeof strike can be float or array")
    # Payoff della swaption
    payoff_sample = np.clip(Rt - strike * (1.0 - lambda_corrector * f0), 0, np.inf)
    payoff_mean = np.mean(payoff_sample, axis=0)

    return payoff_mean


def BlackBasketMidcurvePayoffMC(
    f0_mc: float,
    expiry: float,
    short_annuity: float,
    long_annuity: float,
    strike: float,
    alpha: NDArray[np.float64],
    sigma: NDArray[np.float64],
    theta: float,
    Q_short: NDArray[np.float64],
    Q_long: NDArray[np.float64],
    n_sample=1000000,
    seed=42,
):

    midcurve_basket_dimension = len(alpha)
    european_basket_dimension = midcurve_basket_dimension // 2

    short_delta = -short_annuity / (long_annuity - short_annuity)
    long_delta = long_annuity / (long_annuity - short_annuity)
    delta = np.repeat((short_delta, long_delta), european_basket_dimension)
    alpha_corrected = delta * alpha

    k = strike - f0_mc

    C_z = np.zeros((european_basket_dimension, european_basket_dimension))

    C_z[0, 0] += np.sin(theta)

    C_w = Q_short @ C_z @ Q_long.T

    rho_outer = np.block(
        [
            [np.eye(european_basket_dimension), C_w],
            [C_w.T, np.eye(european_basket_dimension)],
        ]
    )

    drift = -0.5 * (sigma**2) * expiry
    diff = sigma * np.sqrt(expiry)

    cov_matrix = np.diag(diff) @ rho_outer @ np.diag(diff)

    Z = spss.multivariate_normal.rvs(
        mean=drift, cov=cov_matrix, size=n_sample, random_state=seed
    )

    # Somma di tutti i 4 componenti per formare il basket finalez
    Rt_mc = np.sum(alpha_corrected * (np.exp(Z) - 1), axis=1, keepdims=True)

    # Payoff della swaption
    payoff_sample = np.clip(Rt_mc - k, 0, np.inf)
    payoff_mean = np.mean(payoff_sample, axis=0) * (long_annuity - short_annuity)

    return payoff_mean


def lambda_correction(
    expiry: float,
    short_maturity: float,
    short_annuity: float,
    long_maturity: float,
    long_annuity: float,
) -> tuple[float, float]:
    m = ((short_maturity - expiry) ** 2.0 - (long_maturity - expiry) ** 2.0) / (
        long_annuity - short_annuity
    )
    short_lambda = 0.5 * (m + ((short_maturity - expiry) ** 2.0) / short_annuity)
    long_lambda = 0.5 * (m + ((long_maturity - expiry) ** 2.0) / long_annuity)
    return short_lambda, long_lambda


def get_Q_matrix(
    rho_inner_short: NDArray[np.float64],
    rho_inner_long: NDArray[np.float64],
    alpha_short: NDArray[np.float64],
    alpha_long: NDArray[np.float64],
    sigma_short: NDArray[np.float64],
    sigma_long: NDArray[np.float64],
    expiry: float,
):

    g_short = np.array(alpha_short) * np.array(sigma_short) * np.sqrt(expiry)
    g_short /= np.sqrt(g_short @ g_short)
    g_short = g_short[np.newaxis].T

    g_long = np.array(alpha_long) * np.array(sigma_long) * np.sqrt(expiry)
    g_long /= np.sqrt(g_long @ g_long)
    g_long = g_long[np.newaxis].T

    _, rho_inner_short_evec = np.linalg.eig(rho_inner_short)
    _, rho_inner_long_evec = np.linalg.eig(rho_inner_long)

    _, _, Vh_short = np.linalg.svd(g_short.T @ rho_inner_short_evec)
    _, _, Vh_long = np.linalg.svd(g_long.T @ rho_inner_long_evec)

    Q_short = rho_inner_short_evec @ Vh_short
    Q_long = rho_inner_long_evec @ Vh_long
    return Q_short, Q_long


def european_swaption_smile_calibration_mc_lmfit(
    params: Parameters,
    calibration_data: CalibrationData,
) -> float:

    alpha = params["alpha1"].value, params["alpha2"].value
    sigma = params["sigma1"].value, params["sigma2"].value
    rho = params["rho"].value
    lambda_corrector = params["lambda_corrector"].value
    payoffs = (
        BlackBasketPayoffMC(
            calibration_data.forward_swap,
            calibration_data.expiry,
            calibration_data.strikes,
            np.array(alpha),
            np.array(sigma),
            rho,
            lambda_corrector,
        )
        * 1e4
    )
    err = (payoffs - calibration_data.market_prices) / calibration_data.vegas
    return 0.5 * np.sqrt(err @ err)[0]


def european_swaption_smile_calibration_mc_lmfit_array(
    params: Parameters,
    calibration_data: CalibrationData,
) -> NDArray[np.float64]:
    alpha = params["alpha1"].value, params["alpha2"].value
    sigma = params["sigma1"].value, params["sigma2"].value
    rho = params["rho"].value
    lambda_corrector = params["lambda_corrector"].value
    payoffs = (
        BlackBasketPayoffMC(
            calibration_data.forward_swap,
            calibration_data.expiry,
            calibration_data.strikes,
            np.array(alpha),
            np.array(sigma),
            rho,
            lambda_corrector,
        )
        * 1e4
    )
    err = (payoffs - calibration_data.market_prices) / calibration_data.vegas
    return err


def european_swaption_smile_calibration_analytic_iter_lmfit_array(
    params: Parameters, calibration_data: CalibrationData
) -> NDArray[np.float64]:
    alpha = params["alpha1"].value, params["alpha2"].value
    sigma = params["sigma1"].value, params["sigma2"].value
    lambda_corrector = params["lambda_corrector"].value
    payoffs = []
    for k in calibration_data.strikes:
        payoff = (
            BlackBasketPayoffAnalyticIter(
                calibration_data.forward_swap,
                calibration_data.expiry,
                k,
                np.array(alpha),
                np.array(sigma),
                lambda_corrector,
            )
            * 1e4
        )
        payoffs.append(payoff)
    err = (np.array(payoffs) - calibration_data.market_prices) / calibration_data.vegas
    return err


def get_swaption_data(
    calibration_data: pd.DataFrame, expiry_period: str, tenor_period: str
) -> CalibrationData:

    expiry = ql.Actual365Fixed().yearFraction(
        ql.Date.todaysDate(), ql.Date.todaysDate() + ql.Period(expiry_period)
    )
    market_prices = (
        calibration_data.xs(
            ("market_price", expiry_period, tenor_period), level=(0, 1, 2)
        ).values.flatten()
        * 1e4
    )
    strikes = calibration_data.xs(
        ("strike", expiry_period, tenor_period), level=(0, 1, 2)
    ).values.flatten()
    vegas = (
        calibration_data.xs(
            ("vega", expiry_period, tenor_period), level=(0, 1, 2)
        ).values.flatten()
        * 1e4
    )
    annuity = calibration_data.xs(
        ("annuity", expiry_period, tenor_period), level=(0, 1, 2)
    )[0.0].iloc[0]
    forward_swap = calibration_data.xs(
        ("strike", expiry_period, tenor_period), level=(0, 1, 2)
    )[0.0].iloc[0]

    return CalibrationData(
        expiry_period=ql.Period(expiry_period),
        tenor_period=ql.Period(tenor_period),
        expiry=expiry,
        forward_swap=forward_swap,
        strikes=strikes,
        market_prices=market_prices,
        vegas=vegas,
        annuity=annuity,
    )


def plot_calibration_results(
    model_prices: NDArray[np.float64],
    calibration_data: CalibrationData,
    model_name: str = "model",
) -> None:
    model_vols = []
    mkt_vols = []

    for k, mkt_price, model_price in zip(
        calibration_data.strikes, calibration_data.market_prices, model_prices
    ):

        model_ivol = (
            ql.bachelierBlackFormulaImpliedVol(
                ql.Option.Call,
                k,
                calibration_data.forward_swap,
                calibration_data.expiry,
                model_price,
                discount=1e4,
            )
            / np.sqrt(calibration_data.expiry)
            * 1e4
        )
        mkt_ivol = (
            ql.bachelierBlackFormulaImpliedVol(
                ql.Option.Call,
                k,
                calibration_data.forward_swap,
                calibration_data.expiry,
                mkt_price,
                discount=1e4,
            )
            / np.sqrt(calibration_data.expiry)
            * 1e4
        )

        model_vols.append(model_ivol)
        mkt_vols.append(mkt_ivol)

    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(
        (calibration_data.strikes - calibration_data.forward_swap) * 1e4,
        model_vols,
        c="b",
        label=model_name,
    )
    ax.scatter(
        (calibration_data.strikes - calibration_data.forward_swap) * 1e4,
        mkt_vols,
        marker="x",
        label="market",
    )
    ax.legend()
    ax.set_xlabel("Strikes")
    plt.show()

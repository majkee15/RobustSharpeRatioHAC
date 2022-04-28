from typing import List
import numpy as np
import numpy.typing as npt
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import norm

NDArrayInt = npt.NDArray[np.float_]


def hac_inference(returns: npt.NDArray[np.float_], rf: float = 0.0, alpha:float = 0.05):
    sigma_hat = np.std(returns, axis=0)
    mu_hat = np.mean(returns, axis=0)
    SR_hat = mu_hat / sigma_hat
    standard_error = compute_se_parzen(returns)
    ci = norm.ppf((1 - alpha) / 2) * standard_error
    p_val = 2 * norm.cdf(-np.abs(SR_hat) / standard_error)
    return SR_hat, (SR_hat - ci, SR_hat + ci), p_val, standard_error

def compute_se_parzen(returns: npt.NDArray[np.float_], rf: float = 0.0):
    mu_hat = np.mean(returns, axis=0)
    rets_squared = np.square(returns)
    sigma_hat = np.mean(rets_squared, axis=0)
    gradient = np.zeros(2)
    gradient[0] = 1 / sigma_hat
    gradient[1] = -(mu_hat - rf) / (2 * sigma_hat ** 2)
    v_hat = np.array([returns - mu_hat, rets_squared - sigma_hat]).T
    Psi_hat = compute_psi_hat(v_hat)
    standard_error = np.sqrt(gradient @ Psi_hat @ gradient.T)
    return standard_error


def compute_psi_hat(v_hat: npt.NDArray[np.float_]):
    T = len(v_hat)
    alpha_hat = compute_alpha_hat(v_hat)
    s_star = 2.6614 * (alpha_hat * T) ** 0.2
    s_star = np.minimum(s_star, T - 1)
    psi_hat = compute_gamma_hat(v_hat, 0)
    j = 1
    while j < s_star:
        gamma_hat = compute_gamma_hat(v_hat=v_hat, j=j)
        psi_hat = psi_hat + kernel_parzen(j / s_star) * (gamma_hat + gamma_hat.T)
        j = j + 1
    psi_hat = (T / (T - 4)) * psi_hat
    return psi_hat


def compute_gamma_hat(v_hat: npt.NDArray[np.float_], j: int):
    """
    Computes the gamma matrix
    Args:
        v_hat:
        j:

    Returns:

    """
    T = v_hat.shape[0]
    p = v_hat.shape[1]
    gamma_hat = np.zeros((p, p))
    if j >= T:
        raise ValueError("j must be smaller than the number of observations T!")
    for i in range(j, T):
        gamma_hat = gamma_hat + np.outer(v_hat[i,].T, v_hat[i - j, :])
    gamma_hat = gamma_hat / T
    return gamma_hat


def compute_alpha_hat(v_hat):
    # Optimal bandwidth methodology
    # Andrew 1991
    # Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation
    T = v_hat.shape[0]
    p = v_hat.shape[1]
    numerator = 0.0
    denominator = 1.0
    # Model for the circular bootstrap
    # VAR(1)
    for i in range(p):
        fit = AutoReg(v_hat[:, i], lags=1, trend='c', old_names=False).fit()
        rho_hat = fit.params[1]  # select the AR 1 parameter
        sigma_hat = fit.sigma2
        numerator = numerator + 4 * (rho_hat ** 2) * (sigma_hat ** 4) / ((1 - rho_hat) ** 8)
        denominator = denominator + (sigma_hat ** 4) / ((1 - rho_hat) ** 4)
    return numerator / denominator


def kernel_parzen(x: float):
    """
    Kernel defined as per
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation
    Andrew 1991
    Parzen kernel density estimator.
    Args:
        x:

    Returns:

    """
    result = 0.0
    if np.abs(x) <= 0.5:
        result = 1 - 6 * (x ** 2) + 6 * (np.abs(x) ** 3)
    elif np.abs(x) <= 1:
        result = 2 * (1 - np.abs(x)) ** 3
    return result


def block_size_calibrate(returns: npt.NDArray[np.float_], b_vec: List = [1, 3, 6, 10], alpha: float = 0.05,
                         M: int = 199, K: int = 1000, b_av: int = 5, T_start: int = 50):
    b_size = len(b_vec)
    pass


def circ_bootstrap_calibrate():
    pass


if __name__ == '__main__':
    mu = np.array([0.02, 0.02])
    cov_mat = np.array([[0.3, 0.0], [0.0, 0.3]])
    returns_synth = np.random.multivariate_normal(mu, cov_mat, size=100000)
    # print(hac_inference(returns_synth))

    ret_agg = np.load("../../data/ret_agg.npy")
    ret_hedge = np.load("../../data/ret_hedge.npy")
    # print(compute_se_parzen(ret_hedge))
    # vhat = _vhat(ret_agg)
    # print(compute_psi_hat(vhat))
    print(hac_inference(ret_agg[:, 0]))
    # print(hac_inference(ret_hedge))

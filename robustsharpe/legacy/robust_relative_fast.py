from typing import List
import numpy as np
import numpy.typing as npt
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from scipy.stats import norm
from arch import bootstrap
from numba import jit

NDArrayFloat = npt.NDArray[np.float_]
NDArrayInt = npt.NDArray[np.float_]


def hac_inference(ret: npt.NDArray[np.float_], alpha: float = 0.05, rf: float = 0.0):
    """
    This method is based on Ledoit & Wolf 2009
    Args:
        ret (np.ndarray): Return array of dim T x 2
        rf(float): Risk free rate
        alpha (float): Significance level

    Returns:

    """
    # returns = np.vstack([ret1.values.flatten(), ret2.values.flatten()]).T
    sigma_hat = np.std(ret, axis=0)
    mu_hat = np.mean(ret, axis=0)
    SR_hat = mu_hat / sigma_hat
    standard_error = compute_se_parzen(ret)
    ci = norm.ppf(1 - alpha / 2) * standard_error
    SR_diff = np.diff(SR_hat)[0]
    p_val = 2 * norm.cdf(-np.abs(SR_diff) / standard_error)
    return SR_hat, SR_diff, (SR_diff - ci, SR_diff + ci), p_val, standard_error

def compute_se_parzen(ret: npt.NDArray[np.float_], rf: float = 0.0):
    """
        Computes standard error of the Sharpe ratio estimator.
    Args:
        ret (np.ndarray): Return array of dim T x 2
        rf(float): Risk free rate

    Returns:
        (float): Standard error of Sharpe estimator
    """
    mu_hat = np.mean(ret, axis=0)
    rets_squared = np.square(ret)
    sigma_sq_hat = np.mean(rets_squared, axis=0)
    T = ret.shape[0]
    gradient = np.zeros(4)
    gradient[0] = sigma_sq_hat[0] / np.power(sigma_sq_hat[0] - mu_hat[0] ** 2, 1.5)
    gradient[1] = -sigma_sq_hat[1] / np.power(sigma_sq_hat[1] - mu_hat[1] ** 2, 1.5)
    gradient[2] = -0.5 * mu_hat[0] / np.power(sigma_sq_hat[0] - mu_hat[0] ** 2, 1.5)
    gradient[3] = -0.5 * mu_hat[1] / np.power(sigma_sq_hat[1] - mu_hat[1] ** 2, 1.5)
    v_hat = np.array([ret[:, 0] - mu_hat[0], ret[:, 1] - mu_hat[1],
                      rets_squared[:, 0] - sigma_sq_hat[0], rets_squared[:, 1] - sigma_sq_hat[1]]).T
    Psi_hat = compute_psi_hat(v_hat)  #
    standard_error = np.sqrt(gradient.T @ Psi_hat @ gradient / T)
    return standard_error


def compute_psi_hat(v_hat: npt.NDArray[np.float_]):
    """
        Computes limiting covariance matrix \hat{P\si}
    Args:
        v_hat (np.ndarray):

    Returns:

    """
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


def _vhat(returns: npt.NDArray[np.float_], rf: float = 0.0):
    mu_hat = np.mean(returns, axis=0)
    rets_squared = np.square(returns)
    sigma_sqr_hat = np.mean(rets_squared, axis=0)
    v_hat = np.array([returns[:, 0] - mu_hat[0], returns[:, 1] - mu_hat[1],
                      rets_squared[:, 0] - sigma_sqr_hat[0], rets_squared[:, 1] - sigma_sqr_hat[1]]).T
    return v_hat


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
        gamma_hat = gamma_hat + np.outer(v_hat[i, ].T, v_hat[i - j, :])
    gamma_hat = gamma_hat / T
    return gamma_hat


def compute_alpha_hat(v_hat):
    # Optimal bandwidth methodology
    # Andrew 1991
    # Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation
    T = v_hat.shape[0]
    p = v_hat.shape[1]
    numerator = 0.0
    denominator = 0.0
    # Model for the circular bootstrap
    # VAR(1)
    for i in range(p):
        fit = AutoReg(v_hat[:, i], lags=1, trend='c', old_names=False).fit()
        rho_hat = fit.params[1]  # select the AR 1 parameter
        sigma_hat = np.sqrt(fit.sigma2)
        numerator = numerator + 4 * (rho_hat ** 2) * (sigma_hat ** 4) / ((1 - rho_hat) ** 8)
        denominator = denominator + (sigma_hat ** 4) / ((1 - rho_hat) ** 4)
    return numerator / denominator

@jit(nopython=True, cache=True)
def kernel_parzen(x: float):
    """
    Kernel defined as per
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation
    Andrew 1991
    Parzen kernel density estimator.
    Args:
        x (float):

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
    emp_reject_prob = np.zeros(b_size)
    sigma_hat = np.std(returns, axis=0)
    mu_hat = np.mean(returns, axis=0)
    SR_hat = mu_hat / sigma_hat
    SR_diff = np.diff(SR_hat)[0]
    T = returns.shape[0]
    var_data = np.zeros((T + T_start, 2))
    var_data[0, :] = returns[0, :]
    fit1 = sm.OLS(ret_agg[1:, 0], sm.add_constant(ret_agg[:(T - 1), :])).fit()
    fit2 = sm.OLS(ret_agg[1:, 1], sm.add_constant(ret_agg[:(T - 1), :])).fit()
    coef1 = fit1.params
    coef2 = fit2.params
    resid_mat = np.vstack([fit1.resid, fit2.resid]).T
    # circumvent the number of repeats param
    resid_mat_bootstrap_cheat = np.vstack([resid_mat, resid_mat[-T_start:, :]])
    bs = bootstrap.StationaryBootstrap(b_av, resid_mat_bootstrap_cheat[1:, :])
    resid_mat_star = np.zeros_like(resid_mat_bootstrap_cheat)
    for m, _resid_mat_star in enumerate(bs.bootstrap(K)):
        resid_mat_star[1:, :] = _resid_mat_star[0][0]
        for t in range(1, T + T_start - 1):
            var_data[t, 0] = coef1[0] + coef1[1] * var_data[t - 1, 0] + coef1[2] * var_data[t - 1, 0] + resid_mat_star[
                t, 0]
            var_data[t, 1] = coef2[0] + coef2[1] * var_data[t - 1, 1] + coef2[2] * var_data[t - 1, 1] + resid_mat_star[
                t, 1]
        # no truncation here --> no t_start used
        var_data_trunc = var_data[T_start:(T_start + T)]
        for j in range(b_size):
            p_val = bootstrap_inference(var_data_trunc, block_size=b_vec[j], M=M, delta_null=SR_diff)[3]
            if p_val <= alpha:
                emp_reject_prob[j] = emp_reject_prob[j] + 1
        emp_reject_prob /= K
        closest_rejecet_prob = np.abs(emp_reject_prob - alpha)
        return closest_rejecet_prob


def sharpe_diff(returns):
    sigma_hat = np.std(returns, axis=0)
    mu_hat = np.mean(returns, axis=0)
    SR_hat = mu_hat / sigma_hat
    SR_diff = np.diff(SR_hat)[0]
    return SR_diff


def bootstrap_inference(returns: npt.NDArray[np.float_], block_size: int, alpha: float = 0.05, M: int = 100,
                        delta_null: float = 0.0):
    """

    Args:
        returns (np.ndarray): Return array of dim T x 2
        block_size (int):
        alpha (float): Significance level
        M (int): Number of bootstrap draws
        delta_null (float): Risk free rate

    Returns:

    """
    T = returns.shape[0]
    l = int(np.floor(T / block_size))
    sigma_hat = np.std(returns, axis=0)
    mu_hat = np.mean(returns, axis=0)
    SR_hat = mu_hat / sigma_hat
    SR_diff = np.diff(SR_hat)[0]
    hac_se = compute_se_parzen(returns)
    d = np.abs(SR_diff - delta_null) / hac_se
    p_value = 1.0
    se = 0.0
    bs = bootstrap.CircularBlockBootstrap(block_size, returns)
    d_star_arr = np.zeros(M)
    d_star_arr_non_abs = np.zeros(M)
    y_star = np.zeros(4, dtype='float64')
    psi_hat_star = np.zeros((4, 4), dtype='float64')
    for m, ret_star_boot in enumerate(bs.bootstrap(M)):
        SR_diff_star, se_star, d_star = hac_fast_loop(ret_star_boot[0][0], l, block_size, T, d, SR_diff, y_star, psi_hat_star)
        d_star_arr[m] = d_star
        d_star_arr_non_abs[m] = (SR_diff_star - SR_diff) / se_star
        se = se + se_star
        if d_star >= d:
            p_value = p_value + 1

    p_value = p_value / (M + 1)
    se = se / (M + 1)
    ci = np.quantile(d_star_arr, 1 - alpha) * hac_se
    return SR_hat, SR_diff, (SR_diff - ci, SR_diff + ci), p_value, se

@jit(nopython=True)
def hac_fast_loop(ret_star, l, block_size, T, d, SR_diff, y_star, psi_hat_star):
    sigma_hat1 = np.std(ret_star[:, 0])
    sigma_hat2 = np.std(ret_star[:, 1])
    mu_hat_star1 = np.mean(ret_star[:, 0])
    mu_hat_star2 = np.mean(ret_star[:, 0])
    SR_hat_star1 = mu_hat_star1 / sigma_hat1
    SR_hat_star2 = mu_hat_star2 / sigma_hat2
    SR_diff_star = SR_hat_star2 - SR_hat_star1
    ret_star_squared = np.square(ret_star)
    sigma_sq_hat_star1 = np.mean(ret_star_squared[:, 0])
    sigma_sq_hat_star2 = np.mean(ret_star_squared[:, 1])

    gradient = np.zeros(4)
    gradient[0] = sigma_sq_hat_star1 / np.power(sigma_sq_hat_star1 - mu_hat_star1 ** 2, 1.5)
    gradient[1] = -sigma_sq_hat_star2 / np.power(sigma_sq_hat_star2 - mu_hat_star2 ** 2, 1.5)
    gradient[2] = -0.5 * mu_hat_star1 / np.power(sigma_sq_hat_star1 - mu_hat_star1 ** 2, 1.5)
    gradient[3] = -0.5 * mu_hat_star2 / np.power(sigma_sq_hat_star2 - mu_hat_star2 ** 2, 1.5)
    y_star[0] = ret_star[:, 0] - mu_hat_star1
    y_star[1] = ret_star[:, 1] - mu_hat_star2
    y_star[2] = ret_star_squared[:, 0] - sigma_sq_hat_star1
    y_star[3] = ret_star_squared[:, 1] - sigma_sq_hat_star2
    y_star = y_star.T
    psi_hat_star = 0.0
    for j in range(1, l):
        zeta_star = (block_size ** 0.5) * np.mean(y_star[((j - 1) * block_size):(j * block_size), :], axis=0)
        psi_hat_star = psi_hat_star + np.outer(zeta_star.T, zeta_star)
    psi_hat_star = psi_hat_star / l
    psi_hat_star = (T / (T - 4)) * psi_hat_star
    se_star = np.sqrt(gradient.T @ psi_hat_star @ gradient / T)
    d_star = np.abs(SR_diff_star - SR_diff) / se_star
    if d_star >= d:
        p_value = 1
    return SR_diff_star, se_star, d_star


if __name__ == '__main__':
    import timeit
    mu = np.array([0.02, 0.2])
    cov_mat = np.array([[0.3, 0.0], [0.0, 0.3]])
    returns_synth = np.random.multivariate_normal(mu, cov_mat, size=10000)
    # print(hac_inference(returns_synth))

    ret_agg = np.load("../../data/ret_agg.npy")
    ret_hedge = np.load("../../data/ret_hedge.npy")
    # print(compute_se_parzen(ret_hedge))
    vhat = _vhat(ret_agg)
    # print(compute_psi_hat(vhat))
    # print(hac_inference(returns_synth))

    # print(hac_inference(ret_agg))
    # print(bootstrap_inference(ret_agg, block_size=4, alpha=0.05, M=100))

    print(hac_inference(ret_agg))
    print(bootstrap_inference(ret_agg, block_size=4, alpha=0.05, M=10000))

    # print(block_size_calibrate(ret_agg))
    # print(block_size_calibrate(ret_hedge))

    #timings
    # print(timeit.timeit("hac_inference(ret_agg)", globals=globals(), number=10))

###### Your ID ######
# ID1:
# ID2:
#####################

# imports
import numpy as np
import pandas as pd
import scipy.stats
from IPython.lib.pretty import pretty
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy import stats
import matplotlib.pyplot as plt


# IMPORTANT - read before you start
# * Do NOT change the names of the functions or the names/order of their parameters.
#   The grader calls every function with keyword arguments and with OTHER values than the defaults.
# * Every function must RETURN its answer (printing is not enough) in the exact type described.
# * Do not run code at the top level of this file (only imports and definitions).
#   You may add helper functions and extra parameters, as long as extra parameters have default values.
# * Allowed packages: numpy, pandas, scipy, matplotlib and the Python standard library.


### Question 1 ###

def binom_pvalue(k=16, n=40, p0=0.2):
    """
    Right-sided p-value of observing k or more successes out of n under the null model Binom(n, p0).
    Returns:
    - a float: P(X >= k) for X ~ Binom(n, p0).
    """
    cdf_greater = 1 - stats.binom.cdf(k - 1, n, p0)
    return cdf_greater


def critical_k(n=40, p0=0.2, alpha=0.01):
    """
    The rejection threshold of the right-sided binomial test.

    Returns:
    - an int: the smallest k such that P(X >= k) <= alpha for X ~ Binom(n, p0).
      (If no k in 0..n satisfies this, return n + 1.)
    """

    for k in range(1, n, 1):
        pvalue = binom_pvalue(k, n, p0)
        if pvalue <= alpha:
            return k
    return n + 1


def power(n=40, p0=0.2, p1=0.35, alpha=0.01):
    """
    The power of the test that rejects H0 when X >= critical_k(n, p0, alpha).

    Returns:
    - a float: the probability to reject H0 when the true success probability is p1.
    """
    improvement_threshold = critical_k(n, p0, alpha)

    probability_of_improvement = binom_pvalue(improvement_threshold, n, p1)
    return probability_of_improvement


def power_drops(p0=0.2, p1=0.35, alpha=0.01, n_max=150):
    """
    Returns:
    - a sorted 1-d numpy array of ints: every n in {2, ..., n_max} for which
      power(n, p0, p1, alpha) < power(n - 1, p0, p1, alpha).
    """
    additional_patient_perf_decrease = []
    for n in range(2, n_max + 1):
        if power(n, p0, p1, alpha) < power(n - 1, p0, p1, alpha):
            additional_patient_perf_decrease.append(n)

    return np.array(additional_patient_perf_decrease, dtype=int)


def min_sample_size(p0=0.2, p1=0.35, alpha=0.01, target=0.8):
    """
    Returns:
    - an int: the smallest number of patients n >= 1 for which power(n, p0, p1, alpha) >= target.
    """
    # Starting with an exponential search
    power_n = lambda n: power(n, p0, p1, alpha)

    n = 1
    while power_n(n) < target:
        n *= 2

    # Finalize with binary search
    low = 1
    high = n
    mid = low + ((high - low) // 2)

    while low < high:
        value = power_n(mid)
        if value >= target:
            high = mid
        else:
            low = mid + 1
        mid = low + ((high - low) // 2)

    return n


### Question 2 ###

def shots_pmf(t=10, r=3, p=0.4):
    """
    T is the number of shots a player with success probability p takes until the r-th success.

    Input:
    - t: an int OR a 1-d numpy array of ints.

    Returns:
    - P(T = t): a float if t is an int, a 1-d numpy array of the same length if t is an array.
    """

    T = scipy.stats.nbinom(r, p, loc=r)

    is_single = np.ndim(t) == 0
    t_array = np.atleast_1d(t)

    result = np.empty(len(t_array), dtype=float)

    for i, t in enumerate(t_array):
        result[i] = T.pmf(t_array[i])

    return float(result[0]) if is_single else result


def duel_probs(pA=0.3, rA=5, pB=0.45, rB=8):
    """
    Two players shoot synchronously (one shot each per round). Player A needs rA successes and has success
    probability pA; player B needs rB successes and has success probability pB. All shots are independent.

    Returns:
    - a tuple of 3 floats: (P(A completes strictly before B), P(both complete in the same round),
                            P(B completes strictly before A)).
    """

    threshold = pow(10, -6)

    T_a = scipy.stats.nbinom(rA, pA, loc=rA)
    T_b = scipy.stats.nbinom(rB, pB, loc=rB)

    lower_attempts_player = min(rA, rB)
    higher_attempts_player = max(rA, rB)

    # Same case:

    N = higher_attempts_player
    delta = 0
    p_same_attempts = 0
    p_old = 0

    while True:
        for attempts in range(higher_attempts_player, N + 1):
            p_same_attempts += T_a.pmf(attempts) * T_b.pmf(attempts)

        delta = p_same_attempts - p_old
        if delta < threshold:
            break

        p_old = p_same_attempts

        higher_attempts_player = N + 1
        N = N * 2

    # Ta < Tb case:
    start = rA
    N = higher_attempts_player
    p_a_finishes_first = 0

    while True:
        for attempts in range(start, N):
            p_a_finishes_first += T_a.pmf(attempts) * T_b.sf(attempts)

        delta = p_a_finishes_first - p_old
        if delta < threshold:
            break

        p_old = p_a_finishes_first

        start = N + 1
        N = N * 2

    # Ta > Tb case:
    start = rB
    N = higher_attempts_player
    p_b_finishes_first = 0

    while True:
        for attempts in range(start, N):
            p_b_finishes_first += T_a.sf(attempts) * T_b.pmf(attempts)

        delta = p_b_finishes_first - p_old
        if delta < threshold:
            break

        p_old = p_b_finishes_first

        start = N + 1
        N = N * 2

    return p_a_finishes_first, p_same_attempts, p_b_finishes_first


def duel_expected_length(pA=0.3, rA=5, pB=0.45, rB=8):
    """
    Same setting as duel_probs.

    Returns:
    - a float: the expected number of rounds until the FIRST of the two players completes her task.
    """

    threshold = pow(10, -6)

    T_a = scipy.stats.nbinom(rA, pA, loc=rA)
    T_b = scipy.stats.nbinom(rB, pB, loc=rB)

    p_M = lambda t: (T_a.pmf(t) * T_b.sf(t)
                     + T_a.pmf(t) * T_b.pmf(t)
                     + T_a.sf(t) * T_b.pmf(t))

    start = 1
    N = max(rA, rB)
    expected_M = 0
    old_expected_M = 0

    while True:
        for attempts in range(start, N + 1):
            expected_M += attempts * p_M(attempts)

        delta = expected_M - old_expected_M
        if delta < threshold:
            break

        old_expected_M = expected_M
        start = N + 1
        N = N * 2

    return expected_M


### Question 3 ###

def conv_pmf(P, Q):
    """
    Input:
    - P, Q: 2-d numpy arrays [[values], [probabilities]] of two independent random variables X ~ P, Y ~ Q.
      The values are integers (not necessarily sorted, consecutive, or positive).

    Returns:
    - a 2-d numpy array [[values], [probabilities]] with the distribution of X + Y,
      values sorted in increasing order (values with probability 0 may be included or omitted).
    """
    pass


def nfold_pmf(P, n=2):
    """
    Input:
    - P: 2-d numpy array [[values], [probabilities]] with integer values.
    - n: an int >= 1.

    Returns:
    - a 2-d numpy array [[values], [probabilities]] with the distribution of X_1 + ... + X_n,
      where X_1, ..., X_n are independent and each has the distribution P (sorted values).
    """
    pass


### Question 4 ###

def cc_cdf(N=20, t=60):
    """
    T_N is the waiting time (number of draws) for a full collection of N equiprobable coupons.

    Returns:
    - a float: P(T_N <= t), computed exactly (no simulation), with a RELATIVE error below 1e-6
      even when the probability is tiny (e.g. 1e-40).
    """
    pass


def cc_partial_quantile(N=100, m=80, q=0.95):
    """
    Returns:
    - an int: the smallest t such that P(at least m different coupons appear among the first t draws) >= q.
    """
    pass


### Question 5 ###

def clean_log(df):
    """
    Input:
    - df: a pandas DataFrame with the raw inspection log (as returned by pd.read_csv on a log file).

    Returns:
    - a pandas DataFrame with ONE row per usable batch (a batch whose true number of defective items can be
      determined from the log) and (at least) the columns: batch_id, machine, inspected, defective,
      holding the TRUE values. Machine names must be exactly 'A', 'B', 'C' or 'D'.
    """
    valid_machine_names = ['A', 'B', 'C', 'D']
    df = df[["batch_id", "machine", "inspected", "defective"]]

    df = df.loc[df['machine'].isin(valid_machine_names)]
    df = df.loc[
        (df['defective'] >= 0) & (df['defective'] <= df['inspected'])
        ]

    return df


def defect_rates(df):
    """
    Input:
    - df: a raw inspection log (pandas DataFrame).

    Returns:
    - a dict {machine: estimated defect rate}, e.g. {'A': 0.031, 'B': 0.052, 'C': 0.044, 'D': 0.027}.
      The estimated rate of a machine = (total defective items) / (total inspected items) over its usable batches.
    """
    # all_tables = []
    # for cleaned_table in df.values():
    #     all_tables.append(cleaned_table)
    #
    #
    # combined_cleaned = pd.concat(all_tables)

    clean_df = clean_log(df)
    groups = clean_df.groupby('machine')
    rates_by_machine = {}

    for machine_name, machine_rows in groups:
        total_inspected = machine_rows['inspected'].sum()
        total_defective = machine_rows['defective'].sum()
        machine_rate = total_defective / total_inspected
        rates_by_machine[machine_name] = machine_rate

    return rates_by_machine


def log10_pvalue(df, machine='B', p0=0.03):
    """
    Input:
    - df: a raw inspection log (pandas DataFrame).
    - machine: 'A', 'B', 'C' or 'D'.
    - p0: the defect rate claimed by the manufacturer.

    Returns:
    - a float: log10 of the right-sided binomial p-value P(X >= D), X ~ Binom(N, p0), where N and D are the total
      numbers of inspected and defective items of this machine (over its usable batches).
      Must be accurate (relative error < 1e-6) even when the p-value is smaller than 1e-300.
    """
    clean_df = clean_log(df)

    machine_rows = clean_df.groupby('machine').get_group(machine)
    total_inspected = machine_rows['inspected'].sum()
    total_defective = machine_rows['defective'].sum()
    X = stats.binom(total_inspected, p0)
    possible_counts = np.arange(total_defective, total_inspected + 1)
    log_probs = X.logpmf(possible_counts)
    log_pvalue = scipy.special.logsumexp(log_probs) / np.log(10)
    return log_pvalue


def items_needed(df, machine='A', r=10, q=0.95):
    """
    Input:
    - df: a raw inspection log (pandas DataFrame).

    Returns:
    - an int: the smallest number n of products from this machine we need to inspect so that
      P(at least r of them are defective) >= q, assuming each product is defective independently
      with probability equal to the machine's estimated defect rate (from defect_rates).
    """
    clean_df = clean_log(df)
    machine_rates = defect_rates(clean_df)
    rate = machine_rates[machine]
    threshold = q
    X = stats.nbinom(r, rate, loc=r)

    low = 0
    high = r
    while X.cdf(high) < threshold:
        high = high * 2

    while low < high:
        mid = low + ((high - low) // 2)

        p_defect = X.cdf(mid)

        if p_defect >= threshold:
            high = mid
        else:
            low = mid + 1
    return low

###### Your ID ######
# ID1:
# ID2:
#####################

# imports
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.special import logsumexp


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
    return int(stats.binom.isf(alpha, n, p0)) + 1


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

    start = 1
    end = 2
    while True:
        for n in range(start, end + 1, 1):
            if power_n(n) >= target:
                return n
        start *= 2
        end *= 2


### Question 2 ###

def shots_pmf(t=10, r=3, p=0.4):
    """
    T is the number of shots a player with success probability p takes until the r-th success.

    Input:
    - t: an int OR a 1-d numpy array of ints.

    Returns:
    - P(T = t): a float if t is an int, a 1-d numpy array of the same length if t is an array.
    """

    result = stats.nbinom.pmf(np.asarray(t)-r,r,p)
    if np.ndim(t) == 0:
        return float(result)
    return result


def duel_probs(pA=0.3, rA=5, pB=0.45, rB=8):
    """
    Two players shoot synchronously (one shot each per round). Player A needs rA successes and has success
    probability pA; player B needs rB successes and has success probability pB. All shots are independent.

    Returns:
    - a tuple of 3 floats: (P(A completes strictly before B), P(both complete in the same round),
                            P(B completes strictly before A)).
    """

    miss_A = stats.nbinom.ppf(1 - 1e-12, rA, pA)
    miss_B = stats.nbinom.ppf(1 - 1e-12, rB, pB)

    lim_A = miss_A + rA
    lim_B = miss_B + rB

    limit = int(max(lim_A, lim_B))

    t = np.arange(1, limit + 1)

    prob_A = shots_pmf(t, rA, pA)
    prob_B = shots_pmf(t, rB, pB)

    P_A_wins = np.sum(prob_A * stats.binom.cdf(rB - 1, t, pB))

    P_tie = np.sum(prob_A * prob_B)

    p_b_wins = np.sum(prob_B * stats.binom.cdf(rA - 1, t, pA))

    return float(P_A_wins), float(P_tie), float(p_b_wins)


def duel_expected_length(pA=0.3, rA=5, pB=0.45, rB=8):
    """
    Same setting as duel_probs.

    Returns:
    - a float: the expected number of rounds until the FIRST of the two players completes her task.
    """


    lim_A = stats.nbinom.ppf(1 - 1e-12, rA, pA) + rA
    lim_B = stats.nbinom.ppf(1 - 1e-12, rB, pB) + rB

    limit = int(max(lim_A, lim_B))

    t = np.arange(0, limit + 1)

    A_not_done = stats.binom.cdf(rA - 1, t, pA)
    B_not_done = stats.binom.cdf(rB - 1, t, pB)

    expected_length = np.sum(A_not_done * B_not_done)

    return float(expected_length)


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
    values = []
    probabilities = []

    for i in range(len(P[0])):
        for j in range(len(Q[0])):
            new_value = P[0][i] + Q[0][j]
            new_prob = P[1][i] * Q[1][j]

            found = False

            for k in range(len(values)):
                if new_value == values[k]:
                    probabilities[k] += new_prob
                    found = True
                    break
            if not found:
                values.append(new_value)
                probabilities.append(new_prob)

    order = np.argsort(values)
    return np.array([values, probabilities])[:, order]


def nfold_pmf(P, n=2):
    """
    Input:
    - P: 2-d numpy array [[values], [probabilities]] with integer values.
    - n: an int >= 1.

    Returns:
    - a 2-d numpy array [[values], [probabilities]] with the distribution of X_1 + ... + X_n,
      where X_1, ..., X_n are independent and each has the distribution P (sorted values).
    """
    result = P

    for i in range(n - 1):
        result = conv_pmf(result, P)

    order = np.argsort(result[0])
    return result[:, order]


### Question 4 ###

def cc_cdf(N=20, t=60):
    """
    T_N is the waiting time (number of draws) for a full collection of N equiprobable coupons.

    Returns:
    - a float: P(T_N <= t), computed exactly (no simulation), with a RELATIVE error below 1e-6
      even when the probability is tiny (e.g. 1e-40).
    """
    if t < N:
        return 0.0

    j = np.arange(N + 1)
    stay = j / N
    move = (N - j[:-1]) / N

    dp = np.zeros(N + 1)
    dp[0] = 1.0

    for _ in range(t):
        new = dp * stay
        new[1:] += dp[:-1] * move
        dp = new

    return float(dp[N])


def cc_partial_quantile(N=100, m=80, q=0.95):
    """
    Returns:
    - an int: the smallest t such that P(at least m different coupons appear among the first t draws) >= q.
    """
    j = np.arange(N + 1)
    stay = j / N
    move = (N - j[:-1]) / N

    dp = np.zeros(N + 1)
    dp[0] = 1.0

    t = 0
    while dp[m:].sum() < q:
        new = dp * stay
        new[1:] += dp[:-1] * move
        dp = new
        t += 1

    return int(t)


### Question 5 ###

def handle_names_and_bad_values(df):
    # Handle names
    ## Correct those that are recording errors
    df['machine'] = df['machine'].str.strip().str.upper()

    # Keep only the ones that look exactly like that
    valid_names = ['A', 'B', 'C', 'D']
    df = df.loc[df['machine'].isin(valid_names)]

    # Remove any inspection that has negative values or more than the products that inputted for the machine
    df = df.loc[(df['defective'] >= 0) & (df['defective'] <= df['inspected'])]

    return df


def remove_duplicates(df):
    # Check for exactly identical rows
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "log_line"], keep="first")

    # Check if there are batch_ids that appeared more than once, if thats true, make sure it's not a recording problem
    df = df.drop_duplicates(subset=["batch_id", "machine", "sensor_fw", "inspected", "defective"], keep="first")

    # Check if there are timestamps that appeared more than once, if thats true, make sure it's not a recording problem
    df = df.drop_duplicates(subset=["timestamp", "machine", "sensor_fw", "inspected", "defective"], keep="first")

    return df


def firmware_ratios(df):
    # Each (machine, firmware): its defect rate / median rate of the OTHER firmware versions on that machine.
    d = df[df["defective"] >= 0].copy()
    d["machine"] = d["machine"].str.strip().str.upper()
    rates = (d.groupby(["machine", "sensor_fw"])[["defective", "inspected"]].sum()
             .assign(rate=lambda t: t["defective"] / t["inspected"]))
    rows = []
    for (m, fw), r in rates.iterrows():
        others = rates.loc[m].drop(index=fw)["rate"]
        if len(others) == 0:
            continue
        rows.append({"machine": m, "sensor_fw": fw, "rate": r["rate"],
                     "others_rate": others.median(), "ratio": r["rate"] / others.median()})
    return pd.DataFrame(rows)


def detect_scaled_firmware(df, tol=0.25, max_factor=10):
    # {firmware: k} for firmware versions whose rate is about k times the others (k >= 2) and
    # whose every recorded value is divisible by k.
    ratios = firmware_ratios(df)
    d = df[df["defective"] >= 0]
    scaled = {}
    for fw, g in ratios.groupby("sensor_fw"):
        ratio = g["ratio"].median()
        k = int(round(ratio))
        if not 2 <= k <= max_factor or abs(ratio - k) > tol:
            continue
        if (d.loc[d["sensor_fw"] == fw, "defective"] % k == 0).all():
            scaled[fw] = k
    return scaled


def scale_firmware_defective_output(df):
    df = df.copy()
    for fw, k in detect_scaled_firmware(df).items():
        rows = (df["sensor_fw"] == fw) & (df["defective"] >= 0)
        df.loc[rows, "defective"] //= k
    return df


def repeat_thresholds(df, alpha=0.01):
    rows = []
    for (machine, fw), group in df.groupby(["machine", "sensor_fw"]):
        # Defect rate and typical batch size of this machine + firmware
        rate = group["defective"].sum() / group["inspected"].sum()
        batch_size = round(group["inspected"].mean())

        # Chance that two batches get the same number of defective items by luck
        counts = np.arange(batch_size + 1)
        p_same = (stats.binom.pmf(counts, batch_size, rate) ** 2).sum()

        # Smallest run length whose expected number of lucky runs is <= alpha
        threshold = 2
        while len(group) * p_same ** (threshold - 1) > alpha:
            threshold += 1

        rows.append({"machine": machine, "sensor_fw": fw, "batches": len(group),
                     "rate": rate, "p_same": p_same, "threshold": threshold})

    return pd.DataFrame(rows).set_index(["machine", "sensor_fw"])


def remove_stuck_runs(df, thresholds):
    # Keep the first value of a run that is at least `threshold` long, drop its repeats.
    df = df.sort_values(["machine", "log_line"])
    run = (df["defective"] != df.groupby("machine")["defective"].shift()).cumsum()
    run_len = df.groupby(run)["defective"].transform("size")
    is_repeat = df.groupby(run).cumcount() > 0
    limit = df.join(thresholds["threshold"], on=["machine", "sensor_fw"])["threshold"]
    return df[~(is_repeat & (run_len >= limit))].sort_index()


def clean_log(df):
    """
    Input:
    - df: a pandas DataFrame with the raw inspection log (as returned by pd.read_csv on a log file).

    Returns:
    - a pandas DataFrame with ONE row per usable batch (a batch whose true number of defective items can be
      determined from the log) and (at least) the columns: batch_id, machine, inspected, defective,
      holding the TRUE values. Machine names must be exactly 'A', 'B', 'C' or 'D'.
    """

    df_no_names_and_bad_values = handle_names_and_bad_values(df)
    df_no_duplicates = remove_duplicates(df_no_names_and_bad_values)
    df_scaled_fw = scale_firmware_defective_output(df_no_duplicates)
    df_no_faulty_sensors = remove_stuck_runs(df_scaled_fw, repeat_thresholds(df_scaled_fw))
    cleaned_df = df_no_faulty_sensors[["batch_id", "machine", "inspected", "defective"]]

    return cleaned_df


def defect_rates(df):
    """
    Input:
    - df: a raw inspection log (pandas DataFrame).

    Returns:
    - a dict {machine: estimated defect rate}, e.g. {'A': 0.031, 'B': 0.052, 'C': 0.044, 'D': 0.027}.
      The estimated rate of a machine = (total defective items) / (total inspected items) over its usable batches.
    """
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
    log_pvalue = logsumexp(log_probs) / np.log(10)
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
    machine_rates = defect_rates(df)
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

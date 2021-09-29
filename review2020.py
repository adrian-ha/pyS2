import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def compute_fwd_interpolation(fwd, swap_rate_end, zero_rates_known, start_tenor, end_tenor):
    """
    Interpolates forward rates as described by EIOPA (constant forward rates between liquid maturities).
    Source: 'Background Document On The Opinion On The 2020 Review Of Solvency II' p.787f.

    :param fwd: fwd to be calculated: 1d-Ndarray
    :param swap_rate_end: last liquid swap rate in interval. E.g. for 15y --> 20y period this would be 20y swap rate: 1d-Ndarray
    :param zero_rates_known: zero rates known so far: 1d-Ndarray
    :param start_tenor: last known tenor / maturity: float
    :param end_tenor: final tenor / maturity (end of interpolation): float
    :return: cash value of swap (par swap = 1)
    """
    left_side_1 = sum([1. / ((1+rate)**(t+1)) for t, rate in enumerate(zero_rates_known)])
    left_side_2 = sum([1. / ((1+fwd)**t) for t in range(1, end_tenor - start_tenor + 1)])
    left_side = left_side_1 + (1. / (1+zero_rates_known[-1])**start_tenor) * left_side_2
    right_side = 1. / ((1+zero_rates_known[start_tenor-1])**(start_tenor) * (1+fwd)**(end_tenor - start_tenor))
    return swap_rate_end * left_side + right_side


def error_fwd_interpolation(fwd, args):
    """
    Error function for numeric procedure required to interpolate between observed liquid market rates

    :param fwd: dummy forward rate (rate is later used for numeric operation)
    :param args: optimization parameters
    :return: (cash value of swap - 1) ---> if fwd rate is solved for this should be minimized
    """
    swap_rate_end = args[0]
    zero_rates_known = args[1]
    start_tenor = args[2]
    end_tenor = args[3]
    res = compute_fwd_interpolation(fwd, swap_rate_end, zero_rates_known, start_tenor, end_tenor)
    return np.abs(res-1)


def interpolate_fwd(fwd, swap_rate_end, zero_rates_known, start_tenor, end_tenor):
    """
    Interpolates forward rates according to methodology described by EIOPA review 2020.
    Source: 'Background Document On The Opinion On The 2020 Review Of Solvency II' p.788

    :param fwd: fwd to be calculated: 1d-Ndarray
    :param swap_rate_end: last liquid swap rate in interval. E.g. for 15y --> 20y period this would be 20y swap rate: 1d-Ndarray
    :param zero_rates_known: zero rates known so far: 1d-Ndarray
    :param start_tenor: last known tenor / maturity: float
    :param end_tenor: final tenor / maturity (end of interpolation): float
    :return: final_fwd that can be used for calculation of all zero rates in between the observed swap rates: 1d-Ndarray
    """
    # Optimization tolerance
    TOLERANCE = 1e-10

    # Number of assets
    number_of_fwds = len(fwd)

    # Long only weights to be assigned
    bound = (-1.0, 1.0)
    bounds = tuple(bound for asset in range(number_of_fwds))

    # Optimisation (minimize error)
    optimize_result = minimize(fun=error_fwd_interpolation,
                               x0=fwd,
                               args=[swap_rate_end, zero_rates_known, start_tenor, end_tenor],
                               method='SLSQP',
                               bounds=bounds,
                               tol=TOLERANCE,
                               options={'disp': False})

    # Recover the weights from the optimised object
    final_fwd = optimize_result.x

    return final_fwd


def compute_interpolated_zero(swap_rates_market, liquid_maturities):
    """
    Based on swap rates with maturity of 1-12, 15, 20, 25, 30, 40 and 50 years this function builds the forward and zero rates.
    Source: "Background Document On The Opinion On The 2020 Review Of Solvency II" p.787f.

    :param swap_rates_market: market rates where index position indicates maturity (e.g. [4] corresponds to a maturity of 5y): Ndarray
    :param liquid_maturities: maturities of the respective interest rates: 1d-Ndarry
    :return: contains interpolated zero curve based on liquid swap rates: list
    """
    # Creates dummy to start interpolation
    fwd_dummy = np.array([0.01])

    # Number of liquid maturities
    N = len(liquid_maturities)

    # Construct zero rates
    zero_rates_market = np.array(swap_to_zero(swap_rates_market))

    # Starting point of zero curve with interpolation
    zero_rates_market_interpolated = [zero_rates_market[0]]

    # Loop through each liquid rate pair and see whether interpolation is required
    for liquid_idx in range(1, N):

        t1 = liquid_maturities[liquid_idx]
        t2 = liquid_maturities[liquid_idx - 1]
        t = t1 - t2

        if t > 1:

            fwd = interpolate_fwd(fwd=fwd_dummy, swap_rate_end=swap_rates_market[t1 - 1],
                                  zero_rates_known=np.array(zero_rates_market_interpolated), start_tenor=t2,
                                  end_tenor=t1)
            z = (1 + zero_rates_market_interpolated[-1]) ** t2

            for zero_runs in range(1, t + 1):
                z_temp = (z * (1 + fwd) ** zero_runs) ** (1. / (t2 + zero_runs))
                zero_rates_market_interpolated.append(z_temp[0] - 1.)

        else:
            # Assuming this can only happen for short maturities in the beginning
            zero_rates_market_interpolated.append(zero_rates_market[t1 - 1])

    return zero_rates_market_interpolated


def extract_fwds(zero_rates_market_interpolated, liquid_maturities, fsp=20):
    """
    Extracts main (interpolated) forwards.
    Source: "Background Document On The Opinion On The 2020 Review Of Solvency II" p.787ff.

    :param zero_rates_market_interpolated: interpolated zero rates where index position indicates maturity (e.g. [4] corresponds to a maturity of 5y): 1d-Ndarray
    :param liquid_maturities: maturities of the respective interest rates: 1d-Ndarray
    :param fsp: maturity in years of first smoothing point: float
    :return: forwards_pre_fsp & forwards_llfr (rates required for llfr calculation): list
    """
    # Number of liquid maturities
    N = len(liquid_maturities)

    # Init
    forwards_pre_fsp = [zero_rates_market_interpolated[0]]
    forwards_llfr = []

    # Loop through each liquid rate pair and calculate required fwds
    for liquid_idx in range(1, N):

        t1 = liquid_maturities[liquid_idx]
        t2 = liquid_maturities[liquid_idx - 1]
        t = t1 - t2

        if t1 <= fsp:
            fwd = ((1 + zero_rates_market_interpolated[t1 - 1]) ** t1 / (
                        1 + zero_rates_market_interpolated[t2 - 1]) ** t2) ** (1 / t) - 1
            forwards_pre_fsp.append(fwd)

        if t1 == fsp:
            fwd = ((1 + zero_rates_market_interpolated[t1 - 1]) ** t1 / (
                        1 + zero_rates_market_interpolated[t2 - 1]) ** t2) ** (1 / t) - 1
            forwards_llfr.append(fwd)

        if t1 > fsp:
            t = t1 - fsp
            fwd = ((1 + zero_rates_market_interpolated[t1 - 1]) ** t1 / (
                        1 + zero_rates_market_interpolated[fsp - 1]) ** fsp) ** (1 / t) - 1
            forwards_llfr.append(fwd)

    return forwards_pre_fsp, forwards_llfr


def compute_llfr(fwd, volume=np.array([3.3, 1.45 , 6, 0.3, 0.4]), va=0.0):
    """
    Calculates last liquid forward rate (llfr) as described by EIOPA review 2020.
    Source: "Background Document On The Opinion On The 2020 Review Of Solvency II" p.788f.

    volume[0] = 20y
    volume[1] = 25y
    volume[2] = 30y
    volume[3] = 40y
    volume[4] = 50y
    fwd[0] = 15y-20y --> 15y5y + VA
    fwd[1] = 20y-25y --> 20y5y
    fwd[2] = 20y-30y --> 20y10y
    fwd[3] = 20y-40y --> 20y20y
    fwd[4] = 20y-50y --> 20y30y

    :param fwd: forward rates: 1d-Ndarray
    :param volume: respective volume / weighting of forward rates: 1d-Ndarray
    :param va: volatility adjustment in bps (as float, e.g. 10): float
    :return: last liquid forward rate
    """
    weight = volume / volume.sum()
    fwds_incl_va = fwd.copy()
    fwds_incl_va[0] = fwds_incl_va[0] + va / 10000.0
    llfr = fwds_incl_va * weight
    return np.array([llfr.sum()])


def compute_curve_with_va(forwards_pre_fsp, liquid_maturities, fsp=20, va=0):
    """
    Constructs zero curve from forwards until FPS  - potentially including VA.
    Source: "Background Document On The Opinion On The 2020 Review Of Solvency II" p.788f.

    :param forwards_pre_fsp: forwards between liquid rates: 1d-Ndarray
    :param liquid_maturities: maturities of the respective interest rates: 1d-Ndarray
    :param fsp: maturity in years of first smoothing point: float
    :param va: volatility adjustment: float
    :return: contains interpolated zero curve up to a maturity of 20y including va: list
    """
    # Number of liquid maturities smaller than or equal to fsp
    N = len(liquid_maturities[liquid_maturities <= fsp])

    # Input check
    assert N == len(forwards_pre_fsp)

    # Add va to all forwards
    forwards_pre_fsp_incl_va = forwards_pre_fsp.copy()
    forwards_pre_fsp_incl_va = [fwd_rate + va / 10000.0 for fwd_rate in forwards_pre_fsp_incl_va]

    # Init
    zero_rates = [forwards_pre_fsp_incl_va[0]]

    # Loop through each liquid rate pair and calculate required fwds
    for liquid_idx in range(1, N):

        t1 = liquid_maturities[liquid_idx]
        t2 = liquid_maturities[liquid_idx - 1]
        t = t1 - t2

        if t > 1:

            fwd = (1 + forwards_pre_fsp_incl_va[liquid_idx])
            base_rate = ((1 + zero_rates[-1]) ** t2)

            for zero_run in range(1, t + 1):
                z = base_rate * fwd ** zero_run
                z = z ** (1 / (t2 + zero_run)) - 1
                zero_rates.append(z)

        else:
            z = ((1 + zero_rates[-1]) ** t2) * ((1 + forwards_pre_fsp_incl_va[liquid_idx]) ** t)
            z = z ** (1 / t1) - 1
            zero_rates.append(z)

    return zero_rates


def big_b(h, alpha=0.1):
    """
    Helper function
    """
    top = 1 - np.exp(-alpha*h)
    bottom = alpha * h
    return top / bottom


def extrapolate_fwds(h, ufr, llfr, alpha=0.10):
    """
    Extrapolates forward rates beyond fsp.
    Source: "Background Document On The Opinion On The 2020 Review Of Solvency II" p.789.

    fwd_fsp_fsp_plus_h:
        fwd that can be used for calculation of all zero rates in between the observed swap rates

    :param h: forward rate beyond fsp (at year 20+h): 1d-Ndarray
    :param ufr: ultimate forward rate: 1d-Ndarray
    :param llfr: last liquid forward rate: 1d-Ndarray
    :param alpha: convergence speed: float
    :return: fwd_fsp_fsp_plus_h: 1d-Ndarray
    """
    fwd_fsp_fsp_plus_h = np.log(1 + ufr) + (llfr - np.log(1 + ufr)) * big_b(h, alpha)
    return fwd_fsp_fsp_plus_h


def extrapolate_zero(known_zero_rates, ufr, llfr, alpha=0.10, fsp=20):
    """
    Extrapolation of zero rates beyond fsp.
    Source: "Background Document On The Opinion On The 2020 Review Of Solvency II" p.790.

    :param known_zero_rates:
    :param ufr: ultimate forward rate
    :param llfr: last liquid forward rate
    :param alpha: convergence speed
    :param fsp: first-smoothing-point
    :return: extrapolated_zero_rates: 1d-Ndarray
    """
    # FSP
    z_fsp = known_zero_rates[fsp - 1]

    # Extrapolated zero rates
    extrapolated_zero_rates = known_zero_rates[0:fsp]

    # Regardless of fsp we want to calculate the extrapolated rates with a maturity of up to 120y
    up_to = 120 - fsp

    # Calculate extrapolated zero rates up to and including 120y maturity
    for h in range(1, up_to + 1):
        z = np.exp((fsp * z_fsp + h * extrapolate_fwds(h, ufr, llfr, alpha)) / (fsp + h)) - 1
        extrapolated_zero_rates.append(z[0])

    return extrapolated_zero_rates


def alternative_extrapolation(input_rates, input_liquid, ufr, fsp=20, alpha=None, va=0.0, volume_traded=np.array([3.3, 1.45, 6, 0.3, 0.4])):
    """
    Wrapper function for alternative extrapolation method of SII curves.
    Source: "Background Document On The Opinion On The 2020 Review Of Solvency II" p.783-790.

    :param input_rates:
    :param input_liquid:
    :param ufr: ultimate forward rate
    :param fsp: first-smoothing-point
    :param alpha: convergence speed
    :param va: volatility adjustment
    :param volume_traded: weighting for llfr
    :return:
    """

    # Assign base variables
    liquid_maturities = np.where(input_liquid == 1)[0] + 1
    liquid_rates_swap = input_rates[np.where(input_liquid == 1)]

    # Interpolated liquid zero rates
    zero_rates_market_interpolated = compute_interpolated_zero(input_rates, liquid_maturities)

    # Introduction mechanism
    if alpha is None:
        rate_at_fsp = zero_rates_market_interpolated[fsp - 1]
        alpha_interpolator = interp1d([-0.5, 0.5], [0.2, 0.1], kind='linear', bounds_error=False, fill_value=(0.2, 0.1))
        alpha = alpha_interpolator(rate_at_fsp)

    # Relevant forwards
    forwards_pre_fsp, forwards_llfr = extract_fwds(zero_rates_market_interpolated, liquid_maturities, fsp)

    # Last liquid forward rate
    llfr = compute_llfr(forwards_llfr, va=va, volume=volume_traded)

    # Zero curve including potential volatility adjustment
    zero_curve_va = compute_curve_with_va(forwards_pre_fsp, liquid_maturities, fsp, va)

    # Extrapolated zero curve
    extrapolated_zero_curve = extrapolate_zero(zero_curve_va, ufr, llfr, alpha, fsp)

    return extrapolated_zero_curve
import pandas as pd
import numpy as np

def get_eiopa_shock(method="standard"):
    """


    Parameters
    --------------



    Returns
    --------------

    """
    # Input checks
    assert method in ("standard", "new")

    # Different shocks as described by EIOPA
    if method == "standard":
        floor = -100.0  # --> no floor

        up_relative_20 = [0.7, 0.7, 0.64, 0.59, 0.55, 0.52, 0.49, 0.47, 0.44, 0.42, 0.39, 0.37, 0.35, 0.34, 0.33, 0.31, 0.3, 0.29, 0.27, 0.26]
        down_relative_20 = [0.75, 0.65, 0.56, 0.5, 0.46, 0.42, 0.39, 0.36, 0.33, 0.31, 0.3, 0.29, 0.28, 0.28, 0.27, 0.28, 0.28, 0.28, 0.29, 0.29]
        up_absolute_20 = 0.0
        down_absolute_20 = 0.0

        up_relative_90 = 0.2
        down_relative_90 = 0.2
        up_absolute_60 = 0.0
        down_absolute_60 = 0.0

    if method == "new":
        floor = -0.0125

        up_relative_20 = [0.61, 0.53, 0.49, 0.46, 0.45, 0.41, 0.37, 0.34, 0.32, 0.3, 0.3, 0.3, 0.3, 0.29, 0.28, 0.28, 0.27, 0.26, 0.26, 0.25]
        down_relative_20 = [0.58, 0.51, 0.44, 0.4, 0.4, 0.38, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.47, 0.48, 0.49, 0.49, 0.5]
        up_absolute_20 = [0.0214, 0.0186, 0.0172, 0.0161, 0.0158, 0.0144, 0.013, 0.0119, 0.0112, 0.0105, 0.0105, 0.0105, 0.0105, 0.0102, 0.0098, 0.0098, 0.0095, 0.0091, 0.0091, 0.0088]
        down_absolute_20 = [0.0116, 0.0099, 0.0083, 0.0074, 0.0071, 0.0067, 0.0063, 0.0062, 0.0061, 0.0061, 0.006, 0.006, 0.0059, 0.0058, 0.0057, 0.0056, 0.0055, 0.0054, 0.0052, 0.005]

        up_relative_90 = 0.2
        down_relative_90 = 0.2
        up_absolute_60 = 0.0
        down_absolute_60 = 0.0

    # Define output matrix
    output = pd.DataFrame({"MATURITY": np.arange(0, 120) + 1,
                           "DOWN_RELATIVE": np.nan,
                           "DOWN_ABSOLUTE": np.nan,
                           "UP_RELATIVE": np.nan,
                           "UP_ABSOLUTE": np.nan,
                           "FLOOR": floor})

    # Assignment
    output.iloc[0:20, 1] = down_relative_20
    output.iloc[0:20, 2] = down_absolute_20
    output.iloc[0:20, 3] = up_relative_20
    output.iloc[0:20, 4] = up_absolute_20

    # Specific points
    output.iloc[89:, 1] = down_relative_90
    output.iloc[59:, 2] = down_absolute_60
    output.iloc[89:, 3] = up_relative_90
    output.iloc[59:, 4] = up_absolute_60

    # Interpolate between different shocks
    output = output.interpolate(kind="linear")

    return output


def apply_shock(extrapolated_zero_curve, method="standard"):
    """
    Calculation of absolute rate shocks in up & down scenario of S2.

    Parameters
    ----------
    extrapolated_zero_curve: 1 dimensional Ndarray
        basic risk free term structure
    method: str
        One can decide to apply the standard rate shocks or the shocks as proposed by EIOPA in 2020

    Returns
    -------
    rate_shock_abs: pd.DataFrame
        Contains absolute rate upward & downward rate shock
    """

    # Get relative & absolute shocks
    shocks = get_eiopa_schock(method=method)

    if method == "standard":
        # Only rates greater than 0% should be shocked
        rate_floor = 0.0
        rate_floor_idx = extrapolated_zero_curve > rate_floor

        # Calculate shocked curves
        s2_spot_down = extrapolated_zero_curve * (1 - shocks["DOWN_RELATIVE"].values * rate_floor_idx * 1)
        s2_spot_up = extrapolated_zero_curve + np.clip(extrapolated_zero_curve * (shocks["UP_RELATIVE"].values), 0.01,
                                                       None)

    if method == "proposal":
        # Calculate shocked curves
        s2_spot_down = extrapolated_zero_curve * (1 - shocks["DOWN_RELATIVE"].values) - shocks["DOWN_ABSOLUTE"].values
        s2_spot_up = extrapolated_zero_curve * (1 + shocks["UP_RELATIVE"].values) + shocks["UP_ABSOLUTE"].values

    # Absolute rate shock
    rate_shock_down = s2_spot_down - extrapolated_zero_curve
    rate_shock_up = s2_spot_up - extrapolated_zero_curve

    # Output contains absolute rate shocks
    rate_shock_abs = pd.DataFrame({"abs_shock_down": rate_shock_down, "abs_shock_up": rate_shock_up})

    return rate_shock_abs
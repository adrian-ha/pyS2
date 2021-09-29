def dcf_to_zero(dcf):
    """
    Helper function transforms sorted discount factors to zero rates.

    :param dcf: discount factors
    :return: zero rates
    """
    num_dcf = len(dcf)
    zero_rates = num_dcf * [0]
    for index, dcf_ in enumerate(dcf):
        time_to_maturity = index + 1
        zero_rates[index] = (1. / dcf_) ** (1 / time_to_maturity) - 1
    return zero_rates


def zero_to_dcf(zero_rates):
    """
    Helper function transforms sorted zero rates to discount factors.

    :param zero_rates: zero rates
    :return: discount factors
    """
    num_rates = len(zero_rates)
    dcf = num_rates * [0]
    for index, rate in enumerate(zero_rates):
        time_to_maturity = index + 1
        dcf[index] = 1 / (1 + rate) ** time_to_maturity
    return dcf


def swap_to_dcf(swap_rates):
    """
    Helper function transforms sorted swap rates to discount factors.

    :param swap_rates: par swap rates
    :return: discount factors
    """
    num_rates = len(swap_rates)
    dcf = num_rates * [0]
    for index, rate in enumerate(swap_rates):
        if index == 0:
            dcf[index] = 1. / (1. + rate)
        else:
            dcf[index] = (1 - (rate * sum(dcf[0:index + 1]))) / (1. + rate)
    return dcf


def dcf_to_swap(dcf):
    """
    Helper function transforms sorted discount factors to swap rates.

    :param dcf: discount factors
    :return: par swap rates
    """
    num_dcf = len(dcf)
    swap_rates = num_dcf * [0]
    for index, dcf_ in enumerate(dcf):
        if index == 0:
            swap_rates[index] = (1 / dcf_) - 1
        else:
            swap_rates[index] = (1 - dcf_) / sum(dcf[0:index + 1])
    return swap_rates


def swap_to_zero(swap_rates):
    """
    Helper function transforms sorted swap rates to zero rates.

    :param swap_rates: par swap rates
    :return: zero rates
    """
    swap_rates = list(swap_rates)
    dcf = swap_to_dcf(swap_rates)
    zero_rates = dcf_to_zero(dcf)
    return zero_rates


def zero_to_swap(zero_rates):
    """
    Helper function transforms sorted zero rates to swap rates.

    :param zero_rates: zero rates
    :return: par swap rates
    """
    zero_rates = list(zero_rates)
    dcf = zero_to_dcf(zero_rates)
    swap_rates = dcf_to_swap(dcf)
    return swap_rates

import numpy as np


def SNR(pred, tg):
    assert pred.shape == tg.shape, 'Input shape should be the same'
    N = np.shape(pred)[0]
    Numerator = np.mean(np.abs(pred - tg) ** 2)
    Denominator = np.mean(np.abs(tg) ** 2)
    if Numerator == 0: return np.inf
    EVM = np.sqrt(Numerator / Denominator)
    SNR = 10 * np.log10(1 / EVM ** 2)

    EVM = round(EVM, 4)
    SNR = round(SNR, 4)
    return (SNR, EVM)

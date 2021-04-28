import numpy as np
import math


def SNR(pred, tg):
    assert pred.shape == tg.shape, 'Input shape should be the same'
    N = np.shape(pred)[0]
    Numerator = np.mean(np.abs(pred - tg) ** 2)
    Denominator = np.mean(np.abs(tg) ** 2)

    # Numerator = np.mean((((np.real(pred) - np.real(tg)) ** 2) + ((np.imag(pred) - np.imag(tg)) ** 2) ))
    # Denominator = np.abs(7 + 1j* 7) ** 2

    if Numerator == 0: return np.inf
    EVM = np.sqrt(Numerator / Denominator)
    SNR = 10 * np.log10(1 / EVM ** 2)
    # SNR = 10 * np.log10( np.mean(np.abs(tg) ** 2) / np.mean(np.abs(pred) ** 2) )

    EVM = round(EVM, 4)
    SNR = round(SNR, 4)
    # BER = 2 * (1 - 1 / 8) * math.erfc(np.sqrt(3 / 2 / (64 - 1) / (EVM*100) ** 2)) / np.log2(64)


    # return (SNR, EVM, BER)
    return (SNR, EVM)
import numpy as np

def Tx2Bit(Isig, Qsig, PAM_order):
    Tx_symbol = []
    seqlength = len(Isig)
    if PAM_order == 2: #level[-1, 1]
        Ithreshold = np.mean(Isig)
        Qthreshold = np.mean(Qsig)
        for indx in range(seqlength):
            if Isig[indx] > Ithreshold and Qsig[indx] > Qthreshold:
                Tx_symbol.append(1 + 1j)
            if Isig[indx] > Ithreshold and Qsig[indx] < Qthreshold:
                Tx_symbol.append(1 - 1j)
            if Isig[indx] < Ithreshold and Qsig[indx] > Qthreshold:
                Tx_symbol.append(-1 + 1j)
            if Isig[indx] < Ithreshold and Qsig[indx] < Qthreshold:
                Tx_symbol.append(-1 - 1j)
    if PAM_order == 4:
        Tx = np.concatenate((Isig, Qsig), axis=1)
        symboldic = {'[0, 0, 0, 0]': -3 + 3j, '[0, 0, 0, 1]': -1 + 3j, '[0, 0, 1, 1]': 3 + 3j, '[0, 0, 1, 0]': 1 + 3j,
                     '[0, 1, 0, 0]': -3 + 1j, '[0, 1, 0, 1]': -1 + 1j, '[0, 1, 1, 1]': 3 + 1j, '[0, 1, 1, 0]': 1 + 1j,
                     '[1, 1, 0, 0]': -3 - 3j, '[1, 1, 0, 1]': -1 - 3j, '[1, 1, 1, 1]': 3 - 3j, '[1, 1, 1, 0]': 1 - 3j,
                     '[1, 0, 0, 0]': -3 - 1j, '[1, 0, 0, 1]': -1 - 1j, '[1, 0, 1, 1]': 3 - 1j, '[1, 0, 1, 0]': 1 - 1j}
        for indx in range(seqlength):
            Tx_symbol.append(symboldic[str(Tx[indx][:].tolist())])
    return Tx_symbol












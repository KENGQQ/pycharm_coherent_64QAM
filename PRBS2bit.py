import numpy as np

def PRBS2bit(TxI_prbs, TxQ_prbs = []):
    Tx_Prbs = []
    seqlength = len(TxI_prbs)
    bitdic = {(0, 1): 1, (1, 0): -1, (1, 1): 3, (0, 0): -3}
    if(TxQ_prbs == []):
        for indx in range(seqlength):
            Tx_Prbs.append(bitdic[tuple(TxI_prbs[indx])])
        return Tx_Prbs
    else:
        assert len(TxI_prbs) == len(TxQ_prbs), 'Input length should be the same'
        for indx in range(seqlength):
            Tx_Prbs.append(bitdic[tuple(TxI_prbs[indx])] + 1j * bitdic[tuple(TxQ_prbs[indx])])
        return Tx_Prbs

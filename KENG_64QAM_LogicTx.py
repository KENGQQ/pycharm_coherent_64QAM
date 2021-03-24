import numpy as np


def QAM64_LogicTx(Logic_Ichannel_LSB, Logic_Ichannel_CSB ,Logic_Ichannel_MSB,Logic_Qchannel_LSB ,Logic_Qchannel_CSB,Logic_Qchannel_MSB):

    assert((len(Logic_Ichannel_LSB) == len(Logic_Ichannel_CSB)) == (len(Logic_Ichannel_LSB) ==len(Logic_Ichannel_MSB)))
    assert((len(Logic_Qchannel_LSB) == len(Logic_Qchannel_CSB)) == (len(Logic_Qchannel_CSB) ==len(Logic_Qchannel_MSB)))

    TxI = np.zeros(len(Logic_Ichannel_LSB))
    TxQ = np.zeros(len(Logic_Qchannel_LSB))

    TxI_Log = np.vstack([Logic_Ichannel_LSB, Logic_Ichannel_CSB, Logic_Ichannel_MSB])
    TxQ_Log = np.vstack([Logic_Qchannel_LSB, Logic_Qchannel_CSB, Logic_Qchannel_MSB])



    Tx_dict = {'[0. 0. 0.]' : -7 ,'[1. 0. 0.]' : -5 ,'[1. 1. 0.]' : -3 ,'[0. 1. 0.]' : -1 ,\
               '[0. 1. 1.]' :  1 ,'[1. 1. 1.]' :  3 ,'[1. 0. 1.]' :  5 ,'[0. 0. 1.]' :  7 ,}

    for i in range(len(Logic_Ichannel_LSB)):
        TxI[i] = Tx_dict[str(TxI_Log[:, i])]
        TxQ[i] = Tx_dict[str(TxQ_Log[:, i])]

    return TxI, TxQ
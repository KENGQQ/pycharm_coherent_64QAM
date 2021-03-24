import numpy as np
import cmath

def IQimbaCompensator(isig, stepsize):
    isig = np.array(isig)
    if np.shape(isig) == (len(isig), ):
        print("reshape input data")
        isig = np.reshape((1,len(isig)))
    datalength = np.shape(isig)[1]
    batchnum = np.shape(isig)[0]
    rx_iq_imba = np.zeros((batchnum, datalength), dtype="complex_")
    weight = np.zeros((batchnum, datalength+1), dtype="complex_")
    for batch in range(batchnum):
        for indx in range(datalength):
            rx_iq_imba[batch][indx] = isig[batch][indx] + weight[batch][indx] * np.conjugate(isig[batch][indx])
            weight[batch][indx + 1] = weight[batch][indx] - stepsize * rx_iq_imba[batch][indx] ** 2

    return rx_iq_imba
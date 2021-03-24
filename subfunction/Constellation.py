import numpy as np
import math
import cmath


def Constellation(isig , qsig, pamorder):
    constellation_data = []
    datalength = len(isig)
    meani = np.mean(isig)
    meanq = np.mean(qsig)
    for indx in range(datalength):
        tangent = qsig[indx]/isig[indx]
        arg = np.arctan(tangent)
        if isig[indx] < meani and qsig[indx] < meanq:
            constellation_data.append(
                cmath.sqrt(isig[indx] ** 2 + qsig[indx] ** 2) * cmath.exp(1j * (arg+ cmath.pi)))
        elif isig[indx] < meani and qsig[indx] > meanq:
            constellation_data.append(
                cmath.sqrt(isig[indx] ** 2 + qsig[indx] ** 2) * cmath.exp(1j * (arg + cmath.pi)))
        elif isig[indx] > meani and qsig[indx] < meanq:
            constellation_data.append(
                cmath.sqrt(isig[indx] ** 2 + qsig[indx] ** 2) * cmath.exp(1j * arg))
        elif isig[indx] > meani and qsig[indx] > meanq:
            constellation_data.append(
                cmath.sqrt(isig[indx] ** 2 + qsig[indx] ** 2) * cmath.exp(1j * arg))
    return constellation_data



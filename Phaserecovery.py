import numpy as np
import cmath

class Phaserecovery:
    def __init__(self, isig):
        self.isig = np.array(isig)
        self.tapscan = [3, 5, 19, 21, 31]
        self.taps = self.tapscan[2]
        self.center = int((self.taps-1)/2)

    def V_Valg(self):
        if np.shape(self.isig) == (len(self.isig), ):
            print("reshape input data")
            self.isig = np.reshape((1, len(self.isig)))
        batchnum = np.shape(self.isig)[0]
        datalength = np.shape(self.isig)[1]
        self.rx_recovery = np.zeros((batchnum, datalength), dtype="complex_")
        for batch in range(batchnum):
            phase = np.zeros((datalength, 1))
            phaseadj = np.zeros((datalength, 1))
            ak = np.zeros((datalength, 1))
            for indx in range(self.center, datalength-self.center):
                phase[indx] = (cmath.phase(
                    np.sum(self.isig[batch][indx - self.center:indx + self.center + 1] ** 4)) - cmath.pi) / 4
                ak[indx] = ak[indx - 1] + np.floor(0.5 - 4 * (phase[indx] - phase[indx - 1]) / (2 * cmath.pi))
                phaseadj[indx] = phase[indx] + ak[indx] * 2 * cmath.pi / 4
                self.rx_recovery[batch][indx] = self.isig[batch][indx] * cmath.exp(-1j * phaseadj[indx])
        self.rx_recovery = self.rx_recovery[:, self.center:-1-self.center+1]
        return self.rx_recovery


    def DD_PLL(self):
        bandwidth = 0.01
        dampingfactor = 0.707
        theta = bandwidth / (dampingfactor + 1 / (4 * dampingfactor))
        d = 1 + 2 * dampingfactor * theta + theta ** 2
        Kp = 2
        K0 = 1
        g1 = 4 * theta ** 2 / (K0 * Kp * d)
        gp = 4 * dampingfactor * theta / (K0 * Kp * d)
        if np.shape(self.isig) == (len(self.isig),):
            print("reshape input data")
            self.isig = np.reshape((1, len(self.isig)))
        batchnum = np.shape(self.isig)[0]
        datalength = np.shape(self.isig)[1]
        self.rx_recovery = np.zeros((batchnum, datalength), dtype="complex_")
        for batch in range(batchnum):
            err = np.zeros((datalength, 1))
            weight = np.zeros((datalength, 1))
            lamb = np.zeros((datalength, 1))
            self.rx_recovery[batch][0] = self.isig[batch][0] * cmath.exp(-1j * lamb[0])
            err[0] = np.sign(np.real(self.rx_recovery[batch][0])) * np.imag(self.rx_recovery[batch][0]) - np.sign(
                np.imag(self.rx_recovery[batch][0])) * np.real(self.rx_recovery[batch][0])
            weight[0] = err[0] * g1
            for it in range(3):
                for indx in range(1, datalength):
                    lamb[indx] = gp * err[indx - 1] + weight[indx - 1] + lamb[indx - 1]
                    self.rx_recovery[batch][indx] = self.isig[batch][indx] * cmath.exp(-1j * lamb[indx])
                    err[indx] = np.sign(np.real(self.rx_recovery[batch][indx])) * np.imag(
                        self.rx_recovery[batch][indx]) - np.sign(np.imag(self.rx_recovery[batch][indx])) * np.real(
                        self.rx_recovery[batch][indx])
                    weight[indx] = g1 * err[indx] + weight[indx - 1]
        self.rx_recovery = self.rx_recovery[:, 1:]
        return self.rx_recovery

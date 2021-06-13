# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:12:48 2020

@author: KENG
"""
import numpy as np
import cmath
import math
import matplotlib.pyplot as plt
from scipy import fft


class KENG_phaserecovery:
    def __init__(self):
        pass
    def first_QPSK(self, Rx, tap):
        self.tap = tap
        self.center = int((tap - 1) / 2)
        Rx = np.reshape(np.matrix(Rx, dtype='complex_'), [np.size(Rx), 1])
        length = int(np.size(Rx))

        PLL_Rx = np.zeros([length, 1], dtype='complex_')
        ak = np.zeros([1, length])
        self.phase = np.zeros([1, length])
        self.phase_adj = np.zeros([1, length])

        for i in range(self.center, length - self.center):  # i ===center
            Rx_tmp = np.array(Rx[i - self.center:i + self.center + 1, 0])
            self.phase[0, i] = (cmath.phase(np.sum(Rx_tmp) / self.tap) - cmath.pi) / 4
            ak[0, i] = ak[0, i - 1] + math.floor(0.5 - 4 * (self.phase[0, i] - self.phase[0, i - 1]) / 2 / cmath.pi)
            self.phase_adj[0, i] = self.phase[0, i] + ak[0, i] * 2 * cmath.pi / 4
            PLL_Rx[i] = Rx[i, 0] * cmath.exp(-1j * self.phase_adj[0, i])

        return PLL_Rx

    def forth_QPSK(self, Rx, tap):
        self.tap = tap
        self.center = int((tap - 1) / 2)
        Rx = np.reshape(np.matrix(Rx, dtype='complex_'), [np.size(Rx), 1])
        length = int(np.size(Rx))

        PLL_Rx = np.zeros([length, 1], dtype='complex_')
        ak = np.zeros([1, length])
        self.phase = np.zeros([1, length])
        self.phase_adj = np.zeros([1, length])

        for i in range(self.center, length - self.center):  # i ===center
            Rx_tmp = np.array(Rx[i - self.center:i + self.center + 1, 0])
            self.phase[0, i] = (cmath.phase(np.sum(Rx_tmp ** 4) / self.tap) - cmath.pi) / 4
            ak[0, i] = ak[0, i - 1] + math.floor(0.5 - 4 * (self.phase[0, i] - self.phase[0, i - 1]) / 2 / cmath.pi)
            self.phase_adj[0, i] = self.phase[0, i] + ak[0, i] * 2 * cmath.pi / 4
            PLL_Rx[i] = Rx[i, 0] * cmath.exp(-1j * self.phase_adj[0, i])

        return PLL_Rx

    def eighth_QPSK(self, Rx, tap):
        self.tap = tap
        self.center = int((tap - 1) / 2)
        Rx = np.reshape(np.matrix(Rx, dtype='complex_'), [np.size(Rx), 1])
        length = int(np.size(Rx))

        PLL_Rx = np.zeros([length, 1], dtype='complex_')
        ak = np.zeros([1, length])
        self.phase = np.zeros([1, length])
        self.phase_adj = np.zeros([1, length])

        for i in range(self.center, length - self.center):  # i ===center
            Rx_tmp = np.array(Rx[i - self.center:i + self.center + 1, 0])
            self.phase[0, i] = (cmath.phase(np.sum(Rx_tmp ** 8) / self.tap) - cmath.pi) / 8
            ak[0, i] = ak[0, i - 1] + math.floor(0.5 - 8 * (self.phase[0, i] - self.phase[0, i - 1]) / 2 / cmath.pi)
            self.phase_adj[0, i] = self.phase[0, i] + ak[0, i] * 2 * cmath.pi / 8
            PLL_Rx[i] = Rx[i, 0] * cmath.exp(-1j * self.phase_adj[0, i])

        return PLL_Rx

    def QAM_64QAM_1(self, Rx, r1_o, r3_i, r3_o, r9_i):
        # self.c1_radius_i = 0
        # self.c1_radius_o = 1.55
        # self.c3_radius_i = 3.2
        # self.c3_radius_o = 4
        # self.c9_radius_i = 8.2
        # self.c9_radius_o = 12
        self.c1_radius_i = 0
        self.c1_radius_o = r1_o
        self.c3_radius_i = r3_i
        self.c3_radius_o = r3_o
        self.c9_radius_i = r9_i
        self.c9_radius_o = 15
        self.tap = 301
        # self.tap = 101
        # self.tap = 201


        Rx_amplitude = np.abs(Rx)
        Rx_zeropad = np.zeros(len(Rx), dtype = "complex_")

        for i in range(len(Rx_zeropad)):
            if (self.c1_radius_i <= Rx_amplitude[i]) and (Rx_amplitude[i] < self.c1_radius_o):
                Rx_zeropad[i] = Rx[i]
            if (self.c3_radius_i <= Rx_amplitude[i]) and (Rx_amplitude[i] < self.c3_radius_o):
                Rx_zeropad[i] = Rx[i]
            if (self.c9_radius_i <= Rx_amplitude[i]) and (Rx_amplitude[i] < self.c9_radius_o):
                Rx_zeropad[i] = Rx[i]

        a = KENG_phaserecovery()
        Rx_vv = a.forth_QPSK(Rx_zeropad, tap = self.tap)
        phase1 = a.phase_adj

        Rx_ph = np.zeros(int(np.size(phase1)), dtype='complex_')
        for i in range(np.size(Rx_ph)):
            Rx_ph[i] = Rx[i] * cmath.exp(-1j * phase1[0, i])

        return Rx_ph

    def Rotation_algorithm(self, Rx):    #Carrier Phase Estimation Through the Rotation Algorithm for 64-QAM Optical Systems

        Rx_RA = Rx ** 4
        self.Rx_RA_outside = Rx_RA
        c1_theta = cmath.phase(1 + 1j) - cmath.phase(1 + 1j)
        c2_theta = cmath.phase(1 + 1j) - cmath.phase(3 + 1j)
        c3_theta = cmath.phase(1 + 1j) - cmath.phase(1 + 1j)
        c4_theta = cmath.phase(1 + 1j) - cmath.phase(5 + 1j)
        c5_theta = cmath.phase(1 + 1j) - cmath.phase(5 + 3j)
        c6_theta = cmath.phase(1 + 1j) - cmath.phase(7 + 1j)
        c7_theta = cmath.phase(1 + 1j) - cmath.phase(1 + 1j)
        c8_theta = cmath.phase(1 + 1j) - cmath.phase(7 + 3j)
        c9_theta = cmath.phase(1 + 1j) - cmath.phase(7 + 5j)
        c10_theta = cmath.phase(1 + 1j) - cmath.phase(1 + 1j)
        c_theta = [c1_theta ,c2_theta ,c3_theta ,c4_theta ,c5_theta,c6_theta ,c7_theta ,c8_theta ,c9_theta ,c10_theta]

        # c_theta = [c1_theta ,c2_theta ,c3_theta ,c4_theta ,c5_theta,c6_theta ,c7_theta ,c8_theta ,c9_theta ,c10_theta]

        c0_radius_o = 0
        # c1c2_radius_i = 0
        c1_radius_o = 65
        # c2_radius_i = 0
        c2_radius_o = 250
        # c3_radius_i = 0
        c3_radius_o = 500
        # c4_radius_i = 0
        c4_radius_o = 750
        # c5_radius_i = 0
        c5_radius_o = 2050
        # c6_radius_i = 0
        c6_radius_o = 2400
        # c7_radius_i = 0
        c7_radius_o = 5200
        # c8_radius_i = 0
        c8_radius_o = 6300
        # c9_radius_i = 0
        c9_radius_o = 16000
        # c10_radius_i = 0
        # c10_radius_o = 65
        radius_o = [c0_radius_o, c1_radius_o, c2_radius_o, c3_radius_o, c4_radius_o, c5_radius_o, c6_radius_o, c7_radius_o, c8_radius_o, c9_radius_o]
        # radius_o = [ c1_radius_o, c2_radius_o, c3_radius_o, c4_radius_o, c5_radius_o, c6_radius_o, c7_radius_o, c8_radius_o, c9_radius_o]

        for i in range(len(Rx_RA)):
            for j in range(1, 6):
                if radius_o[j - 1] <= abs(Rx_RA[i]) < radius_o[j]:
                    Rx_RA[i] = Rx_RA[i] * np.exp(4j * c_theta[j - 1] * np.sign(np.imag(Rx_RA[i])))
                    break

        for i in range(len(Rx_RA)):
            if radius_o[5] <= abs(Rx_RA[i]) < radius_o[6]:
                if np.real(Rx_RA[i]) > 0 :
                    Rx_RA[i] = Rx_RA[i] * np.exp(4j * c_theta[5] * np.sign(np.imag(Rx_RA[i])))
                else:
                    Rx_RA[i] = Rx_RA[i] * np.exp(4j * c_theta[6] * np.sign(np.imag(Rx_RA[i])))

        for i in range(len(Rx_RA)):
            for j in range(7, 10):
                if radius_o[j - 1] <= abs(Rx_RA[i]) < radius_o[j]:
                    Rx_RA[i] = Rx_RA[i] * np.exp(4j * c_theta[j] * np.sign(np.imag(Rx_RA[i])))
                    break

        a = KENG_phaserecovery()
        Rx_vv = a.first_QPSK(Rx_RA, tap = 201)
        phase1 = a.phase_adj

        Rx_ph = np.zeros(int(np.size(phase1)), dtype='complex_')
        for i in range(np.size(Rx_ph)):
            Rx_ph[i] = Rx[i] * cmath.exp(-1j * phase1[0, i])

        return Rx_ph

    def PLL(self, isig):
        self.isig = isig
        # self.bandwidth = 2e-3
        self.bandwidth = 1e-4
        dampingfactor = 1430.707

        theta = self.bandwidth / (dampingfactor + 1 / (4 * dampingfactor))
        d = 1 + 2 * dampingfactor * theta + theta ** 2
        Kp = 2
        K0 = 1
        g1 = 4 * theta ** 2 / (K0 * Kp * d)
        gp = 4 * dampingfactor * theta / (K0 * Kp * d)
        if np.shape(self.isig) == (len(self.isig),):
            # print("reshape input data")
            self.isig = np.reshape(self.isig, (1, len(self.isig)))
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

    def FreqOffsetComp(self, rx, fsamp=56e9,fres=1e7):  # CoarseFrequencyOffset compensation based on prediogram method for M-QAM
        fr = fres;
        fs = fsamp;
        m_order = 4;  # QAM is 4
        Nfft = int(2 ** np.log2(fs / fr));
        if Nfft > 1e7:
            Nfft = 2 ** 23  # avoid memory error
        elif Nfft < 1e3:
            Nfft = 2 ** 10
        N = len(rx);
        # raiseSig=rx**m_order
        absFFTSig = abs(np.fft.fft(rx ** m_order, Nfft))
        # plt.plot(absFFTSig/max(absFFTSig));plt.show()
        maxIndex = np.argmax(absFFTSig)
        estFreqOffset = fs / Nfft * (maxIndex - 1) / m_order;

        # print(estFreqOffset)
        if maxIndex > Nfft / 2:
            maxIndex = maxIndex - Nfft
        estFreqOffset = fs / Nfft * (maxIndex - 1) / m_order;
        print('Frequency offset={} GHz'.format(np.round(estFreqOffset/ 1e9, 5)))
        rx = rx * np.exp(-estFreqOffset * np.linspace(1, N, N) * 2 * np.pi * 1j / fs)
        # rx = rx * np.exp(estFreqOffset * np.linspace(1, N, N) * 2 * np.pi * 1j / fs)

        return rx

    def FreqOffsetComp_2(self, rx, fsamp=25e9, fres=1e6):
        fs = fsamp
        # fr = fres
        m_order = 4
        N_coarse = len(rx)
        absFFTSig = (abs(np.fft.fft(rx ** m_order, N_coarse)))
        maxIndex = np.argmax(absFFTSig)

        estFreqOffset = fs / N_coarse * (maxIndex - 1) / m_order
        # print(estFreqOffset)

        if maxIndex > N_coarse / 2:
            maxIndex = maxIndex - N_coarse
        estFreqOffset = fs / N_coarse * (maxIndex - 1) / m_order
        print('Frequency offset=', estFreqOffset)

        rx = rx * np.exp(-estFreqOffset * np.linspace(1, N_coarse, N_coarse) * 2 * np.pi * 1j / fs)

        #
        # superposition = np.zeros(512)
        # N_fine = 512
        # iter = 100
        # for i in range(iter):
        #     absFFTSig_fine = (abs(np.fft.fft(rx[100*iter:100*(iter+1)] ** m_order, N_fine)))**2
        #     superposition = superposition + absFFTSig_fine
        #
        # plt.plot(superposition/max(superposition));plt.show()
        #
        # maxIndex_fine = np.argmax(superposition)
        # print(maxIndex_fine)
        # if maxIndex_fine > N_coarse / 2:
        #     maxIndex_fine = maxIndex_fine - N_fine
        # estFreqOffset = fs / N_coarse * (maxIndex_fine - 1) / m_order
        # rx = rx * np.exp(-estFreqOffset * np.linspace(1, N_coarse, N_coarse) * 2 * np.pi * 1j / fs)

        return rx




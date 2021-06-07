import numpy as np
import cmath
import math


class CD_compensator:
    def __init__(self, RxX, RxY, Gbaud, KM):
        self.KM = KM
        self.Gbaud = Gbaud

        self.rx_x = np.array(RxX)
        self.rx_y = np.array(RxY)
        self.datalength = len(RxX)


    def FIR_CD(self):
        c = 3e17     #nm/s
        T = 1 / self.Gbaud
        wavelength = 1553     #nm
        D = 16e-12   # s / nm /km

        N = np.floor(D * self.KM * (wavelength ** 2) / 2 / c / T ** 2)
        tap = int(N * 2 + 1)
        print("FIR_CD_tap_needed : {}".format(tap))
        center = int((tap - 1) / 2)

        self.ak = np.zeros(int(tap), dtype="complex_")
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")

        inputrx = self.rx_x
        inputry = self.rx_y

        tt = -N

        for k in range(0, tap):
            self.ak[k] = np.sqrt(1j * c * (1 / self.Gbaud) ** 2 / D / (wavelength ** 2) / self.KM) * \
                    np.exp(-1j * math.pi * c * (1 / self.Gbaud) ** 2 * tt ** 2 / D / self.KM / (wavelength ** 2))
            tt+=1
        #
        for indx in range(center, self.datalength - center):
            exout[indx] = np.matmul(self.ak, inputrx[indx - center : indx + center + 1])
            eyout[indx] = np.matmul(self.ak, inputry[indx - center : indx + center + 1])
        return exout, eyout

    def overlap_save(self, Nfft, NOverlap):
        c = 3e17  # nm/s
        T = 1 / (self.Gbaud)
        q = np.linspace(int(-Nfft / 2), int(Nfft / 2) - 1, Nfft)
        w = 2 * math.pi * (1 / T)  # angular freq
        wn = 2 * math.pi * (1 / T / 2)
        wavelength = 1553  # nm
        D = 16e-12  # s / nm /km
        N = 2 * np.ceil(np.sqrt((math.pi ** 2) * (c ** 2) * (T ** 4) + 4 * (wavelength ** 4) * (D ** 2) * (self.KM ** 2)) / math.pi / c / (T ** 2)) + 2
        N = int(N + N % 2)
        print("FFT_CD_tap_needed : {}".format(N))

        filter = np.exp(1j * D * (wavelength ** 2) * ((q * w / Nfft) ** 2) * self.KM / 4 / math.pi / c)
        FDE = filter

        L = int(Nfft - NOverlap)
        minimum_tap = np.ceil(6.67 * abs(wavelength ** 2 * D / 2 / math.pi / c) * self.KM * (self.Gbaud ** 2))
        print("FFT_CD_tap_minimum : {}".format(int(minimum_tap)))

        NExtra = 0
        AuxLen = self.datalength / L
        if(AuxLen != np.ceil(AuxLen)):
            NExtra = int(np.ceil(AuxLen) * L - self.datalength)
            self.rx_x = np.concatenate((self.rx_x[int(np.floor(-NExtra / 2)):], self.rx_x, self.rx_x[: int(NExtra / 2)]),axis=None)
            self.rx_y = np.concatenate((self.rx_y[int(np.floor(-NExtra / 2)):], self.rx_y, self.rx_y[: int(NExtra / 2)]),axis=None)
            self.datalength = len(self.rx_x)

        Blocks_X = np.reshape(self.rx_x, [int(self.datalength / L), L])
        Blocks_Y = np.reshape(self.rx_y, [int(self.datalength / L), L])

        outX = np.zeros(np.shape(Blocks_X), dtype='complex_') ; Overlap_X = np.zeros(NOverlap, dtype='complex_')
        outY = np.zeros(np.shape(Blocks_Y), dtype='complex_') ; Overlap_Y = np.zeros(NOverlap, dtype='complex_')

        for k in range(int(self.datalength / L)):
            InB_X = np.concatenate((Overlap_X, Blocks_X[k, :]), axis=None)
            InB_Y = np.concatenate((Overlap_Y, Blocks_Y[k, :]), axis=None)

            Overlap_X = InB_X[-NOverlap:]
            Overlap_Y = InB_Y[-NOverlap:]

            OutB_X = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(InB_X)) * FDE))
            OutB_Y = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(InB_Y)) * FDE))

            outX[k, :] = OutB_X[int(NOverlap/2): int(-NOverlap/2)]
            outY[k, :] = OutB_Y[int(NOverlap/2): int(-NOverlap/2)]

        outX = np.reshape(outX, -1)
        outY = np.reshape(outY, -1)

        if (NExtra == 0):
            return outX, outY
        else :
            return outX[int(np.ceil(NExtra / 2)) + int(NOverlap / 2) : int(-NExtra / 2) - int(NOverlap / 2)], \
                   outY[int(np.ceil(NExtra / 2)) + int(NOverlap / 2) : int(-NExtra / 2) - int(NOverlap / 2)]

        # return outX[int(Nfft/2): int(np.floor(-Nfft/2))], outY[int(Nfft/2): int(np.floor(-Nfft/2))]
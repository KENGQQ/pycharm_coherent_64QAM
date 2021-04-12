import numpy as np
import cmath
import math
# from numba import jit,jitclass
import datetime
import numba as nb
from itertools import combinations_with_replacement


class CMA:
    def __init__(self, rx_x, rx_y, mean = True):
        self.mean = mean
        self.rx_x_single = np.array(rx_x)
        self.rx_y_single = np.array(rx_y)
        self.rx_x = np.array(rx_x)
        self.rx_y = np.array(rx_y)
        self.datalength = len(rx_x)
        self.stepsizelist = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 5e-6]
        # self.batchsize = 32767*2
        self.batchsize = self.datalength
        self.overhead = 1
        self.trainlength = round(self.batchsize * self.overhead)
        self.cmataps = 47
        self.center = int((self.cmataps - 1) / 2)
        self.batchnum = int(self.datalength / self.batchsize)
        self.iterator = 20
        self.earlystop = 0.0001
        self.stepsizeadjust = 1
        self.rx_x.resize((self.batchnum, self.batchsize), refcheck=False)
        self.rx_y.resize((self.batchnum, self.batchsize), refcheck=False)
        self.stepsize = self.stepsizelist[4]

    def run(self):
        self.costfunx = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        for batch in range(self.batchnum):
            # initialize H
            inputrx = self.rx_x[batch]
            inputry = self.rx_y[batch]
            hxx = np.zeros(self.cmataps, dtype="complex_")
            hxy = np.zeros(self.cmataps, dtype="complex_")
            hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.trainlength + self.center, dtype="complex_")
            eyout = np.zeros(self.trainlength + self.center, dtype="complex_")
            errx = np.zeros(self.trainlength + self.center, dtype="complex_")
            erry = np.zeros(self.trainlength + self.center, dtype="complex_")
            for it in range(self.iterator):
                for indx in range(self.center, self.center + self.trainlength):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    errx[indx] = (2 - (np.abs(exout[indx])) ** 2)
                    erry[indx] = (2 - (np.abs(eyout[indx])) ** 2)
                    hxx = hxx + self.stepsize * errx[indx] * exout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy + self.stepsize * errx[indx] * exout[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx + self.stepsize * erry[indx] * eyout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy + self.stepsize * erry[indx] * eyout[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                self.costfunx[batch][it] = np.mean((errx[self.center:]) ** 2)
                self.costfuny[batch][it] = np.mean((erry[self.center:]) ** 2)
                if it > 1:
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
                            self.costfuny[batch][it]:
                        print("Earlybreak at iterator {}".format(it))
                        break
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfuny[batch][it]:
                        self.stepsize *= 0.5
                        print('Stepsize adjust to {}'.format(self.stepsize))
            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def run_single(self):
        # initialize H
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxy = np.zeros(self.cmataps, dtype="complex_")
        hyx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1 + 0j
        hyy[self.center] = 1 + 0j
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                errx[indx] = (1 - (np.abs(exout[indx])) ** 2)
                erry[indx] = (1 - (np.abs(eyout[indx])) ** 2)
                hxx = hxx + self.stepsize * errx[indx] * exout[indx] * np.conjugate(
                    inputrx[indx - self.center:indx + self.center + 1])
                hxy = hxy + self.stepsize * errx[indx] * exout[indx] * np.conjugate(
                    inputry[indx - self.center:indx + self.center + 1])
                hyx = hyx + self.stepsize * erry[indx] * eyout[indx] * np.conjugate(
                    inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy + self.stepsize * erry[indx] * eyout[indx] * np.conjugate(
                    inputry[indx - self.center:indx + self.center + 1])
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            if it > 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break

        self.rx_x_single = exout
        self.rx_y_single = eyout

    def run_16qam(self):
        self.costfunx = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        for batch in range(self.batchnum):
            # initialize H
            inputrx = self.rx_x[batch]
            inputry = self.rx_y[batch]
            hxx = np.zeros(self.cmataps, dtype="complex_")
            hxy = np.zeros(self.cmataps, dtype="complex_")
            hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.trainlength + self.center, dtype="complex_")
            eyout = np.zeros(self.trainlength + self.center, dtype="complex_")
            errx = np.zeros(self.trainlength + self.center, dtype="complex_")
            erry = np.zeros(self.trainlength + self.center, dtype="complex_")
            for it in range(self.iterator):
                for indx in range(self.center, self.center + self.trainlength):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    exout_adj = np.real(exout[indx]) - 2 * np.sign(np.real(exout[indx])) + 1j * (
                            np.imag(exout[indx]) - 2 * np.sign(np.imag(exout[indx])))
                    eyout_adj = np.real(eyout[indx]) - 2 * np.sign(np.real(eyout[indx])) + 1j * (
                            np.imag(eyout[indx]) - 2 * np.sign(np.imag(eyout[indx])))
                    exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                    eyout_adj2 = eyout[indx] - 2 * np.sign(np.real(eyout[indx])) - 2j * (np.sign(np.imag(eyout[indx])))
                    R = 2
                    errx[indx] = ((np.abs(exout_adj)) ** 2 - R)
                    erry[indx] = ((np.abs(eyout_adj)) ** 2 - R)
                    ################
                    # inx=inputrx
                    # iny=inputry
                    # inx=inputrx[indx - self.center:indx + self.center + 1]
                    # iny=inputry[indx - self.center:indx + self.center + 1]
                    # squ_Rx=np.mean(abs(inx-2*np.sign(np.real(inx))-2j*np.sign(np.imag(inx)))**4)\
                    #       /np.mean(abs(inx-2*np.sign(np.real(inx))-2j*np.sign(np.imag(inx)))**2)
                    # squ_Ry=np.mean(abs(iny-2*np.sign(np.real(iny))-2j*np.sign(np.imag(iny)))**4)\
                    #       /np.mean(abs(iny-2*np.sign(np.real(iny))-2j*np.sign(np.imag(iny)))**2)

                    # errx[indx] = ((np.abs(exout_adj)) ** 2 - squ_Rx)
                    # erry[indx] = ((np.abs(eyout_adj)) ** 2 - squ_Ry)
                    hxx = hxx - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    ####
                self.costfunx[batch][it] = np.mean((errx[self.center:]) ** 2)
                self.costfuny[batch][it] = np.mean((erry[self.center:]) ** 2)
                print(self.costfunx[batch][it])
                if it > 1:
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
                            self.costfuny[batch][it]:
                        print("Earlybreak at iterator {}".format(it))
                        break
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfuny[batch][it]:
                        self.stepsize *= self.stepsizeadjust
                        print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def qam_3_butter_conj(self):
        self.costfunx = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize - 2 * self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize - 2 * self.center), dtype="complex_")
        for batch in range(self.batchnum):
            inputrx = self.rx_x[batch]
            inputry = self.rx_y[batch]
            hxx = np.zeros(self.cmataps, dtype="complex_")
            hxy = np.zeros(self.cmataps, dtype="complex_")
            hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.trainlength, dtype="complex_")
            eyout = np.zeros(self.trainlength, dtype="complex_")
            errx = np.zeros(self.trainlength, dtype="complex_")
            erry = np.zeros(self.trainlength, dtype="complex_")
            exiout_adj = np.zeros(self.trainlength, dtype="complex_")
            exqout_adj = np.zeros(self.trainlength, dtype="complex_")
            eyiout_adj = np.zeros(self.trainlength, dtype="complex_")
            eyqout_adj = np.zeros(self.trainlength, dtype="complex_")
            squ_Rxi = np.zeros(self.trainlength, dtype="complex_")
            squ_Rxq = np.zeros(self.trainlength, dtype="complex_")
            squ_Ryi = np.zeros(self.trainlength, dtype="complex_")
            squ_Ryq = np.zeros(self.trainlength, dtype="complex_")
            cost_x = np.zeros(self.trainlength, dtype="complex_")
            cost_y = np.zeros(self.trainlength, dtype="complex_")
            for it in range(self.iterator):
                for indx in range(self.center, self.trainlength - self.center):
                    # if it ==0:
                    inxi = np.real(inputrx[indx - self.center:indx + self.center + 1])
                    inxq = np.imag(inputrx[indx - self.center:indx + self.center + 1])
                    inyi = np.real(inputry[indx - self.center:indx + self.center + 1])
                    inyq = np.imag(inputry[indx - self.center:indx + self.center + 1])
                    # else:
                    #     inxi=np.real(exout[indx - self.center :indx + self.center + 1])
                    #     inxq=np.imag(exout[indx - self.center :indx + self.center + 1])
                    #     inyi=np.real(eyout[indx - self.center :indx + self.center + 1])
                    #     inyq=np.imag(eyout[indx - self.center :indx + self.center + 1])
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                    exiout_adj[indx] = np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])) - 2 * \
                                       np.sign(np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])))
                    exqout_adj[indx] = np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])) - 2 * \
                                       np.sign(np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])))
                    eyiout_adj[indx] = np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])) - 2 * \
                                       np.sign(np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])))
                    eyqout_adj[indx] = np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])) - 2 * \
                                       np.sign(np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])))

                    exiout_adj[indx] = np.conj(np.real(exout[indx]))
                    exqout_adj[indx] = np.conj(np.imag(exout[indx]))
                    eyiout_adj[indx] = np.conj(np.real(eyout[indx]))
                    eyqout_adj[indx] = np.conj(np.imag(eyout[indx]))

                    inxi = inxi - 4 * np.sign(inxi) - 2 * np.sign(inxi - 4 * np.sign(inxi))
                    inxq = inxq - 4 * np.sign(inxq) - 2 * np.sign(inxq - 4 * np.sign(inxq))
                    inyi = inyi - 4 * np.sign(inyi) - 2 * np.sign(inyi - 4 * np.sign(inyi))
                    inyq = inyq - 4 * np.sign(inyq) - 2 * np.sign(inyq - 4 * np.sign(inyq))

                    squ_Rxi[indx] = np.mean(abs(inxi) ** 4) / np.mean(abs(inxi) ** 2)
                    squ_Rxq[indx] = np.mean(abs(inxq) ** 4) / np.mean(abs(inxq) ** 2)
                    squ_Ryi[indx] = np.mean(abs(inyi) ** 4) / np.mean(abs(inyi) ** 2)
                    squ_Ryq[indx] = np.mean(abs(inyq) ** 4) / np.mean(abs(inyq) ** 2)

                    squ_Rxi[indx] = 10
                    squ_Rxq[indx] = 10
                    squ_Ryi[indx] = 10
                    squ_Ryq[indx] = 10

                    errx[indx] = np.real(exout[indx]) * (abs(squ_Rxi[indx]) - np.abs(exiout_adj[indx]) ** 2) + \
                                 1j * np.imag(exout[indx]) * (abs(squ_Rxq[indx]) - np.abs(exqout_adj[indx]) ** 2)
                    erry[indx] = np.real(eyout[indx]) * (abs(squ_Ryi[indx]) - np.abs(eyiout_adj[indx]) ** 2) + \
                                 1j * np.imag(eyout[indx]) * (abs(squ_Ryq[indx]) - np.abs(eyqout_adj[indx]) ** 2)

                    hxx = hxx + self.stepsize * errx[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy + self.stepsize * errx[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx + self.stepsize * erry[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy + self.stepsize * erry[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])

                    cost_x[indx] = (exiout_adj[indx]) ** 2 - squ_Rxi[indx] + 1j * (
                            exqout_adj[indx] ** 2 - squ_Rxq[indx])
                    cost_y[indx] = (eyiout_adj[indx]) ** 2 - squ_Ryi[indx] + 1j * (
                            eyqout_adj[indx] ** 2 - squ_Ryq[indx])
                self.costfunx[batch][it] = -1 * (np.mean(np.real(cost_x)) + np.mean(np.imag(cost_x)))
                self.costfuny[batch][it] = -1 * (np.mean(np.real(cost_y)) + np.mean(np.imag(cost_y)))
                print('iteration = {}'.format(it))
                print(self.costfunx[batch][it])
                print(self.costfuny[batch][it])
                print('-------')
                if it >= 1:
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
                            self.costfuny[batch][it]:
                        print("Earlybreak at iterator {}".format(it))
                        break
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfuny[batch][it]:
                        self.stepsize *= self.stepsizeadjust
                        print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.batchsize - 2 * self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx: indx + 2 * self.center + 1]) + np.matmul(
                    hxy, inputry[indx: indx + 2 * self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx: indx + 2 * self.center + 1]) + np.matmul(
                    hyy, inputry[indx: indx + 2 * self.center + 1])

    def qam_3_butter_oneside(self):
        self.costfunx = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize - 2 * self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize - 2 * self.center), dtype="complex_")
        for batch in range(self.batchnum):
            inputrx = self.rx_x[batch]
            inputry = self.rx_y[batch]
            hxx = np.zeros(self.cmataps, dtype="complex_")
            # hxy = np.zeros(self.cmataps, dtype="complex_")
            # hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.trainlength, dtype="complex_")
            eyout = np.zeros(self.trainlength, dtype="complex_")
            errx = np.zeros(self.trainlength, dtype="complex_")
            erry = np.zeros(self.trainlength, dtype="complex_")
            exiout_adj = np.zeros(self.trainlength, dtype="complex_")
            exqout_adj = np.zeros(self.trainlength, dtype="complex_")
            eyiout_adj = np.zeros(self.trainlength, dtype="complex_")
            eyqout_adj = np.zeros(self.trainlength, dtype="complex_")
            squ_Rxi = np.zeros(self.trainlength, dtype="complex_")
            squ_Rxq = np.zeros(self.trainlength, dtype="complex_")
            squ_Ryi = np.zeros(self.trainlength, dtype="complex_")
            squ_Ryq = np.zeros(self.trainlength, dtype="complex_")
            cost_x = np.zeros(self.trainlength, dtype="complex_")
            cost_y = np.zeros(self.trainlength, dtype="complex_")
            for it in range(self.iterator):
                for indx in range(self.center, self.trainlength - self.center):
                    # if it ==0:
                    inxi = np.real(inputrx[indx - self.center:indx + self.center + 1])
                    inxq = np.imag(inputrx[indx - self.center:indx + self.center + 1])
                    inyi = np.real(inputry[indx - self.center:indx + self.center + 1])
                    inyq = np.imag(inputry[indx - self.center:indx + self.center + 1])
                    # else:
                    #     inxi=np.real(exout[indx - self.center :indx + self.center + 1])
                    #     inxq=np.imag(exout[indx - self.center :indx + self.center + 1])
                    #     inyi=np.real(eyout[indx - self.center :indx + self.center + 1])
                    #     inyq=np.imag(eyout[indx - self.center :indx + self.center + 1])
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                    exiout_adj[indx] = np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])) - 2 * \
                                       np.sign(np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])))
                    exqout_adj[indx] = np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])) - 2 * \
                                       np.sign(np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])))
                    eyiout_adj[indx] = np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])) - 2 * \
                                       np.sign(np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])))
                    eyqout_adj[indx] = np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])) - 2 * \
                                       np.sign(np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])))
                    exiout_adj[indx] = np.real(exout[indx])
                    exqout_adj[indx] = np.imag(exout[indx])
                    eyiout_adj[indx] = np.real(eyout[indx])
                    eyqout_adj[indx] = np.imag(eyout[indx])

                    inxi = inxi - 4 * np.sign(inxi) - 2 * np.sign(inxi - 4 * np.sign(inxi))
                    inxq = inxq - 4 * np.sign(inxq) - 2 * np.sign(inxq - 4 * np.sign(inxq))
                    inyi = inyi - 4 * np.sign(inyi) - 2 * np.sign(inyi - 4 * np.sign(inyi))
                    inyq = inyq - 4 * np.sign(inyq) - 2 * np.sign(inyq - 4 * np.sign(inyq))

                    squ_Rxi[indx] = np.mean(abs(inxi) ** 4) / np.mean(abs(inxi) ** 2)
                    squ_Rxq[indx] = np.mean(abs(inxq) ** 4) / np.mean(abs(inxq) ** 2)
                    squ_Ryi[indx] = np.mean(abs(inyi) ** 4) / np.mean(abs(inyi) ** 2)
                    squ_Ryq[indx] = np.mean(abs(inyq) ** 4) / np.mean(abs(inyq) ** 2)
                    # squ_Rxi[indx] = 10
                    # squ_Rxq[indx] = 10
                    # squ_Ryi[indx] = 10
                    # squ_Ryq[indx] = 10
                    errx[indx] = np.real(exout[indx]) * (abs(squ_Rxi[indx]) - np.abs(exiout_adj[indx]) ** 2) + \
                                 1j * np.imag(exout[indx]) * (abs(squ_Rxq[indx]) - np.abs(exqout_adj[indx]) ** 2)
                    erry[indx] = np.real(eyout[indx]) * (abs(squ_Ryi[indx]) - np.abs(eyiout_adj[indx]) ** 2) + \
                                 1j * np.imag(eyout[indx]) * (abs(squ_Ryq[indx]) - np.abs(eyqout_adj[indx]) ** 2)

                    hxx = hxx + self.stepsize * errx[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    # hxy = hxy + self.stepsize * errx[indx]  * np.conjugate(
                    #     inputry[indx - self.center :indx + self.center + 1])
                    # hyx = hyx + self.stepsize * erry[indx]  * np.conjugate(
                    #     inputrx[indx - self.center :indx + self.center + 1])
                    hyy = hyy + self.stepsize * erry[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])

                    cost_x[indx] = (exiout_adj[indx]) ** 2 - squ_Rxi[indx] + 1j * (
                                exqout_adj[indx] ** 2 - squ_Rxq[indx])
                    cost_y[indx] = (eyiout_adj[indx]) ** 2 - squ_Ryi[indx] + 1j * (
                                eyqout_adj[indx] ** 2 - squ_Ryq[indx])
                self.costfunx[batch][it] = -1 * (np.mean(np.real(cost_x)) + np.mean(np.imag(cost_x)))
                self.costfuny[batch][it] = -1 * (np.mean(np.real(cost_y)) + np.mean(np.imag(cost_y)))
                print('iteration = {}'.format(it))
                print(self.costfunx[batch][it])
                print(self.costfuny[batch][it])
                print('-------')
                # if it >= 1:
                #     if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
                #             self.costfunx[batch][it] :
                #         print("Earlybreak at iterator {}".format(it))
                #         break
                #     if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                #             self.costfunx[batch][it] :
                #         self.stepsize *= self.stepsizeadjust
                #         print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.batchsize - 2 * self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx, inputrx[indx: indx + 2 * self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyy, inputry[indx: indx + 2 * self.center + 1])

    def qam_3_side_single(self):
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        # hxy = np.zeros(self.cmataps, dtype="complex_")
        # hyx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        exiout_adj = np.zeros(self.datalength, dtype="complex_")
        exqout_adj = np.zeros(self.datalength, dtype="complex_")
        eyiout_adj = np.zeros(self.datalength, dtype="complex_")
        eyqout_adj = np.zeros(self.datalength, dtype="complex_")
        squ_Rxi = np.zeros(self.datalength, dtype="complex_")
        squ_Rxq = np.zeros(self.datalength, dtype="complex_")
        squ_Ryi = np.zeros(self.datalength, dtype="complex_")
        squ_Ryq = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")
        cost_y = np.zeros(self.datalength, dtype="complex_")
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                # if it ==0:
                inxi = np.real(inputrx[indx - self.center:indx + self.center + 1])
                inxq = np.imag(inputrx[indx - self.center:indx + self.center + 1])
                inyi = np.real(inputry[indx - self.center:indx + self.center + 1])
                inyq = np.imag(inputry[indx - self.center:indx + self.center + 1])
                # else:
                #     inxi=np.real(exout[indx - self.center :indx + self.center + 1])
                #     inxq=np.imag(exout[indx - self.center :indx + self.center + 1])
                #     inyi=np.real(eyout[indx - self.center :indx + self.center + 1])
                #     inyq=np.imag(eyout[indx - self.center :indx + self.center + 1])
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                exiout_adj[indx] = np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])) - 2 * \
                                   np.sign(np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])))
                exqout_adj[indx] = np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])) - 2 * \
                                   np.sign(np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])))
                eyiout_adj[indx] = np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])) - 2 * \
                                   np.sign(np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])))
                eyqout_adj[indx] = np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])) - 2 * \
                                   np.sign(np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])))
                exiout_adj[indx] = np.real(exout[indx])
                exqout_adj[indx] = np.imag(exout[indx])
                eyiout_adj[indx] = np.real(eyout[indx])
                eyqout_adj[indx] = np.imag(eyout[indx])

                inxi = inxi - 4 * np.sign(inxi) - 2 * np.sign(inxi - 4 * np.sign(inxi))
                inxq = inxq - 4 * np.sign(inxq) - 2 * np.sign(inxq - 4 * np.sign(inxq))
                inyi = inyi - 4 * np.sign(inyi) - 2 * np.sign(inyi - 4 * np.sign(inyi))
                inyq = inyq - 4 * np.sign(inyq) - 2 * np.sign(inyq - 4 * np.sign(inyq))

                squ_Rxi[indx] = np.mean(abs(inxi) ** 4) / np.mean(abs(inxi) ** 2)
                squ_Rxq[indx] = np.mean(abs(inxq) ** 4) / np.mean(abs(inxq) ** 2)
                squ_Ryi[indx] = np.mean(abs(inyi) ** 4) / np.mean(abs(inyi) ** 2)
                squ_Ryq[indx] = np.mean(abs(inyq) ** 4) / np.mean(abs(inyq) ** 2)
                # squ_Rxi[indx] = 10
                # squ_Rxq[indx] = 10
                # squ_Ryi[indx] = 10
                # squ_Ryq[indx] = 10

                errx[indx] = np.real(exout[indx]) * (abs(squ_Rxi[indx]) - np.abs(exiout_adj[indx]) ** 2) + \
                             1j * np.imag(exout[indx]) * (abs(squ_Rxq[indx]) - np.abs(exqout_adj[indx]) ** 2)
                erry[indx] = np.real(eyout[indx]) * (abs(squ_Ryi[indx]) - np.abs(eyiout_adj[indx]) ** 2) + \
                             1j * np.imag(eyout[indx]) * (abs(squ_Ryq[indx]) - np.abs(eyqout_adj[indx]) ** 2)

                hxx = hxx + self.stepsize * errx[indx] * np.conjugate(
                    inputrx[indx - self.center:indx + self.center + 1])
                # hxy = hxy + self.stepsize * errx[indx]  * np.conjugate(
                #     inputry[indx - self.center :indx + self.center + 1])
                # hyx = hyx + self.stepsize * erry[indx]  * np.conjugate(
                #     inputrx[indx - self.center :indx + self.center + 1])
                hyy = hyy + self.stepsize * erry[indx] * np.conjugate(
                    inputry[indx - self.center:indx + self.center + 1])

                cost_x[indx] = (exiout_adj[indx]) ** 2 - squ_Rxi[indx] + 1j * (exqout_adj[indx] ** 2 - squ_Rxq[indx])
                cost_y[indx] = (eyiout_adj[indx]) ** 2 - squ_Ryi[indx] + 1j * (eyqout_adj[indx] ** 2 - squ_Ryq[indx])
            self.costfunx[0][it] = -1 * (np.mean(np.real(cost_x)) + np.mean(np.imag(cost_x)))
            self.costfuny[0][it] = -1 * (np.mean(np.real(cost_y)) + np.mean(np.imag(cost_y)))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print(self.costfuny[0][it])
            print('-------')
            # if it >= 1:
            #     if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
            #             self.costfunx[batch][it] :
            #         print("Earlybreak at iterator {}".format(it))
            #         break
            #     if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
            #             self.costfunx[batch][it] :
            #         self.stepsize *= self.stepsizeadjust
            #         print('Stepsize adjust to {}'.format(self.stepsize))

            self.rx_x_cma = exout
            self.rx_y_cma = eyout

    def qam_3_side_single_conj(self):
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        # hxy = np.zeros(self.cmataps, dtype="complex_")
        # hyx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        exiout_adj = np.zeros(self.datalength, dtype="complex_")
        exqout_adj = np.zeros(self.datalength, dtype="complex_")
        eyiout_adj = np.zeros(self.datalength, dtype="complex_")
        eyqout_adj = np.zeros(self.datalength, dtype="complex_")
        squ_Rxi = np.zeros(self.datalength, dtype="complex_")
        squ_Rxq = np.zeros(self.datalength, dtype="complex_")
        squ_Ryi = np.zeros(self.datalength, dtype="complex_")
        squ_Ryq = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")
        cost_y = np.zeros(self.datalength, dtype="complex_")
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                # if it ==0:
                inxi = np.real(inputrx[indx - self.center:indx + self.center + 1])
                inxq = np.imag(inputrx[indx - self.center:indx + self.center + 1])
                inyi = np.real(inputry[indx - self.center:indx + self.center + 1])
                inyq = np.imag(inputry[indx - self.center:indx + self.center + 1])
                # else:
                #     inxi=np.real(exout[indx - self.center :indx + self.center + 1])
                #     inxq=np.imag(exout[indx - self.center :indx + self.center + 1])
                #     inyi=np.real(eyout[indx - self.center :indx + self.center + 1])
                #     inyq=np.imag(eyout[indx - self.center :indx + self.center + 1])
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                exiout_adj[indx] = np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])) - 2 * \
                                   np.sign(np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])))
                exqout_adj[indx] = np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])) - 2 * \
                                   np.sign(np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])))
                eyiout_adj[indx] = np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])) - 2 * \
                                   np.sign(np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])))
                eyqout_adj[indx] = np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])) - 2 * \
                                   np.sign(np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])))

                exiout_adj[indx] = np.conj(np.real(exout[indx]))
                exqout_adj[indx] = np.conj(np.imag(exout[indx]))
                eyiout_adj[indx] = np.conj(np.real(eyout[indx]))
                eyqout_adj[indx] = np.conj(np.imag(eyout[indx]))

                inxi = inxi - 4 * np.sign(inxi) - 2 * np.sign(inxi - 4 * np.sign(inxi))
                inxq = inxq - 4 * np.sign(inxq) - 2 * np.sign(inxq - 4 * np.sign(inxq))
                inyi = inyi - 4 * np.sign(inyi) - 2 * np.sign(inyi - 4 * np.sign(inyi))
                inyq = inyq - 4 * np.sign(inyq) - 2 * np.sign(inyq - 4 * np.sign(inyq))

                squ_Rxi[indx] = np.mean(abs(inxi) ** 4) / np.mean(abs(inxi) ** 2)
                squ_Rxq[indx] = np.mean(abs(inxq) ** 4) / np.mean(abs(inxq) ** 2)
                squ_Ryi[indx] = np.mean(abs(inyi) ** 4) / np.mean(abs(inyi) ** 2)
                squ_Ryq[indx] = np.mean(abs(inyq) ** 4) / np.mean(abs(inyq) ** 2)

                squ_Rxi[indx] = 10
                squ_Rxq[indx] = 10
                squ_Ryi[indx] = 10
                squ_Ryq[indx] = 10

                errx[indx] = np.real(exout[indx]) * (abs(squ_Rxi[indx]) - np.abs(exiout_adj[indx]) ** 2) + \
                             1j * np.imag(exout[indx]) * (abs(squ_Rxq[indx]) - np.abs(exqout_adj[indx]) ** 2)
                erry[indx] = np.real(eyout[indx]) * (abs(squ_Ryi[indx]) - np.abs(eyiout_adj[indx]) ** 2) + \
                             1j * np.imag(eyout[indx]) * (abs(squ_Ryq[indx]) - np.abs(eyqout_adj[indx]) ** 2)

                hxx = hxx + self.stepsize * errx[indx] * np.conjugate(
                    inputrx[indx - self.center:indx + self.center + 1])
                # hxy = hxy + self.stepsize * errx[indx]  * np.conjugate(
                #     inputry[indx - self.center :indx + self.center + 1])
                # hyx = hyx + self.stepsize * erry[indx]  * np.conjugate(
                #     inputrx[indx - self.center :indx + self.center + 1])
                hyy = hyy + self.stepsize * erry[indx] * np.conjugate(
                    inputry[indx - self.center:indx + self.center + 1])

                cost_x[indx] = (exiout_adj[indx]) ** 2 - squ_Rxi[indx] + 1j * (exqout_adj[indx] ** 2 - squ_Rxq[indx])
                cost_y[indx] = (eyiout_adj[indx]) ** 2 - squ_Ryi[indx] + 1j * (eyqout_adj[indx] ** 2 - squ_Ryq[indx])
            self.costfunx[0][it] = -1 * (np.mean(np.real(cost_x)) + np.mean(np.imag(cost_x)))
            self.costfuny[0][it] = -1 * (np.mean(np.real(cost_y)) + np.mean(np.imag(cost_y)))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print(self.costfuny[0][it])
            print('-------')
            if it >= 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it] :
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.stepsizeadjust * \
                        self.costfunx[0][it] :
                    self.stepsize *= self.stepsizeadjust
                    print('Stepsize adjust to {}'.format(self.stepsize))

            self.rx_x_cma = exout
            self.rx_y_cma = eyout

    def MCMA_MDD(self):
        R = 2.5
        Decision = [R + 1j * R, -R + 1j * R, -R - 1j * R, R - 1j * R]
        self.costfunx_mcma = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny_mcma = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfunx_mdd = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny_mdd = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize - 2 * self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize - 2 * self.center), dtype="complex_")
        for batch in range(self.batchnum):
            inputrx = self.rx_x[batch]
            inputry = self.rx_y[batch]
            # -------h
            hxx_mcma = np.zeros(self.cmataps, dtype="complex_")
            hxy_mcma = np.zeros(self.cmataps, dtype="complex_")
            hyx_mcma = np.zeros(self.cmataps, dtype="complex_")
            hyy_mcma = np.zeros(self.cmataps, dtype="complex_")
            hxx_mcma[self.center] = 1
            hyy_mcma[self.center] = 1
            hxx_mdd = np.zeros(self.cmataps, dtype="complex_")
            hxy_mdd = np.zeros(self.cmataps, dtype="complex_")
            hyx_mdd = np.zeros(self.cmataps, dtype="complex_")
            hyy_mdd = np.zeros(self.cmataps, dtype="complex_")
            hxx_mdd[self.center] = 1
            hyy_mdd[self.center] = 1
            # -------
            exout_mcma = np.zeros(self.trainlength, dtype="complex_")
            eyout_mcma = np.zeros(self.trainlength, dtype="complex_")
            exout_mdd = np.zeros(self.trainlength, dtype="complex_")
            eyout_mdd = np.zeros(self.trainlength, dtype="complex_")
            # -------
            errx_mcma = np.zeros(self.trainlength, dtype="complex_")
            erry_mcma = np.zeros(self.trainlength, dtype="complex_")
            errx_mcma_adj = np.zeros(self.trainlength, dtype="complex_")
            erry_mcma_adj = np.zeros(self.trainlength, dtype="complex_")
            errx_mdd = np.zeros(self.trainlength, dtype="complex_")
            erry_mdd = np.zeros(self.trainlength, dtype="complex_")
            # -------
            exiout_adj_mcma = np.zeros(self.trainlength, dtype="complex_")
            exqout_adj_mcma = np.zeros(self.trainlength, dtype="complex_")
            exout_adj_mcma = np.zeros(self.trainlength, dtype="complex_")
            eyiout_adj_mcma = np.zeros(self.trainlength, dtype="complex_")
            eyqout_adj_mcma = np.zeros(self.trainlength, dtype="complex_")
            eyout_adj_mcma = np.zeros(self.trainlength, dtype="complex_")

            exiout_adj_mdd = np.zeros(self.trainlength, dtype="complex_")
            exqout_adj_mdd = np.zeros(self.trainlength, dtype="complex_")
            exout_adj_mdd = np.zeros(self.trainlength, dtype="complex_")
            eyiout_adj_mdd = np.zeros(self.trainlength, dtype="complex_")
            eyqout_adj_mdd = np.zeros(self.trainlength, dtype="complex_")
            eyout_adj_mdd = np.zeros(self.trainlength, dtype="complex_")
            # -------
            squ_Rxi = np.zeros(self.trainlength, dtype="complex_")
            squ_Rxq = np.zeros(self.trainlength, dtype="complex_")
            squ_Ryi = np.zeros(self.trainlength, dtype="complex_")
            squ_Ryq = np.zeros(self.trainlength, dtype="complex_")
            # -------
            cost_x_mcma = np.zeros(self.trainlength, dtype="complex_")
            cost_y_mcma = np.zeros(self.trainlength, dtype="complex_")
            cost_x_mdd = np.zeros(self.trainlength, dtype="complex_")
            cost_y_mdd = np.zeros(self.trainlength, dtype="complex_")
            for it in range(self.iterator):
                hxx_mdd = np.zeros(self.cmataps, dtype="complex_")
                hxy_mdd = np.zeros(self.cmataps, dtype="complex_")
                hyx_mdd = np.zeros(self.cmataps, dtype="complex_")
                hyy_mdd = np.zeros(self.cmataps, dtype="complex_")
                for indx in range(self.center, self.trainlength - self.center):
                    # --------------------------CMA-------------------------
                    # if it ==0:
                    inxi = np.real(inputrx[indx - self.center:indx + self.center + 1])
                    inxq = np.imag(inputrx[indx - self.center:indx + self.center + 1])
                    inyi = np.real(inputry[indx - self.center:indx + self.center + 1])
                    inyq = np.imag(inputry[indx - self.center:indx + self.center + 1])
                    # else:
                    #     inxi=np.real(exout[indx - self.center :indx + self.center + 1])
                    #     inxq=np.imag(exout[indx - self.center :indx + self.center + 1])
                    #     inyi=np.real(eyout[indx - self.center :indx + self.center + 1])
                    #     inyq=np.imag(eyout[indx - self.center :indx + self.center + 1])
                    exout_mcma[indx] = np.matmul(hxx_mcma, inputrx[indx - self.center:indx + self.center + 1]) + \
                                       np.matmul(hxy_mcma, inputry[indx - self.center:indx + self.center + 1]) + \
                                       np.matmul(hxx_mdd, inputrx[indx - self.center:indx + self.center + 1]) + \
                                       np.matmul(hxy_mdd, inputry[indx - self.center:indx + self.center + 1])
                    eyout_mcma[indx] = np.matmul(hyx_mcma, inputrx[indx - self.center:indx + self.center + 1]) + \
                                       np.matmul(hyy_mcma, inputry[indx - self.center:indx + self.center + 1]) + \
                                       np.matmul(hyx_mdd, inputrx[indx - self.center:indx + self.center + 1]) + \
                                       np.matmul(hyy_mdd, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout_mcma[indx]) or np.isnan(eyout_mcma[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    nn = 3
                    # exiout_adj_mcma[indx] = np.real(exout_mcma[indx]) - 4 * np.sign(np.real(exout_mcma[indx])) - 2 * \
                    #                 np.sign(np.real(exout_mcma[indx]) - 4 * np.sign(np.real(exout_mcma[indx])))
                    # exqout_adj_mcma[indx] = np.imag(exout_mcma[indx]) - 4 * np.sign(np.imag(exout_mcma[indx])) - 2 * \
                    #                 np.sign(np.imag(exout_mcma[indx]) - 4 * np.sign(np.imag(exout_mcma[indx])))
                    exiout_adj_mcma[indx] = np.real(exout_mcma[indx]) - nn * np.sign(np.real(exout_mcma[indx]))
                    exqout_adj_mcma[indx] = np.imag(exout_mcma[indx]) - nn * np.sign(np.imag(exout_mcma[indx]))
                    exout_adj_mcma[indx] = exiout_adj_mcma[indx] + 1j * exqout_adj_mcma[indx]

                    # eyiout_adj_mcma[indx] = np.real(eyout_mcma[indx]) - 4 * np.sign(np.real(eyout_mcma[indx])) - 2 * \
                    #                 np.sign(np.real(eyout_mcma[indx]) - 4 * np.sign(np.real(eyout_mcma[indx])))
                    # eyqout_adj_mcma[indx] = np.imag(eyout_mcma[indx]) - 4 * np.sign(np.imag(eyout_mcma[indx])) - 2 * \
                    #                 np.sign(np.imag(eyout_mcma[indx]) - 4 * np.sign(np.imag(eyout_mcma[indx])))
                    eyiout_adj_mcma[indx] = np.real(eyout_mcma[indx]) - nn * np.sign(np.real(eyout_mcma[indx]))
                    eyqout_adj_mcma[indx] = np.imag(eyout_mcma[indx]) - nn * np.sign(np.imag(eyout_mcma[indx]))
                    eyout_adj_mcma[indx] = eyiout_adj_mcma[indx] + 1j * eyqout_adj_mcma[indx]

                    # inxi=inxi-4*np.sign(inxi)-2*np.sign(inxi-4*np.sign(inxi))
                    # inxq=inxq-4*np.sign(inxq)-2*np.sign(inxq-4*np.sign(inxq))
                    # inyi=inyi-4*np.sign(inyi)-2*np.sign(inyi-4*np.sign(inyi))
                    # inyq=inyq-4*np.sign(inyq)-2*np.sign(inyq-4*np.sign(inyq))

                    inxi = inxi - nn * np.sign(inxi)
                    inxq = inxq - nn * np.sign(inxq)
                    inyi = inyi - nn * np.sign(inyi)
                    inyq = inyq - nn * np.sign(inyq)

                    squ_Rxi[indx] = np.mean(abs(inxi) ** 4) / np.mean(abs(inxi) ** 2)
                    squ_Rxq[indx] = np.mean(abs(inxq) ** 4) / np.mean(abs(inxq) ** 2)
                    squ_Ryi[indx] = np.mean(abs(inyi) ** 4) / np.mean(abs(inyi) ** 2)
                    squ_Ryq[indx] = np.mean(abs(inyq) ** 4) / np.mean(abs(inyq) ** 2)

                    errx_mcma[indx] = np.real(exout_mcma[indx]) * (
                                abs(squ_Rxi[indx]) - np.abs(exiout_adj_mcma[indx]) ** 2) + \
                                      1j * np.imag(exout_mcma[indx]) * (
                                                  abs(squ_Rxq[indx]) - np.abs(exqout_adj_mcma[indx]) ** 2)
                    erry_mcma[indx] = np.real(eyout_mcma[indx]) * (
                                abs(squ_Ryi[indx]) - np.abs(eyiout_adj_mcma[indx]) ** 2) + \
                                      1j * np.imag(eyout_mcma[indx]) * (
                                                  abs(squ_Ryq[indx]) - np.abs(eyqout_adj_mcma[indx]) ** 2)

                    # errx_mcma_adj[indx] = np.real(exout_adj_mcma[indx])*(abs(squ_Rxi[indx])-np.abs(exiout_adj_mcma[indx]) ** 2)+\
                    #                 1j*np.imag(exout_adj_mcma[indx])*(abs(squ_Rxq[indx])-np.abs(exqout_adj_mcma[indx]) ** 2)
                    # erry_mcma_adj[indx] = np.real(eyout_adj_mcma[indx])*(abs(squ_Ryi[indx])-np.abs(eyiout_adj_mcma[indx]) ** 2)+\
                    # 1j*np.imag(eyout_adj_mcma[indx])*(abs(squ_Ryq[indx])-np.abs(eyqout_adj_mcma[indx]) ** 2)

                    hxx_mcma = hxx_mcma + self.stepsize * errx_mcma[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy_mcma = hxy_mcma + self.stepsize * errx_mcma[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx_mcma = hyx_mcma + self.stepsize * erry_mcma[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy_mcma = hyy_mcma + self.stepsize * erry_mcma[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])

                    cost_x_mcma[indx] = (exiout_adj_mcma[indx] ** 2 - squ_Rxi[indx]) + 1j * (
                                exqout_adj_mcma[indx] ** 2 - squ_Rxq[indx])
                    cost_y_mcma[indx] = (eyiout_adj_mcma[indx] ** 2 - squ_Ryi[indx]) + 1j * (
                                eyqout_adj_mcma[indx] ** 2 - squ_Ryq[indx])
                # --------------------------CMA-------------------------
                # --------------------------DD-------------------------
                #     exout_mdd [indx] = np.matmul(hxx_mcma, inputrx[indx - self.center :indx + self.center + 1]) + \
                #                        np.matmul(hxy_mcma, inputry[indx - self.center :indx + self.center + 1]) + \
                #                        np.matmul(hxx_mdd , inputrx[indx - self.center :indx + self.center + 1]) + \
                #                        np.matmul(hxy_mdd , inputry[indx - self.center :indx + self.center + 1])

                #     eyout_mdd [indx] = np.matmul(hyx_mcma, inputrx[indx - self.center :indx + self.center + 1]) + \
                #                        np.matmul(hyy_mcma, inputry[indx - self.center :indx + self.center + 1]) + \
                #                        np.matmul(hyx_mdd , inputrx[indx - self.center :indx + self.center + 1]) + \
                #                        np.matmul(hyy_mdd , inputry[indx - self.center :indx + self.center + 1])
                #     if np.isnan(exout_mdd [indx]) or np.isnan(eyout_mdd[indx]):
                #         raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                #     # exiout_adj_mdd[indx] = np.real(exout_mdd[indx]) - 4 * np.sign(np.real(exout_mdd[indx])) - 2 * \
                #     #                np.sign(np.real(exout_mdd[indx]) - 4 * np.sign(np.real(exout_mdd[indx])))
                #     # exqout_adj_mdd[indx] = np.imag(exout_mdd[indx]) - 4 * np.sign(np.imag(exout_mdd[indx])) - 2 * \
                #     #                np.sign(np.imag(exout_mdd[indx]) - 4 * np.sign(np.imag(exout_mdd[indx])))
                #     exiout_adj_mdd[indx] = np.real(exout_mdd[indx]) - nn * np.sign(np.real(exout_mdd[indx]))
                #     exqout_adj_mdd[indx] = np.imag(exout_mdd[indx]) - nn * np.sign(np.imag(exout_mdd[indx]))
                #     exout_adj_mdd[indx]  = exiout_adj_mdd[indx] + 1j * exqout_adj_mdd[indx]

                #     # eyiout_adj_mdd[indx] = np.real(eyout_mdd[indx]) - 4 * np.sign(np.real(eyout_mdd[indx])) - 2 * \
                #     #                np.sign(np.real(eyout_mdd[indx]) - 4 * np.sign(np.real(eyout_mdd[indx])))
                #     # eyqout_adj_mdd[indx] = np.imag(eyout_mdd[indx]) - 4 * np.sign(np.imag(eyout_mdd[indx])) - 2 * \
                #     #                np.sign(np.imag(eyout_mdd[indx]) - 4 * np.sign(np.imag(eyout_mdd[indx])))
                #     eyiout_adj_mdd[indx] = np.real(eyout_mdd[indx]) - nn * np.sign(np.real(eyout_mdd[indx]))
                #     eyqout_adj_mdd[indx] = np.imag(eyout_mdd[indx]) - nn * np.sign(np.imag(eyout_mdd[indx]))
                #     eyout_adj_mdd[indx]  = eyiout_adj_mdd[indx] + 1j * eyqout_adj_mdd[indx]
                # #---------------------------------------------------
                #     HardDecision_mcma_x = [np.abs(exout_adj_mcma[indx] - Decision[0]),np.abs(exout_adj_mcma[indx] - Decision[1]),
                #                            np.abs(exout_adj_mcma[indx] - Decision[2]),np.abs(exout_adj_mcma[indx] - Decision[3])]
                #     HardDecision_mcma_y = [np.abs(eyout_adj_mcma[indx] - Decision[0]),np.abs(eyout_adj_mcma[indx] - Decision[1]),
                #                            np.abs(eyout_adj_mcma[indx] - Decision[2]),np.abs(eyout_adj_mcma[indx] - Decision[3])]
                #     HardDecision_mdd_x  = [np.abs(exout_adj_mdd [indx] - Decision[0]),np.abs(exout_adj_mdd [indx] - Decision[1]),
                #                            np.abs(exout_adj_mdd [indx] - Decision[2]),np.abs(exout_adj_mdd [indx] - Decision[3])]
                #     HardDecision_mdd_y  = [np.abs(eyout_adj_mdd [indx] - Decision[0]),np.abs(eyout_adj_mdd [indx] - Decision[1]),
                #                            np.abs(eyout_adj_mdd [indx] - Decision[2]),np.abs(eyout_adj_mdd [indx] - Decision[3])]

                #     # errx_mdd[indx] = np.real(exout_mdd [indx])*(np.abs(np.real(Decision[np.argmin(HardDecision_mdd_x)]))-np.abs(exiout_adj_mdd[indx]) ** 2)+\
                #     #               1j*np.imag(exout_mdd [indx])*(np.abs(np.imag(Decision[np.argmin(HardDecision_mdd_x)]))-np.abs(exqout_adj_mdd[indx]) ** 2)
                #     # erry_mdd[indx] = np.real(eyout_mdd [indx])*(np.abs(np.real(Decision[np.argmin(HardDecision_mdd_y)]))-np.abs(eyiout_adj_mdd[indx]) ** 2)+\
                #     #               1j*np.imag(eyout_mdd [indx])*(np.abs(np.imag(Decision[np.argmin(HardDecision_mdd_y)]))-np.abs(eyqout_adj_mdd[indx]) ** 2)
                #     errx_mdd[indx] = np.real(exout_mdd [indx])*(np.abs(R**2-np.abs(exiout_adj_mdd[indx]) ** 2))+\
                #                   1j*np.imag(exout_mdd [indx])*(np.abs(R**2-np.abs(exqout_adj_mdd[indx]) ** 2))
                #     erry_mdd[indx] = np.real(eyout_mdd [indx])*(np.abs(R**2-np.abs(eyiout_adj_mdd[indx]) ** 2))+\
                #                   1j*np.imag(eyout_mdd [indx])*(np.abs(R**2-np.abs(eyqout_adj_mdd[indx]) ** 2))

                #     if Decision[np.argmin(HardDecision_mcma_x)] == Decision[np.argmin(HardDecision_mdd_x)]:
                #         hxx_mdd = hxx_mdd + self.stepsize * errx_mdd[indx]  * np.conjugate(
                #                inputrx[indx - self.center :indx + self.center + 1])
                #         hxy_mdd = hxy_mdd + self.stepsize * errx_mdd[indx]  * np.conjugate(
                #                    inputry[indx - self.center :indx + self.center + 1])
                #     if Decision[np.argmin(HardDecision_mcma_y)] == Decision[np.argmin(HardDecision_mdd_y)]:
                #         hyx_mdd = hyx_mdd + self.stepsize * erry_mdd[indx]  * np.conjugate(
                #                inputrx[indx - self.center :indx + self.center + 1])
                #         hyy_mdd = hyy_mdd + self.stepsize * errx_mdd[indx]  * np.conjugate(
                #                    inputry[indx - self.center :indx + self.center + 1])
                # --------------------------DD-------------------------
                print('iteration = {}'.format(it))
                self.costfunx_mcma[batch][it] = -1 * (np.mean(np.real(cost_x_mcma)) + np.mean(np.imag(cost_x_mcma)))
                self.costfuny_mcma[batch][it] = -1 * (np.mean(np.real(cost_y_mcma)) + np.mean(np.imag(cost_y_mcma)))
                print('CMA costfunc X={}'.format(self.costfunx_mcma[batch][it]))
                print('CMA costfunc Y={}'.format(self.costfuny_mcma[batch][it]))
                # self.costfunx_mdd[batch][it] =np.mean(np.abs(exiout_adj_mdd)**2-np.real(Decision[np.argmin(HardDecision_mdd_x)]))+ \
                #                               np.mean(np.abs(exqout_adj_mdd)**2-np.imag(Decision[np.argmin(HardDecision_mdd_x)]))
                # self.costfuny_mdd[batch][it] =np.mean(np.abs(eyiout_adj_mdd)**2-np.real(Decision[np.argmin(HardDecision_mdd_y)]))+ \
                #                                np.mean(np.abs(eyqout_adj_mdd)**2-np.imag(Decision[np.argmin(HardDecision_mdd_y)]))
                print('DD  costfunc X={}'.format(self.costfunx_mdd[batch][it]))
                print('DD  costfunc Y={}'.format(self.costfuny_mdd[batch][it]))
                print('-------')

                #     if it >= 1:
            #         if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
            #                 self.costfunx[batch][it] and np.abs(
            #             self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
            #                 self.costfuny[batch][it]:
            #             print("Earlybreak at iterator {}".format(it))
            #             break
            #         if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
            #                 self.costfunx[batch][it] and np.abs(
            #             self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
            #                 self.costfuny[batch][it]:
            #             self.stepsize *= self.stepsizeadjust
            #             print('Stepsize adjust to {}'.format(self.stepsize))

            hxx_mdd = np.zeros(self.cmataps, dtype="complex_")
            hxy_mdd = np.zeros(self.cmataps, dtype="complex_")
            hyx_mdd = np.zeros(self.cmataps, dtype="complex_")
            hyy_mdd = np.zeros(self.cmataps, dtype="complex_")
            for indx in range(self.batchsize - 2 * self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx_mcma, inputrx[indx: indx + 2 * self.center + 1]) + \
                                             np.matmul(hxy_mcma, inputry[indx: indx + 2 * self.center + 1])
                # np.matmul(hxx_mdd , inputrx[indx : indx + 2 * self.center + 1]) + \
                # np.matmul(hxy_mdd , inputry[indx : indx + 2 * self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx_mcma, inputrx[indx: indx + 2 * self.center + 1]) + \
                                             np.matmul(hyy_mcma, inputry[indx: indx + 2 * self.center + 1])
                # np.matmul(hyx_mdd , inputrx[indx : indx + 2 * self.center + 1]) + \
                # np.matmul(hyy_mdd , inputry[indx : indx + 2 * self.center + 1])

    def run_16qam_rde(self):
        self.costfunx = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        for batch in range(self.batchnum):
            # initialize H
            inputrx = self.rx_x[batch]
            inputry = self.rx_y[batch]
            hxx = np.zeros(self.cmataps, dtype="complex_")
            hxy = np.zeros(self.cmataps, dtype="complex_")
            hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.trainlength + self.center, dtype="complex_")
            eyout = np.zeros(self.trainlength + self.center, dtype="complex_")
            errx = np.zeros(self.trainlength + self.center, dtype="complex_")
            erry = np.zeros(self.trainlength + self.center, dtype="complex_")
            R = [8, 68, 128]
            # R = [2,10, 18]

            for it in range(self.iterator):
                for indx in range(self.center, self.center + self.trainlength):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    xdistance = [np.abs(np.abs(exout[indx]) - R[0]), np.abs(np.abs(exout[indx]) - R[1]),
                                 np.abs(np.abs(exout[indx]) - R[2])]
                    ydistance = [np.abs(np.abs(eyout[indx]) - R[0]), np.abs(np.abs(eyout[indx]) - R[1]),
                                 np.abs(np.abs(eyout[indx]) - R[2])]
                    errx[indx] = (R[np.argmin(xdistance)] - np.abs(exout[indx]) ** 2)
                    erry[indx] = (R[np.argmin(ydistance)] - np.abs(eyout[indx]) ** 2)

                    # errx[indx] = (R[np.argmin(xdistance)] - np.abs(exout[indx])**2)
                    # erry[indx] = (R[np.argmin(ydistance)] - np.abs(eyout[indx])**2)
                    hxx = hxx + self.stepsize * errx[indx] * exout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy + self.stepsize * errx[indx] * exout[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx + self.stepsize * erry[indx] * eyout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy + self.stepsize * erry[indx] * eyout[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                self.costfunx[batch][it] = np.mean((errx[self.center:]) ** 2)
                self.costfuny[batch][it] = np.mean((erry[self.center:]) ** 2)
                print(self.costfunx[batch][it])
                if it > 1:
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
                            self.costfuny[batch][it]:
                        print("Earlybreak at iterator {}".format(it))
                        break
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfuny[batch][it]:
                        self.stepsize *= self.stepsizeadjust
                        print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def ConstModulusAlgorithm(self, rx, tap_numb, mu, const, iterate=50, title='Constant Modulus Algorithm'):
        T = len(rx)
        N = tap_numb;  # smoothing length N+1
        # Lh=5;  # channel length = Lh+1
        P = round((N) / 2);  # equalization delay
        sig = rx
        Lp = T - N;  # remove several first samples to avoid 0 or negative subscript
        X = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
        x0 = np.hstack((np.zeros(1), sig))
        for i in range(Lp):
            # X[:,i]=np.conj(x0[i+N+1:i:-1].T)
            X[:, i] = np.flipud(x0[i + N + 1:i:-1]).T

        e = np.zeros(Lp, dtype=complex);  # used to save instant error
        f = np.zeros(N + 1, dtype=complex);
        f[P] = 1;  # initial condition
        R2 = const  # np.sqrt(2);                  # constant modulas of QPSK symbols
        mu = mu  # 0.000271;      # parameter to adjust convergence and steady error 16QAM is samller than QAM
        cost = np.zeros([Lp], dtype=complex)
        cost_fun = np.zeros([iterate])
        for k in range(iterate):
            for i in range(Lp):
                y = np.dot(np.conj(f.T), X[:, i])  # update y
                # e[i]=(abs((y.real-2*np.sign(y.real))+1j*(y.imag-2*np.sign(y.imag)))**2-R2)*(y-2*np.sign(y.real)-1j*2*np.sign(y.imag))     # ModifiedCMA cost function, R2=2
                e[i] = y * (abs(y) ** 2 - R2)  # original instant error, R=2/10 for QAM/16QAM
                # e[i]=y.real*(abs(y.real)**2-R2)+1j*y.imag*(abs(y.imag)**2-R2)     # original instant error, R=2/10 for QAM/16QAM
                f = f - mu * np.conj(e[i].T) * X[:, i]  # original update equalizer coefficiency
                # f[P]=1
                cost[i] = e[i] ** 2 - const
            cost_fun[k] = np.mean(cost)
            print(cost_fun[k])
        sb = 1 * np.dot(np.conj(f.T), X)
        return sb

class CMA_single:
    def __init__(self, rx_x, rx_y, taps, iter, mean=True):
        self.mean = mean
        self.rx_x_single = np.array(rx_x)
        self.rx_y_single = np.array(rx_y)
        self.datalength = len(rx_x)
        self.stepsizelist = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 6.409e-6, 1e-6, 2.025e-6 , 8e-7 , 1e-7, 1e-8, 1e9]
        self.overhead = 1
        self.cmataps = taps
        self.center = int((self.cmataps - 1) / 2)
        self.iterator = iter
        self.earlystop = 0.001
        self.stepsizeadjust = 0.95
        self.stepsize = self.stepsizelist[10]
        self.stepsize_x = self.stepsize
        self.stepsize_y = self.stepsize

    def qam_4_side_real(self):
        starttime = datetime.datetime.now()
        self.type = 'single_side_real'
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")
        cost_y = np.zeros(self.datalength, dtype="complex_")
        squ_Rx = np.zeros(self.datalength, dtype="complex_")
        squ_Ry = np.zeros(self.datalength, dtype="complex_")
        squ_Rx, squ_Ry = squ_Rx + 50, squ_Ry + 50

        if self.mean == True:
            squ_Rx = np.zeros(self.datalength, dtype="complex_")
            squ_Ry = np.zeros(self.datalength, dtype="complex_")
            for indx in range(self.center, self.datalength - self.center):
                inx = inputrx[indx - self.center:indx + self.center + 1]
                iny = inputry[indx - self.center:indx + self.center + 1]
                squ_Rx[indx] = np.mean(abs(inx) ** 4) / np.mean(abs(inx) ** 2)
                squ_Ry[indx] = np.mean(abs(iny) ** 4) / np.mean(abs(iny) ** 2)

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                errx[indx] = exout[indx] * (squ_Rx[indx] - np.abs(exout[indx]) ** 2)
                erry[indx] = eyout[indx] * (squ_Ry[indx] - np.abs(eyout[indx]) ** 2)

                hxx = hxx + self.stepsize_x * errx[indx] * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy + self.stepsize_y * erry[indx] * np.conj(
                    inputry[indx - self.center:indx + self.center + 1])

                cost_x[indx] = (abs(exout[indx])) ** 2 - squ_Rx[indx]
                cost_y[indx] = (abs(eyout[indx])) ** 2 - squ_Ry[indx]
            self.costfunx[0][it] = -1 * (np.mean(cost_x))
            self.costfuny[0][it] = -1 * (np.mean(cost_y))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print(self.costfuny[0][it])
            print('-------')
            #
            if it >= 1:
                # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                #         self.costfunx[0][it]:
                #     print("Earlybreak at iterator {}".format(it))
                #     break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop:
                    self.stepsize_x *= self.stepsizeadjust
                    print('Stepsize_x adjust to {}'.format(self.stepsize_x))
                if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop:
                    self.stepsize_y *= self.stepsizeadjust
                    print('Stepsize_y adjust to {}'.format(self.stepsize_y))


        self.rx_x_cma = exout
        self.rx_y_cma = eyout

        endtime = datetime.datetime.now()
        print(endtime - starttime)

    def qam_4_side_conj(self):
        starttime = datetime.datetime.now()
        self.type = 'single_side_conj'
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")
        cost_y = np.zeros(self.datalength, dtype="complex_")
        squ_Rxi, squ_Rxq = np.zeros(self.datalength, dtype="complex_"), np.zeros(self.datalength, dtype="complex_")
        squ_Ryi, squ_Ryq = np.zeros(self.datalength, dtype="complex_"), np.zeros(self.datalength, dtype="complex_")
        squ_Rxi, squ_Rxq = squ_Rxi + 10  * 0.5, squ_Rxq + 10  * 0.5
        squ_Ryi, squ_Ryq = squ_Ryi + 10  * 0.5, squ_Ryq + 10  * 0.5

        if self.mean == True:
            squ_Rxi, squ_Rxq = np.zeros(self.datalength, dtype="complex_"), np.zeros(self.datalength, dtype="complex_")
            squ_Ryi, squ_Ryq = np.zeros(self.datalength, dtype="complex_"), np.zeros(self.datalength, dtype="complex_")
            for indx in range(self.center, self.datalength - self.center):
                inxi = np.real(inputrx[indx - self.center:indx + self.center + 1])
                inxq = np.imag(inputrx[indx - self.center:indx + self.center + 1])
                inyi = np.real(inputry[indx - self.center:indx + self.center + 1])
                inyq = np.imag(inputry[indx - self.center:indx + self.center + 1])
                inxi = inxi - 4 * np.sign(inxi) - 2 * np.sign(inxi - 4 * np.sign(inxi))
                inxq = inxq - 4 * np.sign(inxq) - 2 * np.sign(inxq - 4 * np.sign(inxq))
                inyi = inyi - 4 * np.sign(inyi) - 2 * np.sign(inyi - 4 * np.sign(inyi))
                inyq = inyq - 4 * np.sign(inyq) - 2 * np.sign(inyq - 4 * np.sign(inyq))
                squ_Rxi[indx] = np.mean(abs(inxi) ** 4) / np.mean(abs(inxi) ** 2)
                squ_Rxq[indx] = np.mean(abs(inxq) ** 4) / np.mean(abs(inxq) ** 2)
                squ_Ryi[indx] = np.mean(abs(inyi) ** 4) / np.mean(abs(inyi) ** 2)
                squ_Ryq[indx] = np.mean(abs(inyq) ** 4) / np.mean(abs(inyq) ** 2)

        # if self.mean == True:
        #     squ_Rxi, squ_Rxq = np.zeros(self.datalength, dtype="complex_"), np.zeros(self.datalength, dtype="complex_")
        #     squ_Ryi, squ_Ryq = np.zeros(self.datalength, dtype="complex_"), np.zeros(self.datalength, dtype="complex_")
        #     inxi = np.real(inputrx)
        #     inxq = np.imag(inputrx)
        #     inyi = np.real(inputry)
        #     inyq = np.imag(inputry)
        #     inxi = inxi - 4 * np.sign(inxi) - 2 * np.sign(inxi - 4 * np.sign(inxi))
        #     inxq = inxq - 4 * np.sign(inxq) - 2 * np.sign(inxq - 4 * np.sign(inxq))
        #     inyi = inyi - 4 * np.sign(inyi) - 2 * np.sign(inyi - 4 * np.sign(inyi))
        #     inyq = inyq - 4 * np.sign(inyq) - 2 * np.sign(inyq - 4 * np.sign(inyq))
        #     squ_Rxi = squ_Rxi + np.mean(abs(inxi) ** 4) / np.mean(abs(inxi) ** 2)
        #     squ_Rxq = squ_Rxq + np.mean(abs(inxq) ** 4) / np.mean(abs(inxq) ** 2)
        #     squ_Ryi = squ_Ryi + np.mean(abs(inyi) ** 4) / np.mean(abs(inyi) ** 2)
        #     squ_Ryq = squ_Ryq + np.mean(abs(inyq) ** 4) / np.mean(abs(inyq) ** 2)

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                errx[indx] = np.real(exout[indx]) * (squ_Rxi[indx] - np.abs(np.real(exout[indx])) ** 2) + \
                             1j * np.imag(exout[indx]) * (squ_Rxq[indx] - np.abs(np.imag(exout[indx])) ** 2)
                erry[indx] = np.real(eyout[indx]) * (squ_Ryi[indx] - np.abs(np.real(eyout[indx])) ** 2) + \
                             1j * np.imag(eyout[indx]) * (squ_Ryq[indx] - np.abs(np.imag(eyout[indx])) ** 2)

                hxx = hxx + self.stepsize_x * errx[indx] * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy + self.stepsize_y * erry[indx] * np.conj(
                    inputry[indx - self.center:indx + self.center + 1])

                cost_x[indx] = (np.real(exout[indx]) ** 2 - squ_Rxi[indx]) + 1j * (np.imag(exout[indx]) ** 2 - squ_Rxq[indx])
                cost_y[indx] = (np.real(eyout[indx]) ** 2 - squ_Ryi[indx]) + 1j * (np.imag(eyout[indx]) ** 2 - squ_Ryq[indx])

            self.costfunx[0][it] = -1 * (np.mean(np.real(cost_x)) + np.mean(np.imag(cost_x)))
            self.costfuny[0][it] = -1 * (np.mean(np.real(cost_y)) + np.mean(np.imag(cost_y)))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print(self.costfuny[0][it])
            print('-------')

            if it >= 1:
                # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                #         self.costfunx[0][it]:
                #     print("Earlybreak at iterator {}".format(it))
                #     break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop:
                    self.stepsize_x *= self.stepsizeadjust
                    print('Stepsize_x adjust to {}'.format(self.stepsize_x))
                if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop:
                    self.stepsize_y *= self.stepsizeadjust
                    print('Stepsize_y adjust to {}'.format(self.stepsize_y))

        self.rx_x_cma = exout
        self.rx_y_cma = eyout
        endtime = datetime.datetime.now()
        print(endtime - starttime)

    def qam_4_side_conj_SBD(self):          #A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS
        starttime = datetime.datetime.now()
        self.type = 'single_side_conj_SBD'
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        # errx = np.zeros(self.datalength, dtype="complex_")
        # erry = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")
        cost_y = np.zeros(self.datalength, dtype="complex_")

        Decision = [-7, -5, -3, -1, 1, 3, 5, 7]
        HardDecision_X_real = np.zeros(len(Decision))
        HardDecision_X_imag = np.zeros(len(Decision))
        HardDecision_Y_real = np.zeros(len(Decision))
        HardDecision_Y_imag = np.zeros(len(Decision))

        # epsilon_lambda = 0.5 # forgetting factor
        # epsilon = 2          #initial

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                for i in range(len(Decision)):
                    HardDecision_X_real[i] = Decision[i] - np.real(exout[indx])
                    HardDecision_X_imag[i] = Decision[i] - np.imag(exout[indx])
                    HardDecision_Y_real[i] = Decision[i] - np.real(eyout[indx])
                    HardDecision_Y_imag[i] = Decision[i] - np.imag(eyout[indx])

                errx = np.abs(Decision[np.argmin(abs(HardDecision_X_real))]) * (Decision[np.argmin(abs(HardDecision_X_real))] - np.real(exout[indx])) + \
                    1j*np.abs(Decision[np.argmin(abs(HardDecision_X_imag))]) * (Decision[np.argmin(abs(HardDecision_X_imag))] - np.imag(exout[indx]))
                erry = np.abs(Decision[np.argmin(abs(HardDecision_Y_real))]) * (Decision[np.argmin(abs(HardDecision_Y_real))] - np.real(eyout[indx])) + \
                    1j*np.abs(Decision[np.argmin(abs(HardDecision_Y_imag))]) * (Decision[np.argmin(abs(HardDecision_Y_imag))] - np.imag(eyout[indx]))

                # epsilon = epsilon_lambda * epsilon + (1 - epsilon_lambda)*(abs())
                # p = 7.1467 * (1 - np.exp(8 * epsilon - 0.24)) / (1 + np.exp(8 * epsilon - 0.24)) + 9.1467

                hxx = hxx + self.stepsize_x * errx * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy + self.stepsize_y * erry * np.conj(
                    inputry[indx - self.center:indx + self.center + 1])

                cost_x[indx] = (np.real(exout[indx]) - Decision[np.argmin(abs(HardDecision_X_real))]) + 1j * (np.imag(exout[indx]) - Decision[np.argmin(abs(HardDecision_X_imag))])
                cost_y[indx] = (np.real(eyout[indx]) - Decision[np.argmin(abs(HardDecision_Y_real))]) + 1j * (np.imag(eyout[indx]) - Decision[np.argmin(abs(HardDecision_Y_imag))])


            self.costfunx[0][it] =abs(np.mean(np.real(cost_x)) + abs(np.mean(np.imag(cost_x))))
            self.costfuny[0][it] =abs(np.mean(np.real(cost_y)) + abs(np.mean(np.imag(cost_y))))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print(self.costfuny[0][it])
            print('-------')

            # if it >= 1:
            #     # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
            #     #         self.costfunx[0][it]:
            #     #     print("Earlybreak at iterator {}".format(it))
            #     #     break
            #     if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop:
            #         self.stepsize_x *= self.stepsizeadjust
            #         print('Stepsize_x adjust to {}'.format(self.stepsize_x))
            #     if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop:
            #         self.stepsize_y *= self.stepsizeadjust
            #         print('Stepsize_y adjust to {}'.format(self.stepsize_y))

        self.rx_x_cma = exout
        self.rx_y_cma = eyout
        endtime = datetime.datetime.now()
        print(endtime - starttime)

    def qam_4_side_conj_SBD_polarization(self):  # A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS
        starttime = datetime.datetime.now()
        self.type = 'single_side_conj_SBD_polarization'
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        # errx = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")

        Decision = [-7, -5, -3, -1, 1, 3, 5, 7]
        HardDecision_X_real = np.zeros(len(Decision))
        HardDecision_X_imag = np.zeros(len(Decision))

        epsilon_lambda = 0.2  # forgetting factor
        epsilon_real = 10  # initial
        epsilon_imag = 10  # initial

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                for i in range(len(Decision)):
                    HardDecision_X_real[i] = Decision[i] - np.real(exout[indx])
                    HardDecision_X_imag[i] = Decision[i] - np.imag(exout[indx])

                errx = np.abs(Decision[np.argmin(abs(HardDecision_X_real))]) * (
                            Decision[np.argmin(abs(HardDecision_X_real))] - np.real(exout[indx])) + \
                       1j * np.abs(Decision[np.argmin(abs(HardDecision_X_imag))]) * (
                                   Decision[np.argmin(abs(HardDecision_X_imag))] - np.imag(exout[indx]))

                epsilon_real = epsilon_lambda * epsilon_real + (1 - epsilon_lambda) * (abs(Decision[np.argmin(abs(HardDecision_X_real))]-np.real(exout[indx]))) ** 2
                p_real = 7.1467 * (1 - np.exp(8 * epsilon_real - 0.24)) / (1 + np.exp(8 * epsilon_real - 0.24)) + 9.1467
                gamma_real = 2 ** (-p_real)
                epsilon_imag = epsilon_lambda * epsilon_imag + (1 - epsilon_lambda) * (abs(Decision[np.argmin(abs(HardDecision_X_imag))]-np.imag(exout[indx]))) ** 2
                p_imag = 7.1467 * (1 - np.exp(8 * epsilon_imag - 0.24)) / (1 + np.exp(8 * epsilon_imag - 0.24)) + 9.1467
                gamma_imag = 2 ** (-p_imag)

                if (Decision[np.argmin(abs(HardDecision_X_real))] == -7):
                    errx += gamma_real * (np.abs(Decision[np.argmin(abs(HardDecision_X_real)) + 1]) * (Decision[np.argmin(abs(HardDecision_X_real)) + 1] - np.real(exout[indx])))
                elif (Decision[np.argmin(abs(HardDecision_X_real))] == 7) :
                    errx += gamma_real * (np.abs(Decision[np.argmin(abs(HardDecision_X_real)) - 1]) * (Decision[np.argmin(abs(HardDecision_X_real)) - 1] - np.real(exout[indx])))
                else :
                    errx += gamma_real * (np.abs(Decision[np.argmin(abs(HardDecision_X_real)) - 1]) * (Decision[np.argmin(abs(HardDecision_X_real)) - 1] - np.real(exout[indx])) + \
                                          np.abs(Decision[np.argmin(abs(HardDecision_X_real)) + 1]) * (Decision[np.argmin(abs(HardDecision_X_real)) + 1] - np.real(exout[indx])))

                if (Decision[np.argmin(abs(HardDecision_X_imag))] == -7):
                    errx += gamma_imag * (np.abs(Decision[np.argmin(abs(HardDecision_X_imag)) + 1]) * (Decision[np.argmin(abs(HardDecision_X_imag)) + 1] - np.imag(exout[indx]))) * 1j
                elif (Decision[np.argmin(abs(HardDecision_X_imag))] == 7) :
                    errx += gamma_imag * (np.abs(Decision[np.argmin(abs(HardDecision_X_imag)) - 1]) * (Decision[np.argmin(abs(HardDecision_X_imag)) - 1] - np.imag(exout[indx]))) * 1j
                else :
                    errx += gamma_imag * (np.abs(Decision[np.argmin(abs(HardDecision_X_imag)) - 1]) * (Decision[np.argmin(abs(HardDecision_X_imag)) - 1] - np.imag(exout[indx])) + \
                                          np.abs(Decision[np.argmin(abs(HardDecision_X_imag)) + 1]) * (Decision[np.argmin(abs(HardDecision_X_imag)) + 1] - np.imag(exout[indx]))) * 1j

                hxx = hxx + self.stepsize_x * errx * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])

                cost_x[indx] = (np.real(exout[indx]) - Decision[np.argmin(abs(HardDecision_X_real))]) ** 2 + 1j * (
                            np.imag(exout[indx]) - Decision[np.argmin(abs(HardDecision_X_imag))]) ** 2

            self.costfunx[0][it] = (np.mean(np.real(cost_x)) + (np.mean(np.imag(cost_x))))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print('-------')

            if it >= 1:
                # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                #         self.costfunx[0][it]:
                #     print("Earlybreak at iterator {}".format(it))
                #     break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop:
                    self.stepsize_x *= self.stepsizeadjust
                    print('Stepsize_x adjust to {}'.format(self.stepsize_x))
            #     if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop:
            #         self.stepsize_y *= self.stepsizeadjust
            #         print('Stepsize_y adjust to {}'.format(self.stepsize_y))

        self.rx_x_cma = exout
        endtime = datetime.datetime.now()
        print(endtime - starttime)

    def qam_4_side_conj_MRD_polarization(self):  # A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS
        starttime = datetime.datetime.now()
        self.type = 'single_side_conj_MRD_polarization'
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        # errx = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")

        Decision = [-7, -5, -3, -1, 1, 3, 5, 7]
        HardDecision_X_real = np.zeros(len(Decision))
        HardDecision_X_imag = np.zeros(len(Decision))

        epsilon_lambda = 0.2  # forgetting factor
        epsilon_real = 2  # initial
        epsilon_imag = 2  # initial

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                for i in range(len(Decision)):
                    HardDecision_X_real[i] = Decision[i] - np.real(exout[indx])
                    HardDecision_X_imag[i] = Decision[i] - np.imag(exout[indx])

                errx = ((Decision[np.argmin(abs(HardDecision_X_real))] ** 2) - np.real(exout[indx]) ** 2) * np.real(exout[indx]) + \
                  1j * ((Decision[np.argmin(abs(HardDecision_X_imag))] ** 2) - np.imag(exout[indx]) ** 2) * np.imag(exout[indx])

                epsilon_real = epsilon_lambda * epsilon_real + (1 - epsilon_lambda) * (abs(Decision[np.argmin(abs(HardDecision_X_real))]-np.real(exout[indx]))) ** 2
                p_real = 7.1467 * (1 - np.exp(8 * epsilon_real - 0.24)) / (1 + np.exp(8 * epsilon_real - 0.24)) + 9.1467
                gamma_real = 2 ** (-p_real)
                epsilon_imag = epsilon_lambda * epsilon_imag + (1 - epsilon_lambda) * (abs(Decision[np.argmin(abs(HardDecision_X_imag))]-np.imag(exout[indx]))) ** 2
                p_imag = 7.1467 * (1 - np.exp(8 * epsilon_imag - 0.24)) / (1 + np.exp(8 * epsilon_imag - 0.24)) + 9.1467
                gamma_imag = 2 ** (-p_imag)

                if (Decision[np.argmin(abs(HardDecision_X_real))] == -7):
                    errx += gamma_real * ((Decision[np.argmin(abs(HardDecision_X_real)) + 1] ** 2) - np.real(exout[indx]) ** 2) * np.real(exout[indx])
                elif (Decision[np.argmin(abs(HardDecision_X_real))] == 7) :
                    errx += gamma_real * ((Decision[np.argmin(abs(HardDecision_X_real)) - 1] ** 2) - np.real(exout[indx]) ** 2) * np.real(exout[indx])
                else:
                    errx += gamma_real * (((Decision[np.argmin(abs(HardDecision_X_real)) - 1] ** 2) - np.real(exout[indx]) ** 2) * np.real(exout[indx]) + \
                                          ((Decision[np.argmin(abs(HardDecision_X_real)) + 1] ** 2) - np.real(exout[indx]) ** 2) * np.real(exout[indx]))


                if (Decision[np.argmin(abs(HardDecision_X_imag))] == -7):
                    errx += gamma_imag * ((Decision[np.argmin(abs(HardDecision_X_imag)) + 1] ** 2) - np.imag(exout[indx]) ** 2) * np.imag(exout[indx]) * 1j
                elif (Decision[np.argmin(abs(HardDecision_X_imag))] == 7) :
                    errx += gamma_imag * ((Decision[np.argmin(abs(HardDecision_X_imag)) - 1] ** 2) - np.imag(exout[indx]) ** 2) * np.imag(exout[indx]) * 1j
                else:
                    errx += gamma_imag * (((Decision[np.argmin(abs(HardDecision_X_imag)) + 1] ** 2) - np.imag(exout[indx]) ** 2) * np.imag(exout[indx]) + \
                                          ((Decision[np.argmin(abs(HardDecision_X_imag)) - 1] ** 2) - np.imag(exout[indx]) ** 2) * np.imag(exout[indx])) * 1j

                hxx = hxx + self.stepsize_x * errx * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])

                cost_x[indx] = (np.real(exout[indx]) - Decision[np.argmin(abs(HardDecision_X_real))]) ** 2 + 1j * (
                            np.imag(exout[indx]) - Decision[np.argmin(abs(HardDecision_X_imag))]) ** 2

            self.costfunx[0][it] = (np.mean(np.real(cost_x)) + (np.mean(np.imag(cost_x))))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print('-------')

            if it >= 1:
                # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                #         self.costfunx[0][it]:
                #     print("Earlybreak at iterator {}".format(it))
                #     break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < (self.earlystop * 100):
                    self.stepsize_x *= self.stepsizeadjust
                    print('Stepsize_x adjust to {}'.format(self.stepsize_x))
            #     if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop:
            #         self.stepsize_y *= self.stepsizeadjust
            #         print('Stepsize_y adjust to {}'.format(self.stepsize_y))

        self.rx_x_cma = exout
        endtime = datetime.datetime.now()
        print(endtime - starttime)

    def qam_4_side_conj_RMA_polarization(self):  # A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS
        starttime = datetime.datetime.now()
        self.type = 'single_side_conj_RMA_polarization'
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        # errx = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")

        Decision = [-7, -5, -3, -1, 1, 3, 5, 7]
        HardDecision_X_real = np.zeros(len(Decision))
        HardDecision_X_imag = np.zeros(len(Decision))
        Center = [-6, -2, 2, 6]
        Center_X_real = np.zeros(len(Center))
        Center_X_imag = np.zeros(len(Center))

        epsilon_lambda = 0.3  # forgetting factor
        epsilon_real = 10  # initial
        epsilon_imag = 10  # initial

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                for i in range(len(Decision)):
                    HardDecision_X_real[i] = Decision[i] - np.real(exout[indx])
                    HardDecision_X_imag[i] = Decision[i] - np.imag(exout[indx])
                for i in range(len(Center)):
                    Center_X_real[i] = Center[i] - np.real(exout[indx])
                    Center_X_imag[i] = Center[i] - np.imag(exout[indx])

                if (Center[np.argmin(abs(Center_X_real))] == 2) or (Center[np.argmin(abs(Center_X_real))] == -2):
                    cost_x[indx] = (2.86 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) ** 2

                    errx = (2.86 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))])
                else :
                    cost_x[indx] = (6.39 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) ** 2

                    errx = (6.39 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))])

                if (Center[np.argmin(abs(Center_X_imag))] == 2) or (Center[np.argmin(abs(Center_X_imag))] == -2):
                    cost_x[indx] = 1j * (2.86 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) ** 2

                    errx += 1j * (2.86 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))])
                else :
                    cost_x[indx] = 1j * (6.39 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) ** 2

                    errx += 1j * (6.39 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))])

                # epsilon_real = epsilon_lambda * epsilon_real + (1 - epsilon_lambda) * (abs(Decision[np.argmin(abs(HardDecision_X_real))]-np.real(exout[indx]))) ** 2
                # p_real = 7.1467 * (1 - np.exp(8 * epsilon_real - 0.24)) / (1 + np.exp(8 * epsilon_real - 0.24)) + 9.1467
                # gamma_real = 2 ** (-p_real)
                # epsilon_imag = epsilon_lambda * epsilon_imag + (1 - epsilon_lambda) * (abs(Decision[np.argmin(abs(HardDecision_X_imag))]-np.imag(exout[indx]))) ** 2
                # p_imag = 7.1467 * (1 - np.exp(8 * epsilon_imag - 0.24)) / (1 + np.exp(8 * epsilon_imag - 0.24)) + 9.1467
                # gamma_imag = 2 ** (-p_imag)
                #
                # epsilon_real = epsilon_lambda * epsilon_real + (1 - epsilon_lambda) * (abs(Center[np.argmin(abs(Center_X_real))]-np.real(exout[indx]))) ** 2
                # p_real = 7.1467 * (1 - np.exp(8 * epsilon_real - 0.24)) / (1 + np.exp(8 * epsilon_real - 0.24)) + 9.1467
                # gamma_real = 2 ** (-p_real)
                # epsilon_imag = epsilon_lambda * epsilon_imag + (1 - epsilon_lambda) * (abs(Center[np.argmin(abs(Center_X_imag))]-np.imag(exout[indx]))) ** 2
                # p_imag = 7.1467 * (1 - np.exp(8 * epsilon_imag - 0.24)) / (1 + np.exp(8 * epsilon_imag - 0.24)) + 9.1467
                # gamma_imag = 2 ** (-p_imag)
                #
                # if (Center[np.argmin(abs(Center_X_real))] == Center[0]) :
                #     cost_x[indx] = (6.39 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) ** 2
                #
                #     errx = (6.39 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))])
                #     errx += gamma_real * (2.86 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) + 1]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) + 1])
                # elif (Center[np.argmin(abs(Center_X_real))] == Center[3]) :
                #     cost_x[indx] = (6.39 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) ** 2
                #
                #     errx = (6.39 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))])
                #     errx += gamma_real * (2.86 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) - 1]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) - 1])
                # elif (Center[np.argmin(abs(Center_X_real))] == Center[1]):
                #     cost_x[indx] = (2.86 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) ** 2
                #
                #     errx = (2.86 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))])
                #     errx += gamma_real * (6.39 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) - 1]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) - 1])
                #     errx += gamma_real * (2.86 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) + 1]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) + 1])
                # else:
                #     cost_x[indx] = (2.86 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) ** 2
                #
                #     errx = (2.86 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real))])
                #     errx += gamma_real * (6.39 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) + 1]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) + 1])
                #     errx += gamma_real * (2.86 ** 2) * (1 - (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) - 1]) ** 2) * (np.real(exout[indx]) - Center[np.argmin(abs(Center_X_real)) - 1])
                #
                #
                # if (Center[np.argmin(abs(Center_X_imag))] == Center[0]) :
                #     cost_x[indx] = 1j * (6.39 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) ** 2) ** 2
                #
                #     errx += (6.39 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) * 1j
                #     errx += gamma_imag * (2.86 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) + 1]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) + 1]) * 1j
                # elif (Center[np.argmin(abs(Center_X_imag))] == Center[3]) :
                #     cost_x[indx] = 1j * (6.39 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) ** 2) ** 2
                #
                #     errx += (6.39 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) * 1j
                #     errx += gamma_imag * (2.86 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) - 1]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) - 1]) * 1j
                # elif (Center[np.argmin(abs(Center_X_imag))] == Center[1]):
                #     cost_x[indx] = 1j * (2.86 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) ** 2) ** 2
                #
                #     errx += (2.86 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) * 1j
                #     errx += gamma_imag * (6.39 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) - 1]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) - 1]) * 1j
                #     errx += gamma_imag * (2.86 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) + 1]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) + 1]) * 1j
                # else:
                #     cost_x[indx] = 1j * (2.86 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) ** 2) ** 2
                #
                #     errx += (2.86 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag))]) * 1j
                #     errx += gamma_imag * (6.39 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) + 1]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) + 1]) * 1j
                #     errx += gamma_imag * (2.86 ** 2) * (1 - (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) - 1]) ** 2) * (np.imag(exout[indx]) - Center[np.argmin(abs(Center_X_imag)) - 1]) * 1j

                hxx = hxx + self.stepsize_x * errx * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])

                # cost_x[indx] = (np.real(exout[indx]) - Decision[np.argmin(abs(HardDecision_X_real))]) ** 2 + 1j * (
                #             np.imag(exout[indx]) - Decision[np.argmin(abs(HardDecision_X_imag))]) ** 2

            self.costfunx[0][it] = (np.mean(np.real(cost_x)) + (np.mean(np.imag(cost_x))))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print('-------')

            if it >= 1:
                # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                #         self.costfunx[0][it]:
                #     print("Earlybreak at iterator {}".format(it))
                #     break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < (self.earlystop * 1000):
                    self.stepsize_x *= self.stepsizeadjust
                    print('Stepsize_x adjust to {}'.format(self.stepsize_x))
            #     if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop:
            #         self.stepsize_y *= self.stepsizeadjust
            #         print('Stepsize_y adjust to {}'.format(self.stepsize_y))
        self.rx_x_cma = exout
        endtime = datetime.datetime.now()
        print(endtime - starttime)

    def qam_4_side_RD_polarization(self, stage = 1):  # A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS
        starttime = datetime.datetime.now()
        self.type = 'single_side_RD_polarization'
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        # errx = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")

        epsilon_lambda = 0.1  # forgetting factor
        epsilon = 2  # initial

        if stage == 3:
            radius = [2 ** 0.5, 10 ** 0.5, 18 ** 0.5, 26 ** 0.5, 34 ** 0.5, 50 ** 0.5, 58 ** 0.5, 74 ** 0.5, 98 ** 0.5]
        elif stage == 2:
            radius = [10 ** 0.5, 50 ** 0.5, 74 ** 0.5]
        elif stage == 1:
            radius = [50 ** 0.5]

        HardDecision_X = np.zeros(len(radius))

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):

                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                for i in range(len(radius)):
                    HardDecision_X[i] = radius[i] - np.abs(exout[indx])

                errx = (radius[np.argmin(abs(HardDecision_X))] ** 2 - np.abs(exout[indx]) ** 2) * (exout[indx])

                # epsilon = epsilon_lambda * epsilon + (1 - epsilon_lambda) * (abs(radius[np.argmin(abs(HardDecision_X))]- np.abs(exout[indx]))) ** 2
                # p = 7.1467 * (1 - np.exp(8 * epsilon - 0.24)) / (1 + np.exp(8 * epsilon - 0.24)) + 9.1467
                # gamma = 2 ** (-p)
                #
                # if (radius[np.argmin(abs(HardDecision_X))] == radius[0]):
                #     errx += gamma * ((radius[np.argmin(abs(HardDecision_X)) + 1] ** 2 - np.abs(exout[indx]) ** 2) * (exout[indx]))
                # elif (radius[np.argmin(abs(HardDecision_X))] == radius[-1]):
                #     errx += gamma * ((radius[np.argmin(abs(HardDecision_X)) - 1] ** 2 - np.abs(exout[indx]) ** 2) * (exout[indx]))
                # else:
                #     errx += gamma * ((radius[np.argmin(abs(HardDecision_X)) + 1] ** 2 - np.abs(exout[indx]) ** 2) * (exout[indx])) + \
                #             gamma * ((radius[np.argmin(abs(HardDecision_X)) - 1] ** 2 - np.abs(exout[indx]) ** 2) * (exout[indx]))

                hxx = hxx + self.stepsize_x * errx * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])

                # cost_x[indx] = (np.abs(exout[indx]) - radius[np.argmin(abs(HardDecision_X))]) ** 2

                cost_x[indx] = (abs(exout[indx])) ** 2 - radius[np.argmin(abs(HardDecision_X))] ** 2

            # self.costfunx[0][it] = np.mean(cost_x)
            self.costfunx[0][it] = np.mean(cost_x) * -1
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print('-------')

            if it >= 1:
                # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                #         self.costfunx[0][it]:
                #     print("Earlybreak at iterator {}".format(it))
                #     break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < (self.earlystop * 50):
                    self.stepsize_x *= self.stepsizeadjust
                    print('Stepsize_x adjust to {}'.format(self.stepsize_x))
            #     if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop:
            #         self.stepsize_y *= self.stepsizeadjust
            #         print('Stepsize_y adjust to {}'.format(self.stepsize_y))

        self.rx_x_cma = exout
        endtime = datetime.datetime.now()
        print(endtime - starttime)

    def qam_4_side_real_m(self):
        starttime = datetime.datetime.now()

        self.type = 'single_side_real_m'
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        # errx = np.zeros(self.datalength, dtype="complex_")
        # erry = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")
        cost_y = np.zeros(self.datalength, dtype="complex_")
        inputrx_stack = np.zeros([self.cmataps, self.datalength], dtype="complex_")
        inputry_stack = np.zeros([self.cmataps, self.datalength], dtype="complex_")
        squ_Rx = np.zeros(self.datalength, dtype="complex_")
        squ_Ry = np.zeros(self.datalength, dtype="complex_")
        squ_Rx, squ_Ry = squ_Rx + 10, squ_Ry + 10

        for indx in range(self.center, self.datalength - self.center):
            inputrx_stack[:, indx] = inputrx[indx - self.center:indx + self.center + 1]
            inputry_stack[:, indx] = inputry[indx - self.center:indx + self.center + 1]

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout = np.dot(hxx, inputrx_stack[:, indx])
                eyout = np.dot(hyy, inputry_stack[:, indx])
                if np.isnan(exout) or np.isnan(eyout):
                # if np.isnan(exout) :
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                errx = exout * (squ_Rx[indx] - np.abs(exout) ** 2)
                erry = eyout * (squ_Ry[indx] - np.abs(eyout) ** 2)

                hxx = hxx + self.stepsize_x * errx * np.conj(inputrx_stack[:, indx])
                hyy = hyy + self.stepsize_y * erry * np.conj(inputry_stack[:, indx])

                cost_x[indx] = (abs(exout)) ** 2 - squ_Rx[indx]
                cost_y[indx] = (abs(eyout)) ** 2 - squ_Ry[indx]

            self.costfunx[0][it] = -1 * (np.mean(cost_x))
            self.costfuny[0][it] = -1 * (np.mean(cost_y))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print(self.costfuny[0][it])
            print('-------')
        self.rx_x_cma = np.dot(hxx.T, inputrx_stack)
        self.rx_y_cma = np.dot(hyy.T, inputry_stack)

        endtime = datetime.datetime.now()
        print(endtime - starttime)

    def ConstModulusAlgorithm(self, rx, tap_numb, mu, PAM_order, iterate=50):
        starttime = datetime.datetime.now()
        self.type = 'egg'

        if PAM_order == 2:
            const = 2
        elif PAM_order == 4:
            const = 10
        else:
            ValueError  # 64QAM is developement
        T = len(rx)
        N = tap_numb;  # smoothing length N+1
        # Lh=5;  # channel length = Lh+1
        P = round((N) / 2);  # equalization delay
        sig = rx
        Lp = T - N;  # remove several first samples to avoid 0 or negative subscript
        X = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
        x0 = np.hstack((np.zeros(1), sig))
        for i in range(Lp):
            # X[:,i]=np.conj(x0[i+N+1:i:-1].T)
            X[:, i] = np.flipud(x0[i + N + 1:i:-1]).T

        e = np.zeros(Lp, dtype=complex);  # used to save instant error
        f = np.zeros(N + 1, dtype=complex);
        f[P] = 1;  # initial condition
        R2 = const  # np.sqrt(2);                  # constant modulas of QPSK symbols
        mu = mu  # 0.000271;      # parameter to adjust convergence and steady error 16QAM is samller than QAM

        for k in range(iterate):
            for i in range(int(Lp * 1)):  # int(Lp*0.15)
                y = np.dot(np.conj(f.T), X[:, i])  # update y
                # e[i]=(abs((y.real-2*np.sign(y.real))+1j*(y.imag-2*np.sign(y.imag)))**2-R2)*(y-2*np.sign(y.real)-1j*2*np.sign(y.imag))     # ModifiedCMA cost function, R2=2
                e[i] = y * (abs(y) ** 2 - R2)  # original instant error, R=2/10 for QAM/16QAM
                # e[i]=y.real*(abs(y.real)**2-R2)+1j*y.imag*(abs(y.imag)**2-R2)     # original instant error, R=2/10 for QAM/16QAM
                f = f - mu * np.conj(e[i].T) * X[:, i]  # original update equalizer coefficiency
                # f[P]=1
        sb = 1 * np.dot(np.conj(f.T), X)
        # self.constellation_plot(sb, title=title, bins=75)
        endtime = datetime.datetime.now()
        print(endtime - starttime)
        return sb

    def qam_4_side_real_shift(self):
        self.type = 'single_side_real_shift'
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        exout_adj = np.zeros(self.datalength, dtype="complex_")
        eyout_adj = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")
        cost_y = np.zeros(self.datalength, dtype="complex_")
        squ_R_x, squ_R_y = 2 ** 0.5 ,2 ** 0.5

        if self.mean == True:
            squ_Rx = np.zeros(self.datalength, dtype="complex_")
            squ_Ry = np.zeros(self.datalength, dtype="complex_")
            for indx in range(self.center, self.datalength - self.center):
                inx = inputrx[indx - self.center:indx + self.center + 1]
                iny = inputry[indx - self.center:indx + self.center + 1]
                inx = inx + (- 4 * np.sign(np.real(inx)) - 2 * np.sign(
                    np.real(inx) - 4 * np.sign(np.real(inx)))) + 1j * (
                              - 4 * np.sign(np.imag(inx)) - 2 * np.sign(
                          np.imag(inx) - 4 * np.sign(np.imag(inx))))

                iny = iny + (- 4 * np.sign(np.real(iny)) - 2 * np.sign(
                    np.real(iny) - 4 * np.sign(np.real(iny)))) + 1j * (
                              - 4 * np.sign(np.imag(iny)) - 2 * np.sign(
                          np.imag(iny) - 4 * np.sign(np.imag(iny))))
                # nn = 2
                # inx = inx - nn * np.sign(np.real(inx)) - 1j * nn * np.sign(
                #     np.imag(inx))
                # iny = iny - nn * np.sign(np.real(iny)) - 1j * nn * np.sign(
                #     np.imag(iny))

                squ_R_x[indx] = np.mean(abs(inx) ** 4) / np.mean(abs(inx) ** 2)
                squ_R_y[indx] = np.mean(abs(iny) ** 4) / np.mean(abs(iny) ** 2)

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                exout_adj[indx] = exout[indx] + (- 4 * np.sign(np.real(exout[indx])) - 2 * np.sign(
                    np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])))) + 1j * (
                                              - 4 * np.sign(np.imag(exout[indx])) - 2 * np.sign(
                                          np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx]))))

                # exout_adj = exout + (- 4 * np.sign(np.real(exout)) - 2 * np.sign(
                #     np.real(exout) - 4 * np.sign(np.real(exout)))) + 1j * (
                #                           - 4 * np.sign(np.imag(exout)) - 2 * np.sign(
                #                       np.imag(exout) - 4 * np.sign(np.imag(exout))))

                eyout_adj[indx] = eyout[indx] + (- 4 * np.sign(np.real(eyout[indx])) - 2 * np.sign(
                    np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])))) + 1j * (
                                              - 4 * np.sign(np.imag(eyout[indx])) - 2 * np.sign(
                                          np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx]))))
                # nn = 2
                # exout_adj[indx] = exout[indx] - nn * np.sign(np.real(exout[indx])) - 1j*nn*np.sign(np.imag(exout[indx]))
                # eyout_adj[indx] = eyout[indx] - nn * np.sign(np.real(eyout[indx])) - 1j*nn*np.sign(np.imag(eyout[indx]))

                errx[indx] = (squ_R_x - np.abs(exout_adj[indx]) ** 2)
                erry[indx] = (squ_R_y - np.abs(eyout_adj[indx]) ** 2)
                # hxx = hxx + self.stepsize * np.conj(exout_adj[indx] * errx[indx]) * (
                #     inputrx[indx - self.center:indx + self.center + 1])
                # hyy = hyy + self.stepsize * np.conj(eyout_adj[indx] * erry[indx]) * (
                #     inputry[indx - self.center:indx + self.center + 1])

                # errx[indx] = (R_x - np.abs(exout[indx]) ** 2)
                # erry[indx] = (R_y - np.abs(eyout[indx]) ** 2)
                hxx = hxx + self.stepsize * exout[indx] * errx[indx] * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy + self.stepsize * eyout[indx] * erry[indx] * np.conj(
                    inputry[indx - self.center:indx + self.center + 1])

                cost_x[indx] = (abs(exout_adj[indx])) ** 2 - squ_R_x
                cost_y[indx] = (abs(eyout_adj[indx])) ** 2 - squ_R_y
            self.costfunx[0][it] = -1 * (np.mean(cost_x))
            self.costfuny[0][it] = -1 * (np.mean(cost_y))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print(self.costfuny[0][it])
            print('-------')

            if it >= 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.stepsizeadjust * \
                        self.costfunx[0][it]:
                    self.stepsize *= self.stepsizeadjust
                    print('Stepsize adjust to {}'.format(self.stepsize))

            self.rx_x_cma = exout
            self.rx_y_cma = eyout

    def qam_4_butter_real(self):
        self.type = 'single_butter_real'
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxy = np.zeros(self.cmataps, dtype="complex_")
        hyx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")
        cost_y = np.zeros(self.datalength, dtype="complex_")
        squ_Rx = np.zeros(self.datalength, dtype="complex_")
        squ_Ry = np.zeros(self.datalength, dtype="complex_")
        squ_Rx, squ_Ry = squ_Rx + 50, squ_Ry + 50
        #
        if self.mean == True:
            squ_Rx = np.zeros(self.datalength, dtype="complex_")
            squ_Ry = np.zeros(self.datalength, dtype="complex_")
            for indx in range(self.center, self.datalength - self.center):
                inx = inputrx[indx - self.center:indx + self.center + 1]
                iny = inputry[indx - self.center:indx + self.center + 1]
                squ_Rx[indx] = np.mean(abs(inx) ** 4) / np.mean(abs(inx) ** 2)
                squ_Ry[indx] = np.mean(abs(iny) ** 4) / np.mean(abs(iny) ** 2)

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                errx[indx] = (squ_Rx[indx] - np.abs(exout[indx]) ** 2)
                erry[indx] = (squ_Ry[indx] - np.abs(eyout[indx]) ** 2)

                hxx = hxx + self.stepsize_x * exout[indx] * errx[indx] * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                hxy = hxy + self.stepsize_x * exout[indx] * errx[indx] * np.conj(
                    inputry[indx - self.center:indx + self.center + 1])
                hyx = hyx + self.stepsize_y * eyout[indx] * erry[indx] * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy + self.stepsize_y * eyout[indx] * erry[indx] * np.conj(
                    inputry[indx - self.center:indx + self.center + 1])

                cost_x[indx] = (abs(exout[indx])) ** 2 - squ_Rx[indx]
                cost_y[indx] = (abs(eyout[indx])) ** 2 - squ_Ry[indx]
            self.costfunx[0][it] = -1 * (np.mean(cost_x))
            self.costfuny[0][it] = -1 * (np.mean(cost_y))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print(self.costfuny[0][it])
            print('-------')

            if it >= 1:
                # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                #         self.costfunx[0][it]:
                #     print("Earlybreak at iterator {}".format(it))
                #     break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < 0.001:
                    self.stepsize_x *= self.stepsizeadjust
                    print('Stepsize_x adjust to {}'.format(self.stepsize_x))
                if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < 0.001:
                    self.stepsize_y *= self.stepsizeadjust
                    print('Stepsize_y adjust to {}'.format(self.stepsize_y))

        self.rx_x_cma = exout
        self.rx_y_cma = eyout

    def qam_4_butter_real_shift(self):
        self.type = 'single_butter_real_shift'
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxy = np.zeros(self.cmataps, dtype="complex_")
        hyx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        exout_adj = np.zeros(self.datalength, dtype="complex_")
        eyout_adj = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")
        cost_y = np.zeros(self.datalength, dtype="complex_")
        squ_Rx = np.zeros(self.datalength, dtype="complex_")
        squ_Ry = np.zeros(self.datalength, dtype="complex_")
        squ_Rx, squ_Ry = squ_Rx + 10, squ_Ry + 10

        if self.mean == True:
            squ_Rx = np.zeros(self.datalength, dtype="complex_")
            squ_Ry = np.zeros(self.datalength, dtype="complex_")
            for indx in range(self.center, self.datalength - self.center):
                inx = inputrx[indx - self.center:indx + self.center + 1]
                iny = inputry[indx - self.center:indx + self.center + 1]
                inx = inx + (- 4 * np.sign(np.real(inx)) - 2 * np.sign(
                    np.real(inx) - 4 * np.sign(np.real(inx)))) + 1j * (
                              - 4 * np.sign(np.imag(inx)) - 2 * np.sign(
                          np.imag(inx) - 4 * np.sign(np.imag(inx))))
                iny = iny + (- 4 * np.sign(np.real(iny)) - 2 * np.sign(
                    np.real(iny) - 4 * np.sign(np.real(iny)))) + 1j * (
                              - 4 * np.sign(np.imag(iny)) - 2 * np.sign(
                          np.imag(iny) - 4 * np.sign(np.imag(iny))))
                squ_Rx[indx] = np.mean(abs(inx) ** 4) / np.mean(abs(inx) ** 2)
                squ_Ry[indx] = np.mean(abs(iny) ** 4) / np.mean(abs(iny) ** 2)

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    np.conj(hxy), inputry[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    np.conj(hyy), inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                exout_adj[indx] = exout[indx] + (- 4 * np.sign(np.real(exout[indx])) - 2 * np.sign(
                    np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])))) + 1j * (
                                              - 4 * np.sign(np.imag(exout[indx])) - 2 * np.sign(
                                          np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx]))))

                eyout_adj[indx] = eyout[indx] + (- 4 * np.sign(np.real(eyout[indx])) - 2 * np.sign(
                    np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])))) + 1j * (
                                              - 4 * np.sign(np.imag(eyout[indx])) - 2 * np.sign(
                                          np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx]))))

                errx[indx] = (squ_Rx[indx] - np.abs(exout_adj[indx]) ** 2)
                erry[indx] = (squ_Ry[indx] - np.abs(eyout_adj[indx]) ** 2)

                hxx = hxx + self.stepsize_x * exout[indx] * errx[indx] * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                hxy = hxy + self.stepsize_x * exout[indx] * errx[indx] * np.conj(
                    inputry[indx - self.center:indx + self.center + 1])
                hyx = hyx + self.stepsize_y * eyout[indx] * erry[indx] * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy + self.stepsize_y * eyout[indx] * erry[indx] * np.conj(
                    inputry[indx - self.center:indx + self.center + 1])

                cost_x[indx] = (abs(exout[indx])) ** 2 - squ_Rx[indx]
                cost_y[indx] = (abs(eyout[indx])) ** 2 - squ_Ry[indx]
            self.costfunx[0][it] = -1 * (np.mean(cost_x))
            self.costfuny[0][it] = -1 * (np.mean(cost_y))
            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print(self.costfuny[0][it])
            print('-------')

            if it >= 1:
                # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                #         self.costfunx[0][it]:
                #     print("Earlybreak at iterator {}".format(it))
                #     break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < 0.001:
                    self.stepsize_x *= self.stepsizeadjust
                    print('Stepsize_x adjust to {}'.format(self.stepsize_x))
                if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < 0.001:
                    self.stepsize_y *= self.stepsizeadjust
                    print('Stepsize_y adjust to {}'.format(self.stepsize_y))

            self.rx_x_cma = exout
            self.rx_y_cma = eyout

    def qam_4_butter_conj(self):
            self.type = 'single_butter_conj'
            self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
            self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
            inputrx = self.rx_x_single
            inputry = self.rx_y_single
            hxx = np.zeros(self.cmataps, dtype="complex_")
            hxy = np.zeros(self.cmataps, dtype="complex_")
            hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.datalength, dtype="complex_")
            eyout = np.zeros(self.datalength, dtype="complex_")
            exiout = np.zeros(self.datalength, dtype="complex_")
            exqout = np.zeros(self.datalength, dtype="complex_")
            eyiout = np.zeros(self.datalength, dtype="complex_")
            eyqout = np.zeros(self.datalength, dtype="complex_")
            errx = np.zeros(self.datalength, dtype="complex_")
            erry = np.zeros(self.datalength, dtype="complex_")
            cost_x = np.zeros(self.datalength, dtype="complex_")
            cost_y = np.zeros(self.datalength, dtype="complex_")
            R = 10
            squ_Rxi, squ_Rxq = R, R
            squ_Ryi, squ_Ryq = R, R
            for it in range(self.iterator):
                for indx in range(self.center, self.datalength - self.center):
                    exout[indx] = np.matmul(np.conj(hxx), inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        np.conj(hxy), inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(np.conj(hyx), inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        np.conj(hyy), inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                    if self.mean == True:
                        inxi = np.real(inputrx[indx - self.center:indx + self.center + 1])
                        inxq = np.imag(inputrx[indx - self.center:indx + self.center + 1])
                        inyi = np.real(inputry[indx - self.center:indx + self.center + 1])
                        inyq = np.imag(inputry[indx - self.center:indx + self.center + 1])
                        squ_Rxi = np.mean(abs(inxi) ** 4) / np.mean(abs(inxi) ** 2)
                        squ_Rxq = np.mean(abs(inxq) ** 4) / np.mean(abs(inxq) ** 2)
                        squ_Ryi = np.mean(abs(inyi) ** 4) / np.mean(abs(inyi) ** 2)
                        squ_Ryq = np.mean(abs(inyq) ** 4) / np.mean(abs(inyq) ** 2)

                    exiout[indx] = np.real(exout[indx])
                    exqout[indx] = np.imag(exout[indx])
                    eyiout[indx] = np.real(eyout[indx])
                    eyqout[indx] = np.imag(eyout[indx])

                    errx[indx] = (squ_Rxi - np.abs(exiout[indx]) ** 2) + \
                                 1j * (squ_Rxq - np.abs(exqout[indx]) ** 2)
                    erry[indx] = (squ_Ryi - np.abs(eyiout[indx]) ** 2) + \
                                 1j * (squ_Ryq - np.abs(eyqout[indx]) ** 2)

                    hxx = hxx + self.stepsize * np.conj(exout[indx] * errx[indx]) * (
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy + self.stepsize * np.conj(exout[indx] * errx[indx]) * (
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx + self.stepsize * np.conj(eyout[indx] * erry[indx]) * (
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy + self.stepsize * np.conj(eyout[indx] * erry[indx]) * (
                        inputry[indx - self.center:indx + self.center + 1])

                    cost_x[indx] = ((exiout[indx]) ** 2 - squ_Rxi) + 1j * (exqout[indx] ** 2 - squ_Rxq)
                    cost_y[indx] = ((eyiout[indx]) ** 2 - squ_Ryi) + 1j * (eyqout[indx] ** 2 - squ_Ryq)
                self.costfunx[0][it] = -1 * (np.mean(np.real(cost_x)) + np.mean(np.imag(cost_x)))
                self.costfuny[0][it] = -1 * (np.mean(np.real(cost_y)) + np.mean(np.imag(cost_y)))
                print('iteration = {}'.format(it))
                print(self.costfunx[0][it])
                print(self.costfuny[0][it])
                print('-------')

                if it >= 1:
                    if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                            self.costfunx[0][it]:
                        print("Earlybreak at iterator {}".format(it))
                        break
                    if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.stepsizeadjust * \
                            self.costfunx[0][it]:
                        self.stepsize *= self.stepsizeadjust
                        print('Stepsize adjust to {}'.format(self.stepsize))

                self.rx_x_cma = exout
                self.rx_y_cma = eyout

    def qam_4_butter_conj_shift(self):
            self.type = 'single_butter_conj_shift'
            self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
            self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
            inputrx = self.rx_x_single
            inputry = self.rx_y_single
            hxx = np.zeros(self.cmataps, dtype="complex_")
            hxy = np.zeros(self.cmataps, dtype="complex_")
            hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.datalength, dtype="complex_")
            eyout = np.zeros(self.datalength, dtype="complex_")
            # exiout = np.zeros(self.datalength, dtype="complex_")
            # exqout = np.zeros(self.datalength, dtype="complex_")
            # eyiout = np.zeros(self.datalength, dtype="complex_")
            # eyqout = np.zeros(self.datalength, dtype="complex_")
            exiout_adj = np.zeros(self.datalength, dtype="complex_")
            exqout_adj = np.zeros(self.datalength, dtype="complex_")
            eyiout_adj = np.zeros(self.datalength, dtype="complex_")
            eyqout_adj = np.zeros(self.datalength, dtype="complex_")
            errx = np.zeros(self.datalength, dtype="complex_")
            erry = np.zeros(self.datalength, dtype="complex_")
            cost_x = np.zeros(self.datalength, dtype="complex_")
            cost_y = np.zeros(self.datalength, dtype="complex_")
            squ_Rxi, squ_Rxq = np.zeros(self.datalength, dtype="complex_"), np.zeros(self.datalength, dtype="complex_")
            squ_Ryi, squ_Ryq = np.zeros(self.datalength, dtype="complex_"), np.zeros(self.datalength, dtype="complex_")
            squ_Rxi, squ_Rxq = squ_Rxi + 10 ** 0.5 * 0.5, squ_Rxq + 10 ** 0.5 * 0.5
            squ_Ryi, squ_Ryq = squ_Ryi + 10 ** 0.5 * 0.5, squ_Ryq + 10 ** 0.5 * 0.5

            if self.mean == True:
                squ_Rxi, squ_Rxq = np.zeros(self.datalength, dtype="complex_"), np.zeros(self.datalength, dtype="complex_")
                squ_Ryi, squ_Ryq = np.zeros(self.datalength, dtype="complex_"), np.zeros(self.datalength, dtype="complex_")
                for indx in range(self.center, self.datalength - self.center):
                    inxi = np.real(inputrx[indx - self.center:indx + self.center + 1])
                    inxq = np.imag(inputrx[indx - self.center:indx + self.center + 1])
                    inyi = np.real(inputry[indx - self.center:indx + self.center + 1])
                    inyq = np.imag(inputry[indx - self.center:indx + self.center + 1])
                    # inxi = inxi - 4 * np.sign(inxi) - 2 * np.sign(inxi - 4 * np.sign(inxi))
                    # inxq = inxq - 4 * np.sign(inxq) - 2 * np.sign(inxq - 4 * np.sign(inxq))
                    # inyi = inyi - 4 * np.sign(inyi) - 2 * np.sign(inyi - 4 * np.sign(inyi))
                    # inyq = inyq - 4 * np.sign(inyq) - 2 * np.sign(inyq - 4 * np.sign(inyq))
                    inxi = inxi - 2 * np.sign(inxi)
                    inxq = inxq - 2 * np.sign(inxq)
                    inyi = inyi - 2 * np.sign(inyi)
                    inyq = inyq - 2 * np.sign(inyq)
                    squ_Rxi[indx] = np.mean(abs(inxi) ** 4) / np.mean(abs(inxi) ** 2)
                    squ_Rxq[indx] = np.mean(abs(inxq) ** 4) / np.mean(abs(inxq) ** 2)
                    squ_Ryi[indx] = np.mean(abs(inyi) ** 4) / np.mean(abs(inyi) ** 2)
                    squ_Ryq[indx] = np.mean(abs(inyq) ** 4) / np.mean(abs(inyq) ** 2)

            for it in range(self.iterator):
                for indx in range(self.center, self.datalength - self.center):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                    # exiout_adj[indx] = np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])) - 2 * \
                    #                    np.sign(np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])))
                    # exqout_adj[indx] = np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])) - 2 * \
                    #                    np.sign(np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])))
                    # eyiout_adj[indx] = np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])) - 2 * \
                    #                    np.sign(np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])))
                    # eyqout_adj[indx] = np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])) - 2 * \
                    #                    np.sign(np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])))

                    exiout_adj[indx] = np.real(exout[indx]) - 2 * np.sign(np.real(exout[indx]))
                    exqout_adj[indx] = np.imag(exout[indx]) - 2 * np.sign(np.imag(exout[indx]))
                    eyiout_adj[indx] = np.real(eyout[indx]) - 2 * np.sign(np.real(eyout[indx]))
                    eyqout_adj[indx] = np.imag(eyout[indx]) - 2 * np.sign(np.imag(eyout[indx]))

                    errx[indx] = np.real(exout[indx]) * (squ_Rxi[indx] - np.abs(exiout_adj[indx]) ** 2) + \
                                 1j * np.imag(exout[indx]) * (squ_Rxq[indx] - np.abs(exqout_adj[indx]) ** 2)
                    erry[indx] = np.real(eyout[indx]) * (squ_Ryi[indx] - np.abs(eyiout_adj[indx]) ** 2) + \
                                 1j * np.imag(eyout[indx]) * (squ_Ryq[indx] - np.abs(eyqout_adj[indx]) ** 2)

                    hxx = hxx + self.stepsize_x * errx[indx] * np.conj(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy + self.stepsize_x * errx[indx] * np.conj(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx + self.stepsize_y * erry[indx] * np.conj(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy + self.stepsize_y * erry[indx] * np.conj(
                        inputry[indx - self.center:indx + self.center + 1])

                    cost_x[indx] = (exiout_adj[indx] ** 2 - squ_Rxi[indx]) + 1j * (exqout_adj[indx] ** 2 - squ_Rxq[indx])
                    cost_y[indx] = (eyiout_adj[indx] ** 2 - squ_Ryi[indx]) + 1j * (eyqout_adj[indx] ** 2 - squ_Ryq[indx])
                self.costfunx[0][it] = -1 * (np.mean(np.real(cost_x)) + np.mean(np.imag(cost_x)))
                self.costfuny[0][it] = -1 * (np.mean(np.real(cost_y)) + np.mean(np.imag(cost_y)))
                print('iteration = {}'.format(it))
                print(self.costfunx[0][it])
                print(self.costfuny[0][it])
                print('-------')

                if it >= 1:
                    # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                    #         self.costfunx[0][it]:
                    #     print("Earlybreak at iterator {}".format(it))
                    #     break
                    if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop:
                        self.stepsize_x *= self.stepsizeadjust
                        print('Stepsize_x adjust to {}'.format(self.stepsize_x))
                    if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop:
                        self.stepsize_y *= self.stepsizeadjust
                        print('Stepsize_y adjust to {}'.format(self.stepsize_y))

            self.rx_x_cma = exout
            self.rx_y_cma = eyout

    def MCMA_MDD_(self):
        self.type = 'MCMA_MDD_'
        R = 1
        Decision = [R + 1j * R, -R + 1j * R, -R - 1j * R, R - 1j * R]
        self.costfunx_mcma = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny_mcma = np.zeros((1, self.iterator), dtype="complex_")
        self.costfunx_mdd = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny_mdd = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        # -------h
        hxx_mcma = np.zeros(self.cmataps, dtype="complex_")
        hyy_mcma = np.zeros(self.cmataps, dtype="complex_")
        hxx_mcma[self.center] = 1
        hyy_mcma[self.center] = 1
        hxx_mdd = np.zeros(self.cmataps, dtype="complex_")
        hyy_mdd = np.zeros(self.cmataps, dtype="complex_")
        hxx_mdd[self.center] = 1
        hyy_mdd[self.center] = 1
        # -------
        exout_mcma = np.zeros(self.datalength, dtype="complex_")
        eyout_mcma = np.zeros(self.datalength, dtype="complex_")
        exout_mdd = np.zeros(self.datalength, dtype="complex_")
        eyout_mdd = np.zeros(self.datalength, dtype="complex_")
        # ------
        exout_adj_mcma = np.zeros(self.datalength, dtype="complex_")
        eyout_adj_mcma = np.zeros(self.datalength, dtype="complex_")
        exout_adj_mdd = np.zeros(self.datalength, dtype="complex_")
        eyout_adj_mdd = np.zeros(self.datalength, dtype="complex_")
        # -------
        # errx_mcma = np.zeros(self.datalength, dtype="complex_")
        # erry_mcma = np.zeros(self.datalength, dtype="complex_")
        # errx_mdd = np.zeros(self.datalength, dtype="complex_")
        # erry_mdd = np.zeros(self.datalength, dtype="complex_")
        # -------
        cost_x_mcma = np.zeros(self.datalength)
        cost_y_mcma = np.zeros(self.datalength)
        cost_x_mdd = np.zeros(self.datalength)
        cost_y_mdd = np.zeros(self.datalength)
        squ_Rx = np.zeros(self.datalength, dtype="complex_")
        squ_Ry = np.zeros(self.datalength, dtype="complex_")
        squ_Rx, squ_Ry = squ_Rx + 50, squ_Ry + 50
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout_mcma[indx] = np.matmul(np.conj(hxx_mcma), inputrx[indx - self.center:indx + self.center + 1]) + \
                                   np.matmul(np.conj(hxx_mdd), inputrx[indx - self.center:indx + self.center + 1])
                # eyout_mcma[indx] = np.matmul(np.conj(hyy_mcma), inputry[indx - self.center:indx + self.center + 1]) + \
                #                    np.matmul(np.conj(hyy_mdd), inputry[indx - self.center:indx + self.center + 1])
                # if np.isnan(exout_mcma[indx]) or np.isnan(eyout_mcma[indx]):
                if np.isnan(exout_mcma[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                if self.mean == True:
                    inx = inputrx[indx - self.center:indx + self.center + 1]
                    iny = inputry[indx - self.center:indx + self.center + 1]
                    squ_Rx = np.mean(abs(inx) ** 4) / np.mean(abs(inx) ** 2)
                    squ_Ry = np.mean(abs(iny) ** 4) / np.mean(abs(iny) ** 2)

                errx_mcma = exout_mcma[indx] * (squ_Rx[indx] - np.abs(exout_mcma[indx]) ** 2)
                # erry_mcma = (squ_Ry - np.abs(eyout_mcma[indx]) ** 2)

                hxx_mcma = hxx_mcma + self.stepsize * exout_mcma[indx]  * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                # hyy_mcma = hyy_mcma + self.stepsize * eyout_mcma[indx]  * np.conj(
                #     inputry[indx - self.center:indx + self.center + 1])

                cost_x_mcma[indx] = (abs(exout_mcma[indx])) ** 2 - squ_Rx
                # cost_y_mcma[indx] = (abs(eyout_mcma[indx])) ** 2 - squ_Ry
                # --------------------------CMA-------------------------
                # --------------------------DD-------------------------
                exout_mdd[indx] = np.matmul(np.conj(hxx_mcma), inputrx[indx - self.center:indx + self.center + 1]) + \
                                  np.matmul(np.conj(hxx_mdd), inputrx[indx - self.center:indx + self.center + 1])

                # eyout_mdd[indx] = np.matmul(np.conj(hyy_mcma), inputry[indx - self.center:indx + self.center + 1]) + \
                #                   np.matmul(np.conj(hyy_mdd), inputry[indx - self.center:indx + self.center + 1])
                # if np.isnan(exout_mdd[indx]) or np.isnan(eyout_mdd[indx]):
                if np.isnan(exout_mdd[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                # #---------------------------------------------------

                # exout_adj_mcma[indx] = exout_mcma[indx] + (- 4 * np.sign(np.real(exout_mcma[indx])) - 2 * np.sign(
                #     np.real(exout_mcma[indx]) - 4 * np.sign(np.real(exout_mcma[indx])))) + 1j * (
                #                               - 4 * np.sign(np.imag(exout_mcma[indx])) - 2 * np.sign(
                #                           np.imag(exout_mcma[indx]) - 4 * np.sign(np.imag(exout_mcma[indx]))))
                # eyout_adj_mcma[indx] = eyout_mcma[indx] + (- 4 * np.sign(np.real(eyout_mcma[indx])) - 2 * np.sign(
                #     np.real(eyout_mcma[indx]) - 4 * np.sign(np.real(eyout_mcma[indx])))) + 1j * (
                #                               - 4 * np.sign(np.imag(eyout_mcma[indx])) - 2 * np.sign(
                #                           np.imag(eyout_mcma[indx]) - 4 * np.sign(np.imag(eyout_mcma[indx]))))

                # exout_adj_mdd[indx] = exout_mdd[indx] + (- 4 * np.sign(np.real(exout_mdd[indx])) - 2 * np.sign(
                #     np.real(exout_mdd[indx]) - 4 * np.sign(np.real(exout_mdd[indx])))) + 1j * (
                #                               - 4 * np.sign(np.imag(exout_mdd[indx])) - 2 * np.sign(
                #                           np.imag(exout_mdd[indx]) - 4 * np.sign(np.imag(exout_mdd[indx]))))
                # eyout_adj_mdd[indx] = eyout_mdd[indx] + (- 4 * np.sign(np.real(eyout_mdd[indx])) - 2 * np.sign(
                #     np.real(eyout_mdd[indx]) - 4 * np.sign(np.real(eyout_mdd[indx])))) + 1j * (
                #                               - 4 * np.sign(np.imag(eyout_mdd[indx])) - 2 * np.sign(
                #                           np.imag(eyout_mdd[indx]) - 4 * np.sign(np.imag(eyout_mdd[indx]))))

                HardDecision_mcma_x = [np.abs(exout_adj_mcma[indx] - Decision[0]),np.abs(exout_adj_mcma[indx] - Decision[1]),
                                       np.abs(exout_adj_mcma[indx] - Decision[2]),np.abs(exout_adj_mcma[indx] - Decision[3])]
                HardDecision_mcma_y = [np.abs(eyout_adj_mcma[indx] - Decision[0]),np.abs(eyout_adj_mcma[indx] - Decision[1]),
                                       np.abs(eyout_adj_mcma[indx] - Decision[2]),np.abs(eyout_adj_mcma[indx] - Decision[3])]
                HardDecision_mdd_x  = [np.abs(exout_adj_mdd [indx] - Decision[0]),np.abs(exout_adj_mdd [indx] - Decision[1]),
                                       np.abs(exout_adj_mdd [indx] - Decision[2]),np.abs(exout_adj_mdd [indx] - Decision[3])]
                HardDecision_mdd_y  = [np.abs(eyout_adj_mdd [indx] - Decision[0]),np.abs(eyout_adj_mdd [indx] - Decision[1]),
                                       np.abs(eyout_adj_mdd [indx] - Decision[2]),np.abs(eyout_adj_mdd [indx] - Decision[3])]

                errx_mdd[indx] = np.abs(Decision[np.argmin(HardDecision_mdd_x)])-np.abs(exout_mdd[indx]) ** 2
                erry_mdd[indx] = np.abs(Decision[np.argmin(HardDecision_mdd_y)])-np.abs(eyout_mdd[indx]) ** 2

                if Decision[np.argmin(HardDecision_mcma_x)] == Decision[np.argmin(HardDecision_mdd_x)]:
                    hxx_mdd = hxx_mdd - self.stepsize * np.conj(exout_mdd[indx] * errx_mdd[indx]) * (
                        inputrx[indx - self.center:indx + self.center + 1])
                if Decision[np.argmin(HardDecision_mcma_y)] == Decision[np.argmin(HardDecision_mdd_y)]:
                    hyy_mdd = hyy_mdd - self.stepsize * np.conj(eyout_mdd[indx] * errx_mdd[indx]) * (
                        inputry[indx - self.center:indx + self.center + 1])

                cost_x_mdd[indx] = (abs(exout_mdd[indx])) ** 2 - np.abs(Decision[np.argmin(HardDecision_mdd_x)])
                cost_y_mdd[indx] = (abs(eyout_mdd[indx])) ** 2 - np.abs(Decision[np.argmin(HardDecision_mdd_y)])

            # --------------------------DD-------------------------
            print('iteration = {}'.format(it))
            self.costfunx_mcma[0][it] = -1 * (np.mean(cost_x_mcma))
            self.costfuny_mcma[0][it] = -1 * (np.mean(cost_y_mcma))
            print('CMA costfunc X={}'.format(self.costfunx_mcma[0][it]))
            print('CMA costfunc Y={}'.format(self.costfuny_mcma[0][it]))
            self.costfunx_mdd[0][it] = -1 * (np.mean(cost_x_mdd))
            self.costfuny_mdd[0][it] = -1 * (np.mean(cost_y_mdd))
            print('DD  costfunc X={}'.format(self.costfunx_mdd[0][it]))
            print('DD  costfunc Y={}'.format(self.costfuny_mdd[0][it]))
            print('-------')

            #     if it >= 1:
            #         if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
            #                 self.costfunx[batch][it] and np.abs(
            #             self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
            #                 self.costfuny[batch][it]:
            #             print("Earlybreak at iterator {}".format(it))
            #             break
            #         if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
            #                 self.costfunx[batch][it] and np.abs(
            #             self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
            #                 self.costfuny[batch][it]:
            #             self.stepsize *= self.stepsizeadjust
            #             print('Stepsize adjust to {}'.format(self.stepsize))

            # hxx_mdd = np.zeros(self.cmataps, dtype="complex_")
            # hxy_mdd = np.zeros(self.cmataps, dtype="complex_")
            # hyx_mdd = np.zeros(self.cmataps, dtype="complex_")
            # hyy_mdd = np.zeros(self.cmataps, dtype="complex_")

            self.rx_x_cma = exout_mcma
            self.rx_y_cma = eyout_mcma

    def MCMA_SBD(self):
        self.type = 'MCMA_SBD'

        self.costfunx_mcma = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny_mcma = np.zeros((1, self.iterator), dtype="complex_")
        self.costfunx_mdd = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny_mdd = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        # -------h
        hxx_mcma = np.zeros(self.cmataps, dtype="complex_")
        hyy_mcma = np.zeros(self.cmataps, dtype="complex_")
        hxx_mcma[self.center] = 1
        hyy_mcma[self.center] = 1
        hxx_mdd = np.zeros(self.cmataps, dtype="complex_")
        hyy_mdd = np.zeros(self.cmataps, dtype="complex_")
        hxx_mdd[self.center] = 1
        hyy_mdd[self.center] = 1
        # -------
        exout_mcma = np.zeros(self.datalength, dtype="complex_")
        eyout_mcma = np.zeros(self.datalength, dtype="complex_")
        exout_mdd = np.zeros(self.datalength, dtype="complex_")
        eyout_mdd = np.zeros(self.datalength, dtype="complex_")
        # -------
        cost_x_mcma = np.zeros(self.datalength, dtype="complex_")
        cost_y_mcma = np.zeros(self.datalength, dtype="complex_")
        cost_x_mdd = np.zeros(self.datalength, dtype="complex_")
        cost_y_mdd = np.zeros(self.datalength, dtype="complex_")
        squ_Rx = np.zeros(self.datalength, dtype="complex_")
        squ_Ry = np.zeros(self.datalength, dtype="complex_")
        squ_Rx, squ_Ry = squ_Rx + 50, squ_Ry + 50
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                # print(indx)
                exout_mcma[indx] = np.matmul(hxx_mcma, inputrx[indx - self.center:indx + self.center + 1]) + \
                                   np.matmul(hxx_mdd, inputrx[indx - self.center:indx + self.center + 1])

                if np.isnan(exout_mcma[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                errx_mcma = exout_mcma[indx] * (squ_Rx[indx] - np.abs(exout_mcma[indx]) ** 2)

                hxx_mcma = hxx_mcma + self.stepsize * errx_mcma * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])

                cost_x_mcma[indx] = (abs(exout_mcma[indx])) ** 2 - squ_Rx[indx]
                # --------------------------CMA-------------------------
                # --------------------------DD-------------------------
                exout_mdd[indx] = np.matmul(hxx_mcma, inputrx[indx - self.center:indx + self.center + 1]) + \
                                   np.matmul(hxx_mdd, inputrx[indx - self.center:indx + self.center + 1])

                if np.isnan(exout_mdd[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                # #---------------------------------------------------
                Decision = [-7, -5, -3, -1, 1, 3, 5, 7]
                HardDecision_X_real = np.zeros(len(Decision))
                HardDecision_X_imag = np.zeros(len(Decision))
                HardDecision_X_real_mcama = np.zeros(len(Decision))
                HardDecision_X_imag_mcama = np.zeros(len(Decision))


                for i in range(len(Decision)):
                    HardDecision_X_real[i] = Decision[i] - np.real(exout_mdd[indx])
                    HardDecision_X_imag[i] = Decision[i] - np.imag(exout_mdd[indx])
                    HardDecision_X_real_mcama[i] = Decision[i] - np.real(exout_mcma[indx])
                    HardDecision_X_imag_mcama[i] = Decision[i] - np.imag(exout_mcma[indx])

                if (Decision[np.argmin(abs(HardDecision_X_real))] == Decision[np.argmin(abs(HardDecision_X_real_mcama))]) and (Decision[np.argmin(abs(HardDecision_X_imag))] == Decision[np.argmin(abs(HardDecision_X_imag_mcama))]) :
                    errx_mdd = np.abs(Decision[np.argmin(abs(HardDecision_X_real))]) * (Decision[np.argmin(abs(HardDecision_X_real))] - np.real(exout_mdd[indx])) + \
                          1j * np.abs(Decision[np.argmin(abs(HardDecision_X_imag))]) * (Decision[np.argmin(abs(HardDecision_X_imag))] - np.imag(exout_mdd[indx]))

                    hxx_mdd = hxx_mdd + self.stepsize_x * errx_mdd * np.conj(
                        inputrx[indx - self.center:indx + self.center + 1])

                cost_x_mdd[indx] = (np.real(exout_mdd[indx]) - Decision[np.argmin(abs(HardDecision_X_real))]) ** 2 + 1j * (
                            np.imag(exout_mdd[indx]) - Decision[np.argmin(abs(HardDecision_X_imag))]) ** 2

            # --------------------------DD-------------------------
            print('iteration = {}'.format(it))
            self.costfunx_mcma[0][it] = -1 * (np.mean(cost_x_mcma))
            print('CMA costfunc X={}'.format(self.costfunx_mcma[0][it]))
            self.costfunx_mdd[0][it] = (np.mean(np.real(cost_x_mdd))) + (np.mean(np.imag(cost_x_mdd)))
            print('DD  costfunc X={}'.format(self.costfunx_mdd[0][it]))
            print('-------')

            if it >= 1:
                # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                #         self.costfunx[0][it]:
                #     print("Earlybreak at iterator {}".format(it))
                #     break
                if np.abs(self.costfunx_mcma[0][it] - self.costfunx_mcma[0][it - 1]) < self.earlystop:
                    self.stepsize_x *= self.stepsizeadjust
                    print('Stepsize_x adjust to {}'.format(self.stepsize_x))
            #     if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop:
            #         self.stepsize_y *= self.stepsizeadjust
            #         print('Stepsize_y adjust to {}'.format(self.stepsize_y))

            self.rx_x_cma = exout_mcma
            # self.rx_y_cma = eyout_mcma
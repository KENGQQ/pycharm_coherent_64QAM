import numpy as np
import cmath
import math

class CMA_t:
    def __init__(self, rx_x, rx_y=[], taps=47):
        self.rx_x_single = np.array(rx_x)
        self.rx_y_single = np.array(rx_y)
        self.rx_x = np.array(rx_x)
        self.rx_y = np.array(rx_y)
        self.datalength = len(rx_x)
        self.stepsizelist = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
        # self.batchsize = 32767*2
        self.batchsize = self.datalength
        self.overhead = 0.95
        self.trainlength = round(self.batchsize * self.overhead)
        self.cmataps = taps
        self.center = int((self.cmataps-1)/2)
        self.batchnum = int(self.datalength/self.batchsize)
        self.iterator = 50
        self.earlystop = 0.001
        self.stepsizeadjust = 0
        self.rx_x.resize((self.batchnum, self.batchsize), refcheck=False)
        self.rx_y.resize((self.batchnum, self.batchsize), refcheck=False)
        self.stepsize = self.stepsizelist[2]

    def run(self):
        self.costfunx = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize-self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize-self.center), dtype="complex_")
        for batch in range(self.batchnum):
            #initialize H
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
                    errx[indx] = (10 - (np.abs(exout[indx]))**2)
                    erry[indx] = (10 - (np.abs(eyout[indx]))**2)
                    hxx = hxx + self.stepsize * errx[indx] * exout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy + self.stepsize * errx[indx] * exout[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx + self.stepsize * erry[indx] * eyout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy + self.stepsize * erry[indx] * eyout[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                self.costfunx[batch][it] = np.mean((errx[self.center:])**2)
                self.costfuny[batch][it] = np.mean((erry[self.center:])**2)
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
            for indx in range(self.center, self.batchsize - self.center):  #taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def run_single(self):
        #initialize H
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
        errx = np.zeros(self.datalength, dtype="float32")
        erry = np.zeros(self.datalength, dtype="float32")
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength-self.center):
                exout[indx] = np.matmul(np.conjugate(hxx), inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    np.conjugate(hxy), inputry[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(np.conjugate(hyx), inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    np.conjugate(hyy), inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                errx[indx] = ((np.abs(exout[indx]))**2 - 10 )
                erry[indx] = ((np.abs(eyout[indx]))**2 - 10)
                hxx = hxx - self.stepsize * np.conjugate(errx[indx] * exout[indx]) * inputrx[
                                                                                     indx - self.center:indx + self.center + 1]
                hxy = hxy - self.stepsize * np.conjugate(errx[indx] * exout[indx]) * inputry[
                                                                                     indx - self.center:indx + self.center + 1]
                hyx = hyx - self.stepsize * np.conjugate(erry[indx] * eyout[indx]) * inputrx[
                                                                                     indx - self.center:indx + self.center + 1]
                hyy = hyy - self.stepsize * np.conjugate(erry[indx] * eyout[indx]) * inputry[
                                                                                     indx - self.center:indx + self.center + 1]
            self.costfunx[0][it] = np.mean((errx[self.center:])**2)
            self.costfuny[0][it] = np.mean((erry[self.center:])**2)
            if it > 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                        self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
            print(self.costfunx[0][it])
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]

    def run_single_ch(self):
        #initialize H
        self.costfunx = np.zeros((1, self.iterator), dtype="complex")
        inputrx = self.rx_x_single
        hxx = np.zeros(self.cmataps, dtype="complex")
        hxx[self.center] = 1 + 0j
        exout = np.zeros(self.datalength, dtype="complex")
        errx = np.zeros(self.datalength, dtype="complex")
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength-self.center):
                exout[indx] = np.matmul(np.conjugate(hxx), inputrx[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                errx[indx] = (np.abs(exout[indx])) ** 2 - 10
                hxx = hxx - self.stepsize * np.conjugate(exout[indx] * errx[indx]) * inputrx[indx - self.center:indx + self.center + 1]
            self.costfunx[0][it] = np.mean((errx[self.center:])**2)
            if it > 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
            print(self.costfunx[0][it])
        self.rx_x_single = exout[self.center:-self.center]

    def run_16qam_single(self):
        # initialize H
        self.rx_x_cma = np.zeros((1, self.datalength), dtype="complex_")
        self.rx_y_cma = np.zeros((1, self.datalength), dtype="complex_")
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
        radx = np.zeros(self.datalength, dtype="complex_")
        rady = np.zeros(self.datalength, dtype="complex_")
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                exin = inputrx[indx - self.center:indx + self.center + 1]
                eyin = inputry[indx - self.center:indx + self.center + 1]
                rad_aaa = np.real(exin) - 2 * np.sign(np.real(exin)) + 1j * (
                        np.imag(exin) - 2 * np.sign(np.imag(exin)))
                rad_bbb = np.real(eyin) - 2 * np.sign(np.real(eyin)) + 1j * (
                        np.imag(eyin) - 2 * np.sign(np.imag(eyin)))
                radx[indx] = np.mean(np.abs(rad_aaa ** 4)) / np.mean(np.abs(rad_aaa ** 2))
                rady[indx] = np.mean(np.abs(rad_bbb ** 4)) / np.mean(np.abs(rad_bbb ** 2))
                exout_adj = np.real(exout[indx]) - 2 * np.sign(np.real(exout[indx])) + 1j * (
                        np.imag(exout[indx]) - 2 * np.sign(np.imag(exout[indx])))
                eyout_adj = np.real(eyout[indx]) - 2 * np.sign(np.real(eyout[indx])) + 1j * (
                        np.imag(eyout[indx]) - 2 * np.sign(np.imag(eyout[indx])))
                exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                eyout_adj2 = eyout[indx] - 2 * np.sign(np.real(eyout[indx])) - 2j * (np.sign(np.imag(eyout[indx])))
                errx[indx] = ((np.abs(exout_adj)) ** 2 - radx[indx])
                erry[indx] = ((np.abs(eyout_adj)) ** 2 - rady[indx])
                hxx = hxx - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                    inputrx[indx - self.center:indx + self.center + 1])
                hxy = hxy - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                    inputry[indx - self.center:indx + self.center + 1])
                hyx = hyx - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                    inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                    inputry[indx - self.center:indx + self.center + 1])
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            print(self.costfunx[0][it])
            if it > 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.stepsizeadjust * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.stepsizeadjust * \
                        self.costfuny[0][it]:
                    self.stepsize *= 0.5
                    print('Stepsize adjust to {}'.format(self.stepsize))
        self.rx_x_cma = exout[:-self.center]
        self.rx_y_cma = eyout[:-self.center]

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
                    errx[indx] = ((np.abs(exout_adj)) ** 2 - 2)
                    erry[indx] = ((np.abs(eyout_adj)) ** 2 - 2)
                    hxx = hxx - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
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
                    # if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                    #         self.costfunx[batch][it] and np.abs(
                    #     self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
                    #         self.costfuny[batch][it]:
                    #     self.stepsize *= 0.5
                    #     print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def run_16qam_single_ch(self):
        self.costfunx = np.zeros((1, self.iterator), dtype="complex")
        self.rx_x_cma = np.zeros((1, self.datalength), dtype="complex_")
        inputrx = self.rx_x_single
        # initialize H
        inputrx = self.rx_x_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(np.conjugate(hxx), inputrx[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                exout_adj = np.real(exout[indx]) - 2 * np.sign(np.real(exout[indx])) + 1j * (
                            np.imag(exout[indx]) - 2 * np.sign(np.imag(exout[indx])))
                exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                errx[indx] = ((np.abs(exout_adj)) ** 2 - 10)
                hxx = hxx - self.stepsize * np.conjugate(errx[indx] * exout_adj2) * inputrx[indx - self.center:indx + self.center + 1]
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            print(self.costfunx[0][it])
            if it > 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.stepsizeadjust * \
                        self.costfunx[0][it]:
                    self.stepsize *= 0.5
                    print('Stepsize adjust to {}'.format(self.stepsize))
        self.rx_x_single = exout[self.center:-self.center]

    def run_16qam_butterfly(self):
        self.rx_x_cma = np.zeros((1, self.datalength), dtype="complex_")
        self.rx_y_cma = np.zeros((1, self.datalength), dtype="complex_")
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
                exout[indx] = np.matmul(np.conj(hxx), inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    np.conj(hxy), inputry[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(np.conj(hyx), inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    np.conj(hyy), inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                exout_adj = np.real(exout[indx]) - 2 * np.sign(np.real(exout[indx])) + 1j * (
                        np.imag(exout[indx]) - 2 * np.sign(np.imag(exout[indx])))
                eyout_adj = np.real(eyout[indx]) - 2 * np.sign(np.real(eyout[indx])) + 1j * (
                        np.imag(eyout[indx]) - 2 * np.sign(np.imag(eyout[indx])))
                exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                eyout_adj2 = eyout[indx] - 2 * np.sign(np.real(eyout[indx])) - 2j * (np.sign(np.imag(eyout[indx])))
                errx[indx] = ((np.abs(exout_adj)) ** 2 - 10)
                erry[indx] = ((np.abs(eyout_adj)) ** 2 - 10)
                hxx = hxx - self.stepsize * np.conj(errx[indx] * exout_adj2) * inputrx[indx - self.center:indx + self.center + 1]
                hxy = hxy - self.stepsize * np.conj(errx[indx] * exout_adj2) * inputry[indx - self.center:indx + self.center + 1]
                hyx = hyx - self.stepsize * np.conj(erry[indx] * eyout_adj2) * inputrx[indx - self.center:indx + self.center + 1]
                hyy = hyy - self.stepsize * np.conj(erry[indx] * eyout_adj2) * inputry[indx - self.center:indx + self.center + 1]
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            print(self.costfunx[0][it])
            if it > 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.stepsizeadjust * \
                #         self.costfunx[0][it] and np.abs(
                #     self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.stepsizeadjust * \
                #         self.costfuny[0][it]:
                #     self.stepsize *= 0.5
                #     print('Stepsize adjust to {}'.format(self.stepsize))
        self.rx_x_cma = exout[self.center:-self.center]
        self.rx_y_cma = eyout[self.center:-self.center]

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
            R = [2, 10, 18]
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
                    errx[indx] = (R[np.argmin(xdistance)] - np.abs(exout[indx])**2)
                    erry[indx] = (R[np.argmin(ydistance)] - np.abs(eyout[indx])**2)
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
                        self.stepsize *= 0.5
                        print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def RDE_single(self):
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        # initialize H
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
        R = [2, 10, 18]
        for it in range(self.iterator):
            if it < 10:
                for indx in range(self.center, self.datalength - self.center):
                    exout[indx] = np.matmul(np.conjugate(hxx),
                                            inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        np.conjugate(hxy), inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(np.conjugate(hyx),
                                            inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        np.conjugate(hyy), inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    errx[indx] = ((np.abs(exout[indx])) ** 2 - 10)
                    erry[indx] = ((np.abs(eyout[indx])) ** 2 - 10)
                    hxx = hxx - self.stepsize * np.conjugate(errx[indx] * exout[indx]) * inputrx[
                                                                                         indx - self.center:indx + self.center + 1]
                    hxy = hxy - self.stepsize * np.conjugate(errx[indx] * exout[indx]) * inputry[
                                                                                         indx - self.center:indx + self.center + 1]
                    hyx = hyx - self.stepsize * np.conjugate(erry[indx] * eyout[indx]) * inputrx[
                                                                                         indx - self.center:indx + self.center + 1]
                    hyy = hyy - self.stepsize * np.conjugate(erry[indx] * eyout[indx]) * inputry[
                                                                                         indx - self.center:indx + self.center + 1]
            else:
                for indx in range(self.center, self.datalength - self.center):
                    exout[indx] = np.matmul(np.conj(hxx), inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        np.conj(hxy), inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(np.conj(hyx), inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        np.conj(hyy), inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    xdistance = [np.abs(np.abs(exout[indx]) - R[0]), np.abs(np.abs(exout[indx]) - R[1]),
                                 np.abs(np.abs(exout[indx]) - R[2])]
                    ydistance = [np.abs(np.abs(eyout[indx]) - R[0]), np.abs(np.abs(eyout[indx]) - R[1]),
                                 np.abs(np.abs(eyout[indx]) - R[2])]
                    errx[indx] = np.abs(exout[indx])**2 - R[np.argmin(xdistance)]
                    erry[indx] = np.abs(eyout[indx])**2 - R[np.argmin(ydistance)]
                    hxx = hxx - self.stepsize * np.conj(errx[indx] * exout[indx]) * inputrx[
                                                                                    indx - self.center:indx + self.center + 1]
                    hxy = hxy - self.stepsize * np.conj(errx[indx] * exout[indx]) * inputry[
                                                                                    indx - self.center:indx + self.center + 1]
                    hyx = hyx - self.stepsize * np.conj(erry[indx] * eyout[indx]) * inputrx[
                                                                                    indx - self.center:indx + self.center + 1]
                    hyy = hyy - self.stepsize * np.conj(erry[indx] * eyout[indx]) * inputry[
                                                                                    indx - self.center:indx + self.center + 1]
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            print(self.costfunx[0][it])
            if it > 20:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
            #     if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
            #             self.costfunx[batch][it] and np.abs(
            #         self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
            #             self.costfuny[batch][it]:
            #         self.stepsize *= 0.5
            #         print('Stepsize adjust to {}'.format(self.stepsize))
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]


    def run_RLS(self):
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
            radx = np.zeros(self.trainlength + self.center, dtype="complex_")
            rady = np.zeros(self.trainlength + self.center, dtype="complex_")
            for it in range(self.iterator):
                for indx in range(self.center, self.center + self.trainlength):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    exin = inputrx[indx - self.center:indx + self.center + 1]
                    eyin = inputry[indx - self.center:indx + self.center + 1]
                    rad_aaa = np.real(exin) - 2 * np.sign(np.real(exin)) + 1j * (
                            np.imag(exin) - 2 * np.sign(np.imag(exin)))
                    rad_bbb = np.real(eyin) - 2 * np.sign(np.real(eyin)) + 1j * (
                            np.imag(eyin) - 2 * np.sign(np.imag(eyin)))
                    radx[indx] = np.mean(np.abs(rad_aaa ** 8)) / np.mean(np.abs(rad_aaa ** 8))
                    rady[indx] = np.mean(np.abs(rad_bbb ** 8)) / np.mean(np.abs(rad_bbb ** 8))
                    exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                    eyout_adj2 = eyout[indx] - 2 * np.sign(np.real(eyout[indx])) - 2j * (np.sign(np.imag(eyout[indx])))
                    errx[indx] = ((np.abs(exout_adj2)) ** 4 - radx[indx]) * np.abs(exout_adj2)**2
                    erry[indx] = ((np.abs(eyout_adj2)) ** 4 - rady[indx]) * np.abs(exout_adj2)**2
                    hxx = hxx - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
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
                        self.stepsize *= 0.5
                        print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])


    def run_CIAEMCMA(self):
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
            radx = np.zeros(self.trainlength + self.center, dtype="complex_")
            rady = np.zeros(self.trainlength + self.center, dtype="complex_")
            stepsizex = 1e-4
            stepsizey = 1e-4
            maxstepsize = 0.08
            minstepsize = 1e-6
            gama = 0.1
            for it in range(self.iterator):
                for indx in range(self.center, self.center + self.trainlength):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    exin = inputrx[indx - self.center:indx + self.center + 1]
                    eyin = inputry[indx - self.center:indx + self.center + 1]
                    rad_aaa = np.real(exin) - 2 * np.sign(np.real(exin)) + 1j * (
                            np.imag(exin) - 2 * np.sign(np.imag(exin)))
                    rad_bbb = np.real(eyin) - 2 * np.sign(np.real(eyin)) + 1j * (
                            np.imag(eyin) - 2 * np.sign(np.imag(eyin)))
                    radx[indx] = np.mean(np.abs(rad_aaa ** 4)) / np.mean(np.abs(rad_aaa ** 4))
                    rady[indx] = np.mean(np.abs(rad_bbb ** 4)) / np.mean(np.abs(rad_bbb ** 4))
                    exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                    eyout_adj2 = eyout[indx] - 2 * np.sign(np.real(eyout[indx])) - 2j * (np.sign(np.imag(eyout[indx])))
                    errx[indx] = ((np.abs(exout_adj2)) ** 2 - radx[indx])
                    erry[indx] = ((np.abs(eyout_adj2)) ** 2 - rady[indx])

                    px = math.exp(-np.abs(errx[indx] - errx[indx - 1]))+math.exp(-indx/gama)
                    py = math.exp(-np.abs(erry[indx] - erry[indx - 1]))+math.exp(-indx/gama)
                    stepsizex = stepsizex * px
                    stepsizey = stepsizey * py
                    if stepsizex > maxstepsize:
                        stepsizex = maxstepsize
                    elif stepsizex < minstepsize:
                        stepsizex = minstepsize
                    if stepsizey > maxstepsize:
                        stepsizey = maxstepsize
                    elif stepsizey < minstepsize:
                        stepsizey = minstepsize
                    hxx = hxx + stepsizex * errx[indx] * exout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy + stepsizex * errx[indx] * exout[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx + stepsizey * erry[indx] * eyout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy + stepsizey * erry[indx] * eyout[indx] * np.conjugate(
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
                        self.stepsize *= 0.5
                        print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def MCMA(self):
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
            cost_x = np.zeros(self.trainlength + self.center, dtype="complex_")
            # cost_xq = np.zeros(self.trainlength + self.center, dtype="complex_")
            cost_y = np.zeros(self.trainlength + self.center, dtype="complex_")
            # cost_yq = np.zeros(self.trainlength + self.center, dtype="complex_")
            radxi = np.zeros(self.trainlength + self.center, dtype="complex_")
            radxq = np.zeros(self.trainlength + self.center, dtype="complex_")
            radyi = np.zeros(self.trainlength + self.center, dtype="complex_")
            radyq = np.zeros(self.trainlength + self.center, dtype="complex_")
            ZR, ZI = 2, 2
            for it in range(self.iterator):
                for indx in range(self.center, self.center + self.trainlength):
                    exi = np.real(inputrx[indx - self.center:indx + self.center + 1])
                    exq = np.imag(inputrx[indx - self.center:indx + self.center + 1])
                    eyi = np.real(inputry[indx - self.center:indx + self.center + 1])
                    eyq = np.imag(inputry[indx - self.center:indx + self.center + 1])
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    rad_xi = exi - 4 * np.sign(exi) - 2 * np.sign(exi - 4 * np.sign(exi))
                    rad_xq = exq - 2 * np.sign(exq) - 2 * np.sign(exq - 4 * np.sign(exq))
                    rad_yi = eyi - 2 * np.sign(eyi) - 2 * np.sign(eyi - 4 * np.sign(eyi))
                    rad_yq = eyq - 2 * np.sign(eyq) - 2 * np.sign(eyq - 4 * np.sign(eyq))
                    disxi = np.min([np.abs(np.real(exout[indx]) - 7.5), np.abs(np.real(exout[indx]) - 2.5), np.abs(np.real(exout[indx]) + 2.5),
                                    np.abs(np.real(exout[indx]) + 7.5)])
                    disxq = np.min([np.abs(np.imag(exout[indx]) - 7.5), np.abs(np.imag(exout[indx]) - 2.5), np.abs(np.imag(exout[indx]) + 2.5),
                                    np.abs(np.imag(exout[indx]) + 7.5)])
                    disyi = np.min([np.abs(np.real(eyout[indx]) - 7.5), np.abs(np.real(eyout[indx]) - 2.5), np.abs(np.real(eyout[indx]) + 2.5),
                                    np.abs(np.real(eyout[indx]) + 7.5)])
                    disyq = np.min([np.abs(np.imag(eyout[indx]) - 7.5), np.abs(np.imag(eyout[indx]) - 2.5), np.abs(np.imag(eyout[indx]) + 2.5),
                                    np.abs(np.imag(eyout[indx]) + 7.5)])
                    if (disxi) < ZR:
                        radxi[indx] = 10
                    else:
                        radxi[indx] = np.mean(np.abs(rad_xi) ** 4) / np.mean(np.abs(rad_xi) ** 2)
                    if (disxq) < ZI:
                        radxq[indx] = 10
                    else:
                        radxq[indx] = np.mean(np.abs(rad_xq) ** 4) / np.mean(np.abs(rad_xq) ** 2)
                    if(disyi) < ZR:
                        radyi[indx] = 10
                    else:
                        radyi[indx] = np.mean(np.abs(rad_yi) ** 4) / np.mean(np.abs(rad_yi) ** 2)
                    if (disyq) < ZI:
                        radyq[indx] = 10
                    else:
                        radyq[indx] = np.mean(np.abs(rad_yq) ** 4) / np.mean(np.abs(rad_yq) ** 2)
                    exout_adj_i = np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])) - 2 * \
                                       np.sign(np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])))
                    exout_adj_q = np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])) - 2 * \
                                       np.sign(np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])))
                    eyout_adj_i = np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])) - 2 * \
                                       np.sign(np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])))
                    eyout_adj_q = np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])) - 2 * \
                                       np.sign(np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])))
                    errx[indx] = np.real(exout[indx]) * (radxi[indx] - np.abs(exout_adj_i) ** 2) + \
                                 1j * np.imag(exout[indx]) * (radxq[indx] - np.abs(exout_adj_q) ** 2)
                    erry[indx] = np.real(eyout[indx]) * (radyi[indx] - np.abs(eyout_adj_i) ** 2) + \
                                 1j * np.imag(eyout[indx]) * (radyq[indx] - np.abs(eyout_adj_q) ** 2)
                    hxx = hxx + self.stepsize * errx[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy + self.stepsize * errx[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx + self.stepsize * erry[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy + self.stepsize * erry[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    cost_x[indx] = exout_adj_i ** 2 - radxi[indx] + 1j * (
                            exout_adj_q ** 2 - radxq[indx])
                    cost_y[indx] = eyout_adj_i ** 2 - radyi[indx] + 1j * (
                            eyout_adj_q ** 2 - radyq[indx])
                self.costfunx[batch][it] = -1 * (np.mean(np.real(cost_x[self.center:])) + np.mean(np.imag(cost_x[self.center:])))
                self.costfuny[batch][it] = -1 * (np.mean(np.real(cost_y[self.center:])) + np.mean(np.imag(cost_y[self.center:])))
                print(self.costfunx[batch][it])
                if it > 1:
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
                            np.abs(self.costfunx[batch][it]) and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
                            np.abs(self.costfuny[batch][it]):
                        print("Earlybreak at iterator {}".format(it))
                        break
                if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                        np.abs(self.costfunx[batch][it]) and np.abs(
                    self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
                        np.abs(self.costfuny[batch][it]):
                    self.stepsize *= 0.5
                    print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def run_16qam_3(self):
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

                    inxi = inxi - 4 * np.sign(inxi) - 2 * np.sign(inxi - 4 * np.sign(inxi))
                    inxq = inxq - 4 * np.sign(inxq) - 2 * np.sign(inxq - 4 * np.sign(inxq))
                    inyi = inyi - 4 * np.sign(inyi) - 2 * np.sign(inyi - 4 * np.sign(inyi))
                    inyq = inyq - 4 * np.sign(inyq) - 2 * np.sign(inyq - 4 * np.sign(inyq))

                    squ_Rxi[indx] = np.mean(abs(inxi) ** 4) / np.mean(abs(inxi) ** 2)
                    squ_Rxq[indx] = np.mean(abs(inxq) ** 4) / np.mean(abs(inxq) ** 2)
                    squ_Ryi[indx] = np.mean(abs(inyi) ** 4) / np.mean(abs(inyi) ** 2)
                    squ_Ryq[indx] = np.mean(abs(inyq) ** 4) / np.mean(abs(inyq) ** 2)

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
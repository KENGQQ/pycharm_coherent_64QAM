import numpy as np
import cmath
from itertools import product
from itertools import combinations_with_replacement
from tqdm import trange
class Equalizer:
    def __init__(self,  txsig, rxsig, order, taps=[0, 0, 0], trainoverhead=0.2):
        self.rxsig = np.array(rxsig)
        self.txsig = np.array(txsig)
        self.volterraorder = order
        self.taps = taps
        self.taps_first = taps[0]
        self.taps_second = taps[1]
        self.taps_third = taps[2]
        self.datalength = len(rxsig)
        self.overhead = trainoverhead
        self.trainlength = int(self.datalength * trainoverhead)
        self.testlength = self.datalength - self.trainlength


    def complexvolterra(self):
        featuremat = []
        featuretest = []
        trainrx = self.rxsig[:self.trainlength]
        traintx = self.txsig[:self.trainlength]
        testrx = self.rxsig[self.trainlength:]
        testtx = self.txsig[self.trainlength:]
        tapscen = int((np.max(self.taps) - 1) / 2)
        taps1cen = int((self.taps_first - 1) / 2)
        taps2cen = int((self.taps_second - 1) / 2)
        taps3cen = int((self.taps_third - 1) / 2)
        if self.volterraorder == 1:
        #1st order
            for indx in range(taps1cen, self.trainlength-taps1cen+1):
                x = trainrx[indx-taps1cen:indx+taps1cen+1].tolist()
                featuremat.append(([1] + x))
            for indx in range(taps1cen, self.testlength-taps1cen+1):
                x = testrx[indx-taps1cen:indx+taps1cen+1].tolist()
                featuretest.append(([1] + x))
        if self.volterraorder == 2:
            for indx in range(tapscen, self.trainlength-tapscen+1):
                x1 = trainrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x2 = trainrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                x2_ = np.reshape(x2, (1, len(x2)))
                od2indx = np.triu_indices(self.taps_second)
                A = np.matmul(np.transpose(x2_), x2_)[od2indx].tolist()
                B = np.matmul(np.transpose(x2_), np.conjugate(x2_))[od2indx].tolist()
                C = np.matmul(np.conjugate(np.transpose(x2_)), np.conjugate(x2_))[od2indx].tolist()
                featuremat.append(([1] + x1 + A + B + C))
            for indx in range(tapscen, self.testlength-tapscen+1):
                x1 = testrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x2 = testrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                x2_ = np.reshape(x2, (1, len(x2)))
                od2indx = np.triu_indices(self.taps_second)
                A = np.matmul(np.transpose(x2_), x2_)[od2indx].tolist()
                B = np.matmul(np.transpose(x2_), np.conjugate(x2_))[od2indx].tolist()
                C = np.matmul(np.conjugate(np.transpose(x2_)), np.conjugate(x2_))[od2indx].tolist()
                featuretest.append(([1] + x1 + A + B + C))
        if self.volterraorder == 3:
            for indx in trange(tapscen, self.trainlength-tapscen):
                x1 = trainrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x1_conj = np.conjugate(x1).tolist()
                x2 = trainrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                x2_conj = np.conjugate(x2).tolist()
                A_2, B_2, C_2 = [], [], []
                iter = list(combinations_with_replacement(range(self.taps_second), 2))
                for it in iter:
                    A_2.append(x2[it[0]] * x2[it[1]])
                    B_2.append(x2[it[0]] * x2_conj[it[1]])
                    C_2.append((x2_conj[it[0]] * x2_conj[it[1]]))
                x3 = trainrx[indx - taps3cen:indx + taps3cen + 1].tolist()
                x3_conj = np.conjugate(x3).tolist()
                A_3, B_3, C_3, D_3 = [], [], [], []
                iter = list(combinations_with_replacement(range(self.taps_third), 3))
                for it in iter:
                    A_3.append(x3[it[0]] * x3[it[1]] * x3[it[2]])
                    B_3.append(x3[it[0]] * x3[it[1]] * x3_conj[it[2]])
                    C_3.append(x3[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                    D_3.append(x3_conj[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                featuremat.append([1] + x1 + x1_conj + A_2 + B_2 + C_2 + A_3 + B_3 + C_3 + D_3)

            for indx in trange(tapscen, self.testlength - tapscen):
                x1 = testrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x1_conj = np.conjugate(x1).tolist()
                x2 = testrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                x2_conj = np.conjugate(x2).tolist()
                A_2, B_2, C_2 = [], [], []
                iter = list(combinations_with_replacement(range(self.taps_second), 2))
                for it in iter:
                    A_2.append(x2[it[0]] * x2[it[1]])
                    B_2.append(x2[it[0]] * x2_conj[it[1]])
                    C_2.append((x2_conj[it[0]] * x2_conj[it[1]]))
                x3 = testrx[indx - taps3cen:indx + taps3cen + 1].tolist()
                x3_conj = np.conjugate(x3).tolist()
                A_3, B_3, C_3, D_3 = [], [], [], []
                iter = list(combinations_with_replacement(range(self.taps_third), 3))
                for it in iter:
                    A_3.append(x3[it[0]] * x3[it[1]] * x3[it[2]])
                    B_3.append(x3[it[0]] * x3[it[1]] * x3_conj[it[2]])
                    C_3.append(x3[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                    D_3.append(x3_conj[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                featuretest.append([1] + x1 + x1_conj + A_2 + B_2 + C_2 + A_3 + B_3 + C_3 + D_3)
        traintx = traintx[tapscen:-1 - tapscen + 1]
        testtx = testtx[tapscen:-1 - tapscen + 1]
        trainarr = np.array(featuremat)
        testarr = np.array(featuretest)
        wiener = np.matmul(np.linalg.inv(np.matmul(np.conjugate(np.transpose(trainarr)), trainarr)),
                           np.matmul(np.conjugate(np.transpose(trainarr)), traintx))
        rx_predict = np.matmul(testarr, wiener)
        return (testtx, rx_predict)

    def complexvolterra_reduceterm(self):
        featuremat = []
        featuretest = []
        trainrx = self.rxsig[:self.trainlength]
        traintx = self.txsig[:self.trainlength]
        testrx = self.rxsig[self.trainlength:]
        testtx = self.txsig[self.trainlength:]
        tapscen = int((np.max(self.taps) - 1) / 2)
        taps1cen = int((self.taps_first - 1) / 2)
        taps2cen = int((self.taps_second - 1) / 2)
        taps3cen = int((self.taps_third - 1) / 2)
        if self.volterraorder == 1:
        #1st order
            for indx in range(taps1cen, self.trainlength-taps1cen+1):
                x = trainrx[indx-taps1cen:indx+taps1cen+1].tolist()
                featuremat.append(([1] + x))
            for indx in range(taps1cen, self.testlength-taps1cen+1):
                x = testrx[indx-taps1cen:indx+taps1cen+1].tolist()
                featuretest.append(([1] + x))
        if self.volterraorder == 2:
            for indx in range(tapscen, self.trainlength-tapscen+1):
                x1 = trainrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x2 = trainrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                x2_ = np.reshape(x2, (1, len(x2)))
                od2indx = np.triu_indices(self.taps_second)
                A = np.matmul(np.transpose(x2_), x2_)[od2indx].tolist()
                B = np.matmul(np.transpose(x2_), np.conjugate(x2_))[od2indx].tolist()
                C = np.matmul(np.conjugate(np.transpose(x2_)), np.conjugate(x2_))[od2indx].tolist()
                featuremat.append(([1] + x1 + A + B + C))
            for indx in range(tapscen, self.testlength-tapscen+1):
                x1 = testrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x2 = testrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                x2_ = np.reshape(x2, (1, len(x2)))
                od2indx = np.triu_indices(self.taps_second)
                A = np.matmul(np.transpose(x2_), x2_)[od2indx].tolist()
                B = np.matmul(np.transpose(x2_), np.conjugate(x2_))[od2indx].tolist()
                C = np.matmul(np.conjugate(np.transpose(x2_)), np.conjugate(x2_))[od2indx].tolist()
                featuretest.append(([1] + x1 + A + B + C))
        if self.volterraorder == 3:
            for indx in range(tapscen, self.trainlength-tapscen):
                x1 = trainrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x1_conj = np.conjugate(x1).tolist()
                x3 = trainrx[indx - taps3cen:indx + taps3cen + 1].tolist()
                x3_conj = np.conjugate(x3).tolist()
                A_3, B_3, C_3, D_3 = [], [], [], []
                iter = list(combinations_with_replacement(range(self.taps_third), 3))
                for it in iter:
                    A_3.append(x3[it[0]] * x3[it[1]] * x3[it[2]])
                    B_3.append(x3[it[0]] * x3[it[1]] * x3_conj[it[2]])
                    # C_3.append(x3[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                    # D_3.append(x3_conj[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                featuremat.append([1] + x1 + x1_conj + A_3 + B_3)
            for indx in range(tapscen, self.testlength - tapscen):
                x1 = testrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x1_conj = np.conjugate(x1).tolist()
                x3 = testrx[indx - taps3cen:indx + taps3cen + 1].tolist()
                x3_conj = np.conjugate(x3).tolist()
                A_3, B_3, C_3, D_3 = [], [], [], []
                iter = list(combinations_with_replacement(range(self.taps_third), 3))
                for it in iter:
                    A_3.append(x3[it[0]] * x3[it[1]] * x3[it[2]])
                    B_3.append(x3[it[0]] * x3[it[1]] * x3_conj[it[2]])
                    # C_3.append(x3[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                    # D_3.append(x3_conj[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                featuretest.append([1] + x1 + x1_conj + A_3 + B_3)
        traintx = traintx[tapscen:-1 - tapscen + 1]
        testtx = testtx[tapscen:-1 - tapscen + 1]
        trainarr = np.array(featuremat)
        testarr = np.array(featuretest)
        wiener = np.matmul(np.linalg.inv(np.matmul(np.conjugate(np.transpose(trainarr)), trainarr)),
                           np.matmul(np.conjugate(np.transpose(trainarr)), traintx))
        rx_predict = np.matmul(testarr, wiener)
        return (testtx, rx_predict)

    def complexvolterra_MMSE(self):
        featuremat = []
        featuretest = []
        trainrx = self.rxsig[:self.trainlength]
        traintx = self.txsig[:self.trainlength]
        testrx = self.rxsig[self.trainlength:]
        testtx = self.txsig[self.trainlength:]
        tapscen = int((np.max(self.taps) - 1) / 2)
        taps1cen = int((self.taps_first - 1) / 2)
        taps2cen = int((self.taps_second - 1) / 2)
        taps3cen = int((self.taps_third - 1) / 2)
        if self.volterraorder == 1:
        #1st order
            for indx in range(taps1cen, self.trainlength-taps1cen+1):
                x = trainrx[indx-taps1cen:indx+taps1cen+1].tolist()
                featuremat.append(([1] + x))
            for indx in range(taps1cen, self.testlength-taps1cen+1):
                x = testrx[indx-taps1cen:indx+taps1cen+1].tolist()
                featuretest.append(([1] + x))
        if self.volterraorder == 2:
            for indx in range(tapscen, self.trainlength-tapscen+1):
                x1 = trainrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x2 = trainrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                x2_ = np.reshape(x2, (1, len(x2)))
                od2indx = np.triu_indices(self.taps_second)
                A = np.matmul(np.transpose(x2_), x2_)[od2indx].tolist()
                B = np.matmul(np.transpose(x2_), np.conjugate(x2_))[od2indx].tolist()
                C = np.matmul(np.conjugate(np.transpose(x2_)), np.conjugate(x2_))[od2indx].tolist()
                featuremat.append(([1] + x1 + A + B + C))
            for indx in range(tapscen, self.testlength-tapscen+1):
                x1 = testrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x2 = testrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                x2_ = np.reshape(x2, (1, len(x2)))
                od2indx = np.triu_indices(self.taps_second)
                A = np.matmul(np.transpose(x2_), x2_)[od2indx].tolist()
                B = np.matmul(np.transpose(x2_), np.conjugate(x2_))[od2indx].tolist()
                C = np.matmul(np.conjugate(np.transpose(x2_)), np.conjugate(x2_))[od2indx].tolist()
                featuretest.append(([1] + x1 + A + B + C))
        if self.volterraorder == 3:
            MSE = np.zeros((1000,))
            step_size = 1e-5
            weight_length = 1 + 2*len(list(
                range(self.taps_first))) + 3 * len(
                list(combinations_with_replacement(range(self.taps_second), 2))) + 4 * len(list(
                combinations_with_replacement(range(self.taps_third), 3)))
            weight_arr = np.ones((weight_length, ), dtype='complex_')
            err = np.zeros((self.trainlength, 1), dtype='complex_')
            gradient = np.zeros((self.trainlength, weight_length), dtype='complex_')
            rx_train = np.zeros((self.trainlength, ), dtype='complex_')
            rx_predict = np.zeros((self.testlength, ), dtype='complex_')
            for epoch in range(1000):
                for indx in range(tapscen, self.trainlength-tapscen):
                    x1 = trainrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                    x1_conj = np.conjugate(x1).tolist()
                    x2 = trainrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                    x2_conj = np.conjugate(x2).tolist()
                    A_2, B_2, C_2 = [], [], []
                    iter = list(combinations_with_replacement(range(self.taps_second), 2))
                    for it in iter:
                        A_2.append(x2[it[0]] * x2[it[1]])
                        B_2.append(x2[it[0]] * x2_conj[it[1]])
                        C_2.append((x2_conj[it[0]] * x2_conj[it[1]]))
                    x3 = trainrx[indx - taps3cen:indx + taps3cen + 1].tolist()
                    x3_conj = np.conjugate(x3).tolist()
                    A_3, B_3, C_3, D_3 = [], [], [], []
                    iter = list(combinations_with_replacement(range(self.taps_third), 3))
                    for it in iter:
                        A_3.append(x3[it[0]] * x3[it[1]] * x3[it[2]])
                        B_3.append(x3[it[0]] * x3[it[1]] * x3_conj[it[2]])
                        C_3.append(x3[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                        D_3.append(x3_conj[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                    trainarr = np.array([1] + x1 + x1_conj + A_2 + B_2 + C_2 + A_3 + B_3 + C_3 + D_3)
                    rx_train[indx] = np.matmul(weight_arr, trainarr)
                    err[indx] = traintx[indx] - rx_train[indx]
                    gradient[indx] = - 2 * step_size * err[indx] * np.conjugate(trainarr)
                    weight_arr = weight_arr - gradient[indx]

                MSE[epoch] = np.mean(np.abs(traintx - rx_train)**2)
                if epoch > 1:
                    if np.abs(MSE[epoch] -MSE[epoch-1]) < MSE[epoch] * 0.001:
                        print('early break')
                        break
                    elif np.abs(MSE[epoch] -MSE[epoch-1]) < MSE[epoch] * 0.005:
                        print('step size adjust')
                        step_size *= 0.5
                print(MSE[epoch])

            for indx in range(tapscen, self.testlength - tapscen):
                x1 = testrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x1_conj = np.conjugate(x1).tolist()
                x2 = testrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                x2_conj = np.conjugate(x2).tolist()
                A_2, B_2, C_2 = [], [], []
                iter = list(combinations_with_replacement(range(self.taps_second), 2))
                for it in iter:
                    A_2.append(x2[it[0]] * x2[it[1]])
                    B_2.append(x2[it[0]] * x2_conj[it[1]])
                    C_2.append((x2_conj[it[0]] * x2_conj[it[1]]))
                x3 = testrx[indx - taps3cen:indx + taps3cen + 1].tolist()
                x3_conj = np.conjugate(x3).tolist()
                A_3, B_3, C_3, D_3 = [], [], [], []
                iter = list(combinations_with_replacement(range(self.taps_third), 3))
                for it in iter:
                    A_3.append(x3[it[0]] * x3[it[1]] * x3[it[2]])
                    B_3.append(x3[it[0]] * x3[it[1]] * x3_conj[it[2]])
                    C_3.append(x3[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                    D_3.append(x3_conj[it[0]] * x3_conj[it[1]] * x3_conj[it[2]])
                testarr = np.array([1] + x1 + x1_conj + A_2 + B_2 + C_2 + A_3 + B_3 + C_3 + D_3)
                rx_predict[indx] = np.matmul(weight_arr, testarr)
        traintx = traintx[tapscen:-1 - tapscen + 1]
        testtx = testtx[tapscen:-1 - tapscen + 1]
        rx_predict = rx_predict[tapscen:-1 - tapscen + 1]
        return (testtx, rx_predict)


    def realvolterra(self):
        featuremat = []
        featuretest = []
        trainrx = self.rxsig[:self.trainlength]
        traintx = self.txsig[:self.trainlength]
        testrx = self.rxsig[self.trainlength:]
        testtx = self.txsig[self.trainlength:]
        tapscen = int((np.max(self.taps) - 1) / 2)
        taps1cen = int((self.taps_first - 1) / 2)
        taps2cen = int((self.taps_second - 1) / 2)
        taps3cen = int((self.taps_third - 1) / 2)
        if self.volterraorder == 1:
        #1st order
            for indx in range(taps1cen, self.trainlength-taps1cen+1):
                x = trainrx[indx-taps1cen:indx+taps1cen+1].tolist()
                featuremat.append(([1] + x))
            for indx in range(taps1cen, self.testlength-taps1cen+1):
                x = testrx[indx-taps1cen:indx+taps1cen+1].tolist()
                featuretest.append(([1] + x))
        if self.volterraorder == 2:
            for indx in range(tapscen, self.trainlength-tapscen+1):
                x1 = trainrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x2 = trainrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                x2_ = np.reshape(x2, (1, len(x2)))
                od2indx = np.triu_indices(self.taps_second)
                A = np.matmul(np.transpose(x2_), x2_)[od2indx].tolist()
                B = np.matmul(np.transpose(x2_), np.conjugate(x2_))[od2indx].tolist()
                C = np.matmul(np.conjugate(np.transpose(x2_)), np.conjugate(x2_))[od2indx].tolist()
                featuremat.append(([1] + x1 + A + B + C))
            for indx in range(tapscen, self.testlength-tapscen+1):
                x1 = testrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x2 = testrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                x2_ = np.reshape(x2, (1, len(x2)))
                od2indx = np.triu_indices(self.taps_second)
                A = np.matmul(np.transpose(x2_), x2_)[od2indx].tolist()
                B = np.matmul(np.transpose(x2_), np.conjugate(x2_))[od2indx].tolist()
                C = np.matmul(np.conjugate(np.transpose(x2_)), np.conjugate(x2_))[od2indx].tolist()
                featuretest.append(([1] + x1 + A + B + C))
        if self.volterraorder == 3:
            for indx in trange(tapscen, self.trainlength-tapscen):
                x1 = trainrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x2 = trainrx[indx - taps2cen:indx + taps2cen + 1].tolist()
                A_2 = []
                iter = list(combinations_with_replacement(range(self.taps_second), 2))
                for it in iter:
                    A_2.append(x2[it[0]] * x2[it[1]])
                x3 = trainrx[indx - taps3cen:indx + taps3cen + 1].tolist()
                A_3 = []
                iter = list(combinations_with_replacement(range(self.taps_third), 3))
                for it in iter:
                    A_3.append(x3[it[0]] * x3[it[1]] * x3[it[2]])
                featuremat.append([1] + x1 + A_2 + A_3)

            for indx in trange(tapscen, self.testlength - tapscen):
                x1 = testrx[indx - taps1cen:indx + taps1cen + 1].tolist()
                x2 = testrx[indx - taps2cen:indx + taps2cen + 1].tolist()

                A_2 = []
                iter = list(combinations_with_replacement(range(self.taps_second), 2))
                for it in iter:
                    A_2.append(x2[it[0]] * x2[it[1]])
                x3 = testrx[indx - taps3cen:indx + taps3cen + 1].tolist()
                A_3 = []
                iter = list(combinations_with_replacement(range(self.taps_third), 3))
                for it in iter:
                    A_3.append(x3[it[0]] * x3[it[1]] * x3[it[2]])
                featuretest.append([1] + x1 + A_2 + A_3)
        traintx = traintx[tapscen:-1 - tapscen + 1]
        testtx = testtx[tapscen:-1 - tapscen + 1]
        trainarr = np.array(featuremat)
        testarr = np.array(featuretest)
        wiener = np.matmul(np.linalg.inv(np.matmul(np.transpose(trainarr), trainarr)),
                           np.matmul(np.transpose(trainarr), traintx))
        rx_predict = np.matmul(testarr, wiener)

        return (testtx, rx_predict)
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:02:30 2020

@author: KENG
"""
import numpy as np
import time


class KENG_corr:
    def __init__(self, window_length):
        self.window_length = window_length
        self.start = 0
        self.end = 10000

    def calculate(self, Rx, Tx):
        if np.size(Rx) != np.size(Tx):
            print('Length is not euqal')

        else:
            Rx = np.reshape(np.mat(Rx), [np.size(Rx), 1])
            Tx = np.reshape(np.mat(Tx), [np.size(Tx), 1])
            if self.window_length > np.size(Tx):
                print('Tx is not enough')
            if self.end > np.size(Rx):
                print('Rx is not enough (reduce the endpoint)')
            Tx_correlation_window_0 = Tx[0:self.window_length]

            positive_cor_record_0 = []
            positive_cor_record_0_abs = []

            for indx in range(self.start, self.end - self.window_length):
                c = np.corrcoef(Rx[indx:indx + self.window_length, 0].T, Tx_correlation_window_0.T)[0][1]
                c_abs = abs(c)
                positive_cor_record_0.append(c)
                positive_cor_record_0_abs.append(c_abs)

            positive_max_corr_0 = np.max(positive_cor_record_0_abs)
            positive_shift_0 = np.argmax(positive_cor_record_0_abs)

            # negative_cor_record = []
            # for indx in range(self.start, self.end-self.window_length):
            #     c = np.corrcoef(-Rx[self.indx:indx+self.window_length,0],Tx_correlation_window)[0][1]
            #     negative_cor_record.append(c)
            # negative_max_corr = np.max(negative_cor_record)
            # negative_shift = np.argmax(negative_cor_record)
            print('Shift length = {}'.format(positive_shift_0))
            print('Positive corr = {} ,Positive corr abs = {}'.format(positive_cor_record_0[positive_shift_0],
                                                                      positive_max_corr_0))

            Rx_corr = Rx[positive_shift_0:]
            Tx_corr = Tx[:Rx_corr.size]

            return Rx_corr, Tx_corr

    def calculate_Rx(self, Rx, Tx):
        if np.size(Rx) != np.size(Tx):
            print('Length is not euqal')

        else:
            Rx = np.reshape(np.mat(Rx), [np.size(Rx), 1])
            Tx = np.reshape(np.mat(Tx), [np.size(Tx), 1])
            if self.window_length > np.size(Tx):
                print('Tx is not enough')
            if self.end > np.size(Rx):
                print('Rx is not enough (reduce the endpoint)')

            Rx_correlation_window_0 = Rx[0:self.window_length]
            positive_cor_record_0 = []
            positive_cor_record_0_abs = []

            for indx in range(self.start, self.end - self.window_length):
                c = np.corrcoef(Tx[indx:indx + self.window_length, 0].T, Rx_correlation_window_0.T)[0][1]
                c_abs = abs(c)
                positive_cor_record_0.append(c)
                positive_cor_record_0_abs.append(c_abs)

            positive_max_corr_0 = np.max(positive_cor_record_0_abs)
            positive_shift_0 = np.argmax(positive_cor_record_0_abs)

            print('Shift length = {}'.format(positive_shift_0))
            print('Positive corr = {} ,Positive corr abs = {}'.format(positive_cor_record_0[positive_shift_0],
                                                                      positive_max_corr_0))

            Tx_corr = Tx[positive_shift_0:]
            Rx_corr = Rx[:Tx_corr.size]

            return Rx_corr, Tx_corr

    def corr(self, Tx, Rx, Prbs):
        Rx = np.array(Rx)
        Tx = np.array(Tx)
        if Prbs == 15:
            # print("Tx: PRBS15 data")
            assert len(Tx) >= 2 ** 15 - 1, "TX length not enough"
            Tx = Tx[:2 ** 15 - 1]
            Tx_ = np.tile(Tx, 10)
        if Prbs == 13:
            # print("Tx :PRBS13 data")
            assert len(Tx) >= 2 ** 13 - 1, "TX length not enough"
            Tx = Tx[0:2 ** 13 - 1]
            Tx_ = np.tile(Tx, 15)
        output_length = 200000
        window = self.window_length
        start = 0
        end = 8192
        Tx_cor = Tx_[:window]
        if end > Rx.size:
            end = Rx.size
        if window > Rx.size:
            window = Rx.size

        # positive_cor_record = []
        # for indx in range(start, end - window + 1):
        #     c = np.corrcoef(Rx[indx:indx + window], Tx_cor)[0][1]
        #     positive_cor_record.append(c)
        # positive_max_corr = np.max(positive_cor_record)
        # positive_shift = np.argmax(positive_cor_record)

        positive_cor_record = []
        for indx in range(start, end):
            c = np.corrcoef(Rx[indx:indx + window], Tx_cor)[0][1]
            positive_cor_record.append(c)
        positive_max_corr = np.max(positive_cor_record)
        positive_max_corr = round(positive_max_corr, 4)
        positive_shift = np.argmax(positive_cor_record)

        # negative_cor_record = []
        # for indx in range(start, end - window + 1):
        #     c = np.corrcoef(-Rx[indx:indx + window], Tx_cor)[0][1]
        #     negative_cor_record.append(c)
        # negative_max_corr = np.max(negative_cor_record)
        # negative_shift = np.argmax(negative_cor_record)

        negative_cor_record = []
        for indx in range(start, end):
            c = np.corrcoef(-Rx[indx:indx + window], Tx_cor)[0][1]
            negative_cor_record.append(c)
        negative_max_corr = np.max(negative_cor_record)
        negative_max_corr = round(negative_max_corr, 4)
        negative_shift = np.argmax(negative_cor_record)

        print('Positive corr:{}'.format(positive_max_corr))
        print('Shift length:%d' % (positive_shift))
        print('Negative corr:{}'.format(negative_max_corr))
        print('Shift length:%d' % (negative_shift))
        if positive_max_corr >= negative_max_corr:
            print("Calculate Rx")
            if positive_shift + output_length > Rx.size:
                R_output = Rx[positive_shift:]
            else:
                R_output = Rx[positive_shift:positive_shift + output_length]
            T_output = Tx_[:R_output.size]
            self.shift = positive_shift
            self.corr = positive_max_corr
            return T_output, R_output, positive_max_corr
        else:
            print("Calculate -Rx")
            if negative_shift + output_length > Rx.size:
                R_output = -Rx[negative_shift:]
            else:
                R_output = -Rx[negative_shift:negative_shift + output_length]
            T_output = Tx_[:R_output.size]
            self.shift = negative_shift
            self.corr = negative_max_corr
            return T_output, R_output, negative_max_corr


    def corr_ex(self, Tx, Rx, Prbs):         #input should be complex
        Rx = np.array(Rx)
        Tx = np.array(Tx)
        if Prbs == 15:
            # print("Tx: PRBS15 data")
            assert len(Tx) >= 2 ** 15 - 1, "TX length not enough"
            Tx = Tx[:2 ** 15 - 1]
            Tx_ = np.tile(Tx, 10)
        if Prbs == 13:
            # print("Tx :PRBS13 data")
            assert len(Tx) >= 2 ** 13 - 1, "TX length not enough"
            Tx = Tx[0:2 ** 13 - 1]
            Tx_ = np.tile(Tx, 15)
        output_length = 200000
        window = self.window_length
        start = 0
        end = 8192
        Tx_cor = Tx_[:window]
        if end > Rx.size:
            end = Rx.size
        if window > Rx.size:
            window = Rx.size

        RxI = np.real(Rx)
        RxQ = np.imag(Rx)

        cal = ['RxI', '-RxI', 'RxQ', '-RxQ']

        positive_cor_record_I = []
        for indx in range(start, end):
            c = np.corrcoef(RxI[indx:indx + window], Tx_cor)[0][1]
            positive_cor_record_I.append(c)
        positive_max_corr_I = np.max(positive_cor_record_I)
        positive_max_corr_I = round(positive_max_corr_I, 4)
        positive_shift_I = np.argmax(positive_cor_record_I)

        negative_cor_record_I = []
        for indx in range(start, end):
            c = np.corrcoef(-RxI[indx:indx + window], Tx_cor)[0][1]
            negative_cor_record_I.append(c)
        negative_max_corr_I = np.max(negative_cor_record_I)
        negative_max_corr_I = round(negative_max_corr_I, 4)
        negative_shift_I = np.argmax(negative_cor_record_I)

        positive_cor_record_Q = []
        for indx in range(start, end):
            c = np.corrcoef(RxQ[indx:indx + window], Tx_cor)[0][1]
            positive_cor_record_Q.append(c)
        positive_max_corr_Q = np.max(positive_cor_record_Q)
        positive_max_corr_Q = round(positive_max_corr_Q, 4)
        positive_shift_Q = np.argmax(positive_cor_record_Q)

        negative_cor_record_Q = []
        for indx in range(start, end):
            c = np.corrcoef(-RxQ[indx:indx + window], Tx_cor)[0][1]
            negative_cor_record_Q.append(c)
        negative_max_corr_Q = np.max(negative_cor_record_Q)
        negative_max_corr_Q = round(negative_max_corr_Q, 4)
        negative_shift_Q = np.argmax(negative_cor_record_Q)

        corr_value = [positive_max_corr_I, negative_max_corr_I, positive_max_corr_Q, negative_max_corr_Q]
        corr_shift = [positive_shift_I, negative_shift_I, positive_shift_Q, negative_shift_Q]

        max_corr = max(corr_value)
        corr_shift = corr_shift[np.argmax(corr_value)]
        print('corr value: {}'.format(max_corr))
        print('Shift length : {}'.format(corr_shift))

        if (max_corr == positive_max_corr_I):
            if corr_shift + output_length > Rx.size:
                R_output = RxI[corr_shift:]
            else:
                R_output = RxI[corr_shift:corr_shift + output_length]

        elif (max_corr == negative_max_corr_I):
            if corr_shift + output_length > Rx.size:
                R_output = -RxI[corr_shift:]
            else:
                R_output = -RxI[corr_shift:corr_shift + output_length]

        elif (max_corr == positive_max_corr_Q):
            if corr_shift + output_length > Rx.size:
                R_output = RxQ[corr_shift:]
            else:
                R_output = RxQ[corr_shift:corr_shift + output_length]

        elif (max_corr == negative_max_corr_Q):
            if corr_shift + output_length > Rx.size:
                R_output = -RxQ[corr_shift:]
            else:
                R_output = -RxQ[corr_shift:corr_shift + output_length]

        T_output = Tx_[:R_output.size]
        self.shift = corr_shift
        self.corr = max_corr
        self.cal_vec = cal[np.argmax(corr_value)]
        print('calculate {}'.format(self.cal_vec))
        return T_output, R_output, max_corr

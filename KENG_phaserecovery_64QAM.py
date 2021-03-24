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

    def QAM_64QAM_1(self, Rx):
        c1_radius_i = 0
        c1_radius_o = 1.55
        c3_radius_i = 3.2
        c3_radius_o = 4
        c9_radius_i = 8.2
        c9_radius_o = 10.2

        Rx_amplitude = np.abs(Rx)
        Rx_zeropad = np.zeros(len(Rx), dtype = "complex_")

        for i in range(len(Rx_zeropad)):
            if (c1_radius_i <= Rx_amplitude[i]) and (Rx_amplitude[i] < c1_radius_o):
                Rx_zeropad[i] = Rx[i]
            if (c3_radius_i <= Rx_amplitude[i]) and (Rx_amplitude[i] < c3_radius_o):
                Rx_zeropad[i] = Rx[i]
            if (c9_radius_i <= Rx_amplitude[i]) and (Rx_amplitude[i] < c9_radius_o):
                Rx_zeropad[i] = Rx[i]

        a = KENG_phaserecovery()
        Rx_vv = a.forth_QPSK(Rx_zeropad, tap = 61)
        phase1 = a.phase_adj

        Rx_ph = np.zeros(int(np.size(phase1)), dtype='complex_')
        for i in range(np.size(Rx_ph)):
            Rx_ph[i] = Rx[i] * cmath.exp(-1j * phase1[0, i])

        return Rx_ph

    def Rotation_algorithm(self, Rx):    #Carrier Phase Estimation Through the Rotation Algorithm for 64-QAM Optical Systems

        Rx_RA = Rx ** 4
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
        c4_radius_o = 930
        # c5_radius_i = 0
        c5_radius_o = 1950
        # c6_radius_i = 0
        c6_radius_o = 3000
        # c7_radius_i = 0
        c7_radius_o = 4550
        # c8_radius_i = 0
        c8_radius_o = 7000
        # c9_radius_i = 0
        c9_radius_o = 100000
        # c10_radius_i = 0
        # c10_radius_o = 65
        radius_o = [c0_radius_o, c1_radius_o, c2_radius_o, c3_radius_o, c4_radius_o, c5_radius_o, c6_radius_o, c7_radius_o, c8_radius_o, c9_radius_o]
        # radius_o = [ c1_radius_o, c2_radius_o, c3_radius_o, c4_radius_o, c5_radius_o, c6_radius_o, c7_radius_o, c8_radius_o, c9_radius_o]

        # aaa = [i for i in range(len(Rx_RA))]

        for i in range(len(Rx_RA)):
            for j in range(1, 6):
                if radius_o[j - 1] <= abs(Rx_RA[i]) < radius_o[j]:
                    Rx_RA[i] = Rx_RA[i] * np.exp(4j * c_theta[j - 1] * np.sign(np.imag(Rx_RA[i])))
                    # aaa.remove(i)
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
        Rx_vv = a.first_QPSK(Rx_RA, tap = 61)
        phase1 = a.phase_adj

        Rx_ph = np.zeros(int(np.size(phase1)), dtype='complex_')
        for i in range(np.size(Rx_ph)):
            Rx_ph[i] = Rx[i] * cmath.exp(-1j * phase1[0, i])

        return Rx_ph
    def QAM_3(self, Rx, c1_radius,c2_radius):  # A New Algorithm for 16QAM Carrier Phase Estimation Using QPSK Partitioning
        c1_radius = c1_radius
        c2_radius = c2_radius

        for iteration in range(0, 1):
            Rx = np.reshape(np.matrix(Rx), [np.size(Rx), 1])
            Rx_amplitude = abs(Rx)
            Rx_c1, Rx_c2, Rx_c3 = [], [], []

            for i in range(0, np.size(Rx)):
                if Rx_amplitude[i, 0] <= c1_radius:
                    Rx_c1.append(Rx[i, 0])
                    Rx_c2.append(0)
                    Rx_c3.append(0)

                if Rx_amplitude[i, 0] > c1_radius and Rx_amplitude[i, 0] <= c2_radius:
                    Rx_c1.append(0)
                    Rx_c2.append(Rx[i, 0])
                    Rx_c3.append(0)

                if Rx_amplitude[i, 0] > c2_radius:
                    Rx_c1.append(0)
                    Rx_c2.append(0)
                    Rx_c3.append(Rx[i, 0])

            Rx_c1 = np.array(Rx_c1)
            Rx_c2 = np.array(Rx_c2)
            Rx_c3 = np.array(Rx_c3)

            tap = 9
            a = KENG_phaserecovery()
            c1c3_vv = a.forth_QPSK(Rx_c1 + Rx_c3, tap=tap)
            phase1 = a.phase_adj

            for k in range(0, np.size(Rx_c2)):
                angle = cmath.phase(Rx_c2[k])

                if angle > 0:

                    if angle < cmath.pi / 4:
                        Rx_c2[k] = Rx_c2[k] * cmath.exp(1j * (cmath.pi / 4 - angle))
                    if angle <= cmath.pi / 2 and angle > cmath.pi / 4:
                        Rx_c2[k] = Rx_c2[k] * cmath.exp(-1j * (angle - cmath.pi / 4))
                    if angle < 3 * cmath.pi / 4 and angle > cmath.pi / 2:
                        Rx_c2[k] = Rx_c2[k] * cmath.exp(1j * (3 * cmath.pi / 4 - angle))
                    if angle < cmath.pi and angle > 3 * cmath.pi / 4:
                        Rx_c2[k] = Rx_c2[k] * cmath.exp(-1j * (angle - 3 * cmath.pi / 4))

                if angle < 0:

                    if angle > -cmath.pi / 4:
                        Rx_c2[k] = Rx_c2[k] * cmath.exp(-1j * -(-cmath.pi / 4 - angle))
                    if angle > -cmath.pi / 2 and angle < -cmath.pi / 4:
                        Rx_c2[k] = Rx_c2[k] * cmath.exp(1j * -(angle - -cmath.pi / 4))
                    if angle > -3 * cmath.pi / 4 and angle < -cmath.pi / 2:
                        Rx_c2[k] = Rx_c2[k] * cmath.exp(-1j * -(3 * -cmath.pi / 4 - angle))
                    if angle > -cmath.pi and angle < -3 * cmath.pi / 4:
                        Rx_c2[k] = Rx_c2[k] * cmath.exp(1j * -(angle - -3 * cmath.pi / 4))

            b = KENG_phaserecovery()
            c2_vv = b.forth_QPSK(Rx_c2, tap=tap)
            phase2 = b.phase_adj

            phase0 = np.array(phase1) + np.array(phase2)
            c1c2c3_vv = c1c3_vv + c2_vv
            NEW_PLL_Rx = np.zeros([int(np.size(c1c2c3_vv)), 1], dtype='complex_')

            for i in range(a.oneside_span, np.size(c1c2c3_vv) + a.oneside_span):
                NEW_PLL_Rx[i - (a.oneside_span), 0] = Rx[i] * cmath.exp(-1j * phase0[0, i - (a.oneside_span)])

            Rx = NEW_PLL_Rx

            return NEW_PLL_Rx

    def constellation_assisted_ML(self, Rx_ori, Rx_assist, Rx_PLL, tap_2, tap_3, stage):
        inner = 2.5
        outter = 7.5
        target = [inner + 1j * inner, inner - 1j * inner, inner + 1j * outter, inner - 1j * outter,
                  outter + 1j * outter, outter - 1j * outter, outter + 1j * inner, outter - 1j * inner,
                  -inner + 1j * inner, -inner - 1j * inner, -inner + 1j * outter, -inner - 1j * outter,
                  -outter + 1j * outter, -outter - 1j * outter, -outter + 1j * inner, -outter - 1j * inner]
        # target=[1+1j, 1-1j, -1-1j, -1+1j]        
        decision_assist = np.zeros([1, np.size(Rx_assist)], dtype='complex_')
        decision_PLL = np.zeros([1, np.size(Rx_PLL)], dtype='complex_')
        Rx = Rx_ori
        if stage == 2:
            halftap = int((tap_2 - 1) / 2)
            for i in range(0, np.size(Rx_assist)):
                if Rx_assist[i] == 0:
                    decision_assist[0, i] = 0
                else:
                    decision = []
                    for k in range(len(target)):
                        decision.append(abs(Rx_assist[i] - target[k]))

                    decision_assist[0, i] = target[np.argmin(decision)]

            for i in range(0, np.size(Rx_PLL)):
                if Rx_PLL[i] == 0:
                    decision_PLL[0, i] = 0
                else:
                    decision = []
                    for k in range(len(target)):
                        decision.append(abs(Rx_PLL[i] - target[k]))

                    decision_PLL[0, i] = target[np.argmin(decision)]

            h = np.zeros([1, np.size(decision_assist)], dtype='complex_')
            decision_assist = np.mat(decision_assist)
            decision_PLL = np.mat(decision_PLL).T

            for i in range(halftap, np.size(h) - halftap):
                h[0, i] = decision_assist[0, i - halftap:i + halftap] * np.conj(
                    decision_PLL[i - halftap:i + halftap, 0])

            theta = np.zeros([1, np.size(h)])
            for i in range(0, np.size(theta)):
                theta[0, i] = cmath.phase(h[0, i])
                if h[0, i] == 0:
                    theta[0, i] = 0
                # else:
                #     theta[0,i] = math.atan( np.imag(h[0,i]) / np.real(h[0,i]) )

            ML = np.zeros([int(np.size(theta)), 1], dtype='complex_')
            for i in range(halftap, np.size(Rx) - halftap):
                ML[i, 0] = Rx[i, 0] * cmath.exp(-1j * theta[0, i])

            return ML

        # if stage==3:
        #     halftap_2=int((tap_2-1)/2)
        #     halftap=int((tap_3-1)/2)
        #     for i in range(0,np.size(Rx_PLL)):  
        #         a=[]
        #         for k in range(16):
        #             aa=np.abs(Rx_PLL[i]-target[k])
        #             a.append(aa[0])
        #         b=a.index(min(a))
        #         decision[0,i+halftap_2]=target[b]            

        #     h=np.zeros([1,np.size(decision)],dtype='complex_')
        #     decision=np.mat(decision).T             
        #     Rx=np.mat(Rx)

        #     for i in range(halftap,np.size(h)-halftap):
        #         h[0,i]=Rx[0,i-halftap:i+halftap]*np.conj(decision[i-halftap:i+halftap,0])

        #     theta=np.zeros([1,np.size(h)-2*halftap])
        #     for i in range(0,np.size(theta)):
        #         theta[0,i]=cmath.phase(h[0,i+halftap])            

        #     ML=np.zeros([int(np.size(theta)),1],dtype='complex_')
        #     for i in range(halftap,np.size(Rx)-halftap):
        #         ML[i-halftap]=Rx[0,i]*cmath.exp(-1j*theta[0,i-halftap])

        #     return ML            

    def QAM_4(self, Rx, c1_radius, c2_radius):  # https://www.sciencedirect.com/science/article/pii/S0030401814012152
        self.c1_radius = c1_radius
        self.c2_radius = c2_radius

        c1_radius = c1_radius
        c2_radius = c2_radius

        Rx = np.reshape(np.matrix(Rx), [np.size(Rx), 1])
        Rx_amplitude = abs(Rx)
        Rx_c1, Rx_c2, Rx_c3 = [], [], []

        for i in range(0, np.size(Rx)):
            if Rx_amplitude[i, 0] <= c1_radius:
                Rx_c1.append(Rx[i, 0])
                Rx_c2.append(0)
                Rx_c3.append(0)

            if Rx_amplitude[i, 0] > c1_radius and Rx_amplitude[i, 0] <= c2_radius:
                Rx_c1.append(0)
                Rx_c2.append(Rx[i, 0])
                Rx_c3.append(0)

            if Rx_amplitude[i, 0] > c2_radius:
                Rx_c1.append(0)
                Rx_c2.append(0)
                Rx_c3.append(Rx[i, 0])

        Rx_c1 = np.array(Rx_c1)
        Rx_c2 = np.array(Rx_c2)
        Rx_c3 = np.array(Rx_c3)
        # -----
        # c1mean=np.mean(abs(Rx_c1[Rx_c1!=0]))
        # c1_d=np.max(abs(Rx_c1[Rx_c1!=0]))-np.min(abs(Rx_c1[Rx_c1!=0]))
        # c2mean=np.mean(abs(Rx_c2[Rx_c2!=0]))
        # c2_d=np.max(abs(Rx_c2[Rx_c2!=0]))-np.min(abs(Rx_c2[Rx_c2!=0])) 
        # c3mean=np.mean(abs(Rx_c3[Rx_c3!=0]))
        # c3_d=np.max(abs(Rx_c3[Rx_c3!=0]))-np.min(abs(Rx_c3[Rx_c3!=0])) 

        # for i in range(np.size(Rx_c1)):
        #     if Rx_c1[i] !=0:
        #         Rx_c1[i] =  (abs(Rx_c1[i])-c1mean) / c1_d *(math.cos(cmath.phase(Rx_c1[i]))+1j*math.sin(cmath.phase(Rx_c1[i])))   
        #     if Rx_c2[i] !=0:
        #         Rx_c2[i] =  (abs(Rx_c2[i])-c2mean) / c2_d *(math.cos(cmath.phase(Rx_c2[i]))+1j*math.sin(cmath.phase(Rx_c2[i])))
        #     if Rx_c3[i] !=0:    
        #         Rx_c3[i] =  (abs(Rx_c3[i])-c3mean) / c3_d *(math.cos(cmath.phase(Rx_c3[i]))+1j*math.sin(cmath.phase(Rx_c3[i])))

        # print(np.mean(abs(Rx_c1[Rx_c1!=0])))
        # -----
        Rx_c1c3 = Rx_c1 + Rx_c3
        a = KENG_phaserecovery()
        c1c3_vv = a.forth_QPSK(Rx_c1c3, tap=61)

        phase1 = a.phase_adj

        Rx_ph = np.zeros([int(np.size(phase1)), 1], dtype='complex_')
        for i in range(np.size(Rx_ph)):
            Rx_ph[i, 0] = Rx[i, 0] * cmath.exp(-1j * phase1[0, i])

        # return Rx_ph
        ML = self.ML(Rx, Rx_ph, tap=39)
        # ML=self.constellation_assisted_ML(Rx,Rx_c1c3,Rx_ph,tap_2=a.tap,tap_3=stage3tap,stage=2)
        # ML=self.constellation_assisted_ML(Rx_c1_ori,Rx_c2_ori,Rx_c3_ori,ML,tap_2=a.tap,tap_3=stage3tap,stage=3)
        return ML

    def ML(self, Rx_ori, Rx_ph, tap):
        inner = 0.8
        outter = 2.7
        # inner = 1.5
        # outter = 4
        target = [inner + 1j * inner, inner - 1j * inner, inner + 1j * outter, inner - 1j * outter,
                  outter + 1j * outter, outter - 1j * outter, outter + 1j * inner, outter - 1j * inner,
                  -inner + 1j * inner, -inner - 1j * inner, -inner + 1j * outter, -inner - 1j * outter,
                  -outter + 1j * outter, -outter - 1j * outter, -outter + 1j * inner, -outter - 1j * inner]

        decision_Rx_ph = np.zeros([1, np.size(Rx_ph)], dtype='complex_')
        center = int((tap - 1) / 2)

        for i in range(0, np.size(Rx_ph)):
            decision = []
            for k in range(len(target)):
                decision.append(abs(Rx_ph[i,0] - target[k]))

            decision_Rx_ph[0, i] = target[np.argmin(decision)]

        h = np.zeros([1, np.size(decision_Rx_ph)], dtype='complex_')
        decision_Rx_ph = np.mat(decision_Rx_ph.T)
        Rx_ph = np.mat(Rx_ph.T)

        for i in range(center, np.size(h) - center):
            h[0, i] = Rx_ph[0, i - center:i + center + 1] * np.conj(decision_Rx_ph[i - center:i + center + 1, 0])

        theta = np.zeros([1, np.size(h)])
        ak = np.zeros([1, np.size(h)])
        theta_adj = np.zeros([1, np.size(h)])
        for i in range(0, np.size(theta)):
            theta[0, i] = cmath.phase(h[0, i])
            # ak[0, i] = ak[0, i - 1] + math.floor(0.5 - 4 * (theta[0, i] - theta[0, i - 1]) / 2 / cmath.pi)
            # theta_adj[0, i] = theta[0, i] + ak[0, i] * 2 * cmath.pi / 4
        ML = np.zeros([int(np.size(theta)), 1], dtype='complex_')
        for i in range(center, np.size(Rx_ori) - center):
            ML[i, 0] = Rx_ph[0, i] * cmath.exp(-1j * theta[0, i])

        return ML

    def QAM_5(self, Rx, c1_radius, c2_radius):  # https://www.sciencedirect.com/science/article/pii/S0030401814012152
        c1_radius = c1_radius
        c2_radius = c2_radius

        Rx = np.reshape(np.matrix(Rx), [np.size(Rx), 1])
        Rx_amplitude = abs(Rx)
        Rx_c1, Rx_c2, Rx_c3 = [], [], []

        for i in range(0, np.size(Rx)):
            if Rx_amplitude[i, 0] <= c1_radius:
                Rx_c1.append(Rx[i, 0])
                Rx_c2.append(0)
                Rx_c3.append(0)

            if Rx_amplitude[i, 0] > c1_radius and Rx_amplitude[i, 0] <= c2_radius:
                Rx_c1.append(0)
                Rx_c2.append(Rx[i, 0])
                Rx_c3.append(0)

            if Rx_amplitude[i, 0] > c2_radius:
                Rx_c1.append(0)
                Rx_c2.append(0)
                Rx_c3.append(Rx[i, 0])

        Rx_c1 = np.array(Rx_c1)
        Rx_c2 = np.array(Rx_c2)
        Rx_c3 = np.array(Rx_c3)
        # -----
        # c1mean=np.mean(abs(Rx_c1[Rx_c1!=0]))
        # c1_d=np.max(abs(Rx_c1[Rx_c1!=0]))-np.min(abs(Rx_c1[Rx_c1!=0]))
        # c2mean=np.mean(abs(Rx_c2[Rx_c2!=0]))
        # c2_d=np.max(abs(Rx_c2[Rx_c2!=0]))-np.min(abs(Rx_c2[Rx_c2!=0]))
        # c3mean=np.mean(abs(Rx_c3[Rx_c3!=0]))
        # c3_d=np.max(abs(Rx_c3[Rx_c3!=0]))-np.min(abs(Rx_c3[Rx_c3!=0]))

        # for i in range(np.size(Rx_c1)):
        #     if Rx_c1[i] !=0:
        #         Rx_c1[i] =  (abs(Rx_c1[i])-c1mean) / c1_d *(math.cos(cmath.phase(Rx_c1[i]))+1j*math.sin(cmath.phase(Rx_c1[i])))
        #     if Rx_c2[i] !=0:
        #         Rx_c2[i] =  (abs(Rx_c2[i])-c2mean) / c2_d *(math.cos(cmath.phase(Rx_c2[i]))+1j*math.sin(cmath.phase(Rx_c2[i])))
        #     if Rx_c3[i] !=0:
        #         Rx_c3[i] =  (abs(Rx_c3[i])-c3mean) / c3_d *(math.cos(cmath.phase(Rx_c3[i]))+1j*math.sin(cmath.phase(Rx_c3[i])))

        # print(np.mean(abs(Rx_c1[Rx_c1!=0])))
        # -----
        Rx_c1c3 = Rx_c1 + Rx_c3
        a = KENG_phaserecovery()
        c1c3_vv = a.forth_QPSK(Rx_c1c3, tap=101)
        phase1 = a.phase_adj

        b = KENG_phaserecovery()
        c2_vv = b.eighth_QPSK(Rx_c2, tap=301)
        # phase2 = b.phase_adj
        #
        Rx_ph = np.zeros([int(np.size(Rx_c1c3)), 1], dtype='complex_')
        for i in range(np.size(Rx_ph)):
            if Rx_c2[i] == 0:
                Rx_ph[i] = c1c3_vv[i]
            else:
                Rx_ph[i] = c2_vv[i]
        #
        # Rx_ph = np.zeros([int(np.size(phase1)), 1], dtype='complex_')
        # for i in range(np.size(Rx_ph)):
        #     Rx_ph[i, 0] = Rx[i, 0] * cmath.exp(-1j * phase1[0, i])

        # return Rx_ph
        # Rx_ph = np.reshape(Rx_ph, (-1,))
        # Rx_ph = b.forth_QPSK(Rx_ph, tap=101)
        # phase3 = b.phase_adj

        # for i in range(np.size(Rx_ph)):
        #     Rx_ph[i] = Rx_ph[i] * cmath.exp(-1j * phase3[0, i])
        #

        # return Rx_ph

        ML = self.ML(Rx, Rx_ph, tap=39)
        return ML

    def QAM_6(self, Rx, c1_radius, c2_radius):
        Rx_ph = self.QAM_4(Rx, c1_radius, c2_radius)
        Rx_ph = self.QAM_4(Rx_ph, c1_radius, c2_radius)

        return Rx_ph

    def PLL(self, isig):
        self.isig = isig
        self.bandwidth = 1e-3
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

    def FreqOffsetComp(self, rx, fsamp=56e9,fres=1e6):  # CoarseFrequencyOffset compensation based on prediogram method for M-QAM
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
        print('Frequency offset=', estFreqOffset)
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




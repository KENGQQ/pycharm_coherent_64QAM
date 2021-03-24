# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:28:33 2020

@author: KENG
"""

import numpy as np

class KENG_Tx2Bit:
    def __init__(self,PAM_order):
        self.PAM_order=PAM_order
        self.level_value = np.linspace(- PAM_order + 1 , PAM_order -1 ,PAM_order)
        level_classification=np.array([float('-Inf')])
        level_classification=np.append(level_classification,np.linspace(- PAM_order + 2,PAM_order - 2,PAM_order-1))
        self.level_classification=np.append(level_classification,float('Inf'))
     
    def return_Tx(self, Tx_Signal):
        self.Tx_signal_symbol=np.zeros([np.size(Tx_Signal),1])
        for i in range(0, self.PAM_order):
            self.Tx_signal_symbol[list(self.level_classification[i+1]>Tx_Signal) and 
                                  list(self.level_classification[i]<=Tx_Signal)]=self.level_value[i]

        return self.Tx_signal_symbol
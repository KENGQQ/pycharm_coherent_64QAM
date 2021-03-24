# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:42:41 2020

@author: KENG
"""
import numpy as np
class KENG_downsample:
    def __init__(self,down_coeff):
        self.down_coeff=down_coeff

    def return_value(self,Signal):
        self.Signal =Signal
        # self.Signal=np.matrix(self.Signal,dtype='complex_'
        self.Signal=np.matrix(self.Signal)

        self.Signal=np.reshape(self.Signal,[1,np.size(self.Signal)])
        self.quantity=int((np.size(self.Signal)-1)/self.down_coeff)

        downsample_Signal=[self.Signal[0,0]]
        for i in range(1,self.quantity+1):
            downsample_Signal.append(self.Signal[0,self.down_coeff*i])
        
        return np.array(downsample_Signal)
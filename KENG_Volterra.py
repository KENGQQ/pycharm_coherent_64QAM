# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
class KENG_volterra:                  #only first and third order, (based on  https://www.nt.tf.uni-kiel.de/de/veroffentlichungen/konferenzen-fachblaetter/2014_rath_itg.pdf?fbclid=IwAR0C9XNNf7j-vVqGZgBmsTmisx7H8d20MyVktPb7O4hJNs4UmWoG-vHksTM)
    def __init__(self, Rx_signal, Tx_signal, Memory_length, train_number):
        self.train_number=train_number
        
        self.Memory_length_firstorder=Memory_length[0]
        self.Memory_length_secondorder=Memory_length[1]      #useless
        self.Memory_length_thirdorder=Memory_length[2]
        
        self.Tx_signal=np.reshape(np.matrix(Tx_signal),[np.size(Tx_signal),1])

        self.Rx_vol_train=Rx_signal[:train_number]     #row
        self.Rx_vol_train=np.reshape(np.matrix(self.Rx_vol_train),[1,np.size(self.Rx_vol_train)])
        
        self.Rx_vol_test=Rx_signal[train_number:]    #row
        self.Rx_vol_test=np.reshape(np.matrix(self.Rx_vol_test),[1,np.size(self.Rx_vol_test)])
        
    def first_order_only(self):
        self.Memory_length_oneside_span_firstorder=int((self.Memory_length_firstorder-1)/2)
        
        self.F_matrix_train_row=int(np.size(self.Rx_vol_train,1)-2*self.Memory_length_oneside_span_firstorder)
        self.F_matrix_train_colume=int(1+self.Memory_length_firstorder)
        self.F_matrix_train=np.zeros([self.F_matrix_train_row,self.F_matrix_train_colume],dtype='complex_')

        self.Tx_train_center=int((self.Memory_length_firstorder-1)/2)
        for k in range(0,self.F_matrix_train_row):
            self.F_matrix_train[k,:]=np.append(1,self.Rx_vol_train[0,self.Tx_train_center-self.Memory_length_oneside_span_firstorder : self.Tx_train_center+self.Memory_length_oneside_span_firstorder+1])
            self.Tx_train_center+=1

        self.F_matrix_train=np.matrix(self.F_matrix_train)
        self.F_matrix_train_Trans=self.F_matrix_train.T
     #---------------------------------------------------------------------------------------------------   
        self.F_matrix_test_row=int(np.size(self.Rx_vol_test,1)-2*self.Memory_length_oneside_span_firstorder)
        self.F_matrix_test_colume=int(1+self.Memory_length_firstorder)
        self.F_matrix_test=np.zeros([self.F_matrix_test_row,self.F_matrix_test_colume],dtype='complex_')
        self.Tx_test_center=int((self.Memory_length_firstorder-1)/2)
        for k in range(0,self.F_matrix_test_row):       
            self.F_matrix_test[k,:]=np.append(1,self.Rx_vol_test[0,self.Tx_test_center-self.Memory_length_oneside_span_firstorder : self.Tx_test_center+self.Memory_length_oneside_span_firstorder+1])
            self.Tx_test_center+=1
            
        self.F_matrix_test=np.matrix(self.F_matrix_test)
        
       #---------------------------------------------------------------------------------------------------         
        self.Tx_vol_train=self.Tx_signal[self.Memory_length_oneside_span_firstorder :self.train_number-self.Memory_length_oneside_span_firstorder,0]
        self.Tx_vol_train=np.matrix(self.Tx_vol_train)
        
        A=np.linalg.inv(np.conj(self.F_matrix_train_Trans)*self.F_matrix_train)
        B=np.conj(self.F_matrix_train_Trans)*self.Tx_vol_train

        self.wiener_matrix=A*B
        self.Rx_vol_train=self.F_matrix_train*self.wiener_matrix
        self.Rx_vol_test =self.F_matrix_test *self.wiener_matrix
        self.Rx_vol_test=np.array(self.Rx_vol_test)
        
        self.Tx_vol_test =self.Tx_signal[self.train_number+self.Memory_length_oneside_span_firstorder:-self.Memory_length_oneside_span_firstorder,0]        
        self.Tx_vol_test=np.array(self.Tx_vol_test)
    

    def first_third_order(self):
        if self.Memory_length_thirdorder<=self.Memory_length_firstorder:

            self.Memory_length_oneside_span_firstorder=int((self.Memory_length_firstorder-1)/2)
            self.Memory_length_oneside_span_thirdorder=int((self.Memory_length_thirdorder-1)/2)

        
            self.F_matrix_train_row=int(np.size(self.Rx_vol_train,1)-2*self.Memory_length_oneside_span_firstorder)
            self.F_matrix_train_colume=int(1+self.Memory_length_firstorder+0.5*self.Memory_length_thirdorder**2*(self.Memory_length_thirdorder+1))
            self.F_matrix_train=np.zeros([self.F_matrix_train_row,self.F_matrix_train_colume],dtype='complex_')

            self.Tx_train_center=int((self.Memory_length_firstorder-1)/2)     

            for k in range(0,self.F_matrix_train_row):
                Rx_in_memory_length=np.mat(self.Rx_vol_train[0,self.Tx_train_center-self.Memory_length_oneside_span_thirdorder:self.Tx_train_center+self.Memory_length_oneside_span_thirdorder+1],dtype='complex_')
                Rx_matrix=Rx_in_memory_length.T*Rx_in_memory_length
                Rx_matrix=np.reshape(np.triu(Rx_matrix),[np.size(Rx_matrix),1])
                Rx_matrix=np.append(Rx_matrix[0],np.delete(Rx_matrix, np.where(Rx_matrix== [0]), axis=0))
                Rx_matrix=np.matrix(Rx_matrix).T
                F_matrix_train_thirdorder_coeff=Rx_matrix*np.conj(Rx_in_memory_length)   
                F_matrix_train_thirdorder_coeff=np.reshape(F_matrix_train_thirdorder_coeff.T,[1,np.size(F_matrix_train_thirdorder_coeff)])
                a=np.append(1,self.Rx_vol_train[0,self.Tx_train_center-self.Memory_length_oneside_span_firstorder : self.Tx_train_center+self.Memory_length_oneside_span_firstorder+1])
                aa=np.append(a,F_matrix_train_thirdorder_coeff)
  
                self.F_matrix_train[k,:]=aa           
                self.Tx_train_center+=1
            
            self.F_matrix_train=np.matrix(self.F_matrix_train)
            self.F_matrix_train_Trans=self.F_matrix_train.T
        # #---------------------------------------------------------------------------------------------------           
            self.F_matrix_test_row=int(np.size(self.Rx_vol_test,1)-2*self.Memory_length_oneside_span_firstorder)
            self.F_matrix_test_colume=int(1+self.Memory_length_firstorder+0.5*self.Memory_length_thirdorder**2*(self.Memory_length_thirdorder+1))
            self.F_matrix_test=np.zeros([self.F_matrix_test_row,self.F_matrix_test_colume],dtype='complex_')
            
            self.Tx_test_center=int((self.Memory_length_firstorder-1)/2)
            for k in range(0,self.F_matrix_test_row):
                Rx_in_memory_length=np.mat(self.Rx_vol_test[0,self.Tx_test_center-self.Memory_length_oneside_span_thirdorder:self.Tx_test_center+self.Memory_length_oneside_span_thirdorder+1],dtype='complex_')
                Rx_matrix=Rx_in_memory_length.T*Rx_in_memory_length
                Rx_matrix=np.reshape(np.triu(Rx_matrix),[np.size(Rx_matrix),1])
                Rx_matrix=np.append(Rx_matrix[0],np.delete(Rx_matrix, np.where(Rx_matrix== [0]), axis=0))
                Rx_matrix=np.matrix(Rx_matrix).T
                F_matrix_test_thirdorder_coeff=Rx_matrix*np.conj(Rx_in_memory_length)
                F_matrix_test_thirdorder_coeff=np.reshape(F_matrix_test_thirdorder_coeff.T,[1,np.size(F_matrix_test_thirdorder_coeff)])
                
                a=np.append(1,self.Rx_vol_test[0,self.Tx_test_center-self.Memory_length_oneside_span_firstorder : self.Tx_test_center+self.Memory_length_oneside_span_firstorder+1])
                aa=np.append(a,F_matrix_test_thirdorder_coeff)       
                
                self.F_matrix_test[k,:]=aa
                self.Tx_test_center+=1      
                
            self.F_matrix_test=np.matrix(self.F_matrix_test)

        # #---------------------------------------------------------------------------------------------------           
        self.Tx_vol_train=self.Tx_signal[self.Memory_length_oneside_span_firstorder :self.train_number-self.Memory_length_oneside_span_firstorder,0]
        self.Tx_vol_train=np.matrix(self.Tx_vol_train)
        
        A=np.linalg.inv(np.conj(self.F_matrix_train_Trans)*self.F_matrix_train)
        B=np.conj(self.F_matrix_train_Trans)*self.Tx_vol_train

        self.wiener_matrix=A*B
        self.Rx_vol_train=self.F_matrix_train*self.wiener_matrix
        self.Rx_vol_test =self.F_matrix_test *self.wiener_matrix
        self.Rx_vol_test=np.array(self.Rx_vol_test)

        self.Tx_vol_test =self.Tx_signal[self.train_number+self.Memory_length_oneside_span_firstorder:-self.Memory_length_oneside_span_firstorder,0]
        self.Tx_vol_test=np.array(self.Tx_vol_test)

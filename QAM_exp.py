import numpy as np
import scipy.signal as signal
from subfunction.Constellation import *
from subfunction.DataNormalize import *
from subfunction.Histogram2D import *
from subfunction.IQimbaCompensator import *
# from subfunction.corr import *
from subfunction.BERcount import *
from subfunction.SNR import *
from subfunction.Downsample import *

from CMA import *

from KENG_Tx2Bit import *
from KENG_downsample import *
from KENG_phaserecovery import *
from KENG_Parameter_16QAM import *
from KENG_phaserecovery import *
from KENG_correlation import *

from Equalizer import *
from Phaserecovery import *


parameter = Parameter(r'data\20200811_finisar B2B16QAM\Vblock1.mat',simulation=False)


print('SymbolRate={}'.format(parameter.symbolRate / 1e9), 'Pamorder={}'.format(parameter.pamorder),
      'resamplenumber={}'.format(parameter.resamplenumber))
print('Tx Length={}'.format(len(parameter.TxXI)), 'Rx Length={}'.format(len(parameter.RxXI)))
# Tx Normalize
Tx2bit=KENG_Tx2Bit(PAM_order=4)


TxXI ,TxXQ = DataNormalize(parameter.TxXI,parameter.TxXQ,parameter.pamorder)
TxYI ,TxYQ = DataNormalize(parameter.TxYI,parameter.TxYQ,parameter.pamorder)

TxXI=Tx2bit.return_Tx(TxXI)
TxXQ=Tx2bit.return_Tx(TxXQ)
TxYI=Tx2bit.return_Tx(TxYI)
TxYQ=Tx2bit.return_Tx(TxYQ)

Tx_Signal_X=Constellation(TxXI,TxXQ,parameter.pamorder)
Tx_Signal_Y=Constellation(TxYI,TxYQ,parameter.pamorder)
Histogram2D('Tx',Tx_Signal_X[0:10000])
# Rx Upsample
Rx_XI, Rx_XQ = DataNormalize(signal.resample_poly(parameter.RxXI, up=parameter.upsamplenum, down=1),
                              signal.resample_poly(parameter.RxXQ, up=parameter.upsamplenum, down=1),
                              parameter.pamorder)
Rx_YI, Rx_YQ = DataNormalize(signal.resample_poly(parameter.RxYI, up=parameter.upsamplenum, down=1),
                              signal.resample_poly(parameter.RxYQ, up=parameter.upsamplenum, down=1),
                              parameter.pamorder)

print('Tx_Resample Length={}'.format(len(Tx_Signal_X)), 'Rx_Resample Length={}'.format(len(Rx_XI)))
prbs = np.ceil(DataNormalize(parameter.PRBS, [], parameter.pamorder))
snrscan = np.zeros((parameter.resamplenumber, 1))
evmscan = np.zeros((parameter.resamplenumber, 1))

#Eye position scan2
for eyepos in range(7,8):
    down_num = eyepos
    Rx_XI_eye = signal.resample_poly(Rx_XI[down_num:], up=1, down=parameter.resamplenumber / 2)
    Rx_XQ_eye = signal.resample_poly(Rx_XQ[down_num:], up=1, down=parameter.resamplenumber / 2)
    Rx_YI_eye = signal.resample_poly(Rx_YI[down_num:], up=1, down=parameter.resamplenumber / 2)
    Rx_YQ_eye = signal.resample_poly(Rx_YQ[down_num:], up=1, down=parameter.resamplenumber / 2)
    
    Rx_Signal_X = Constellation(Rx_XI_eye, Rx_XQ_eye, parameter.pamorder)
    Rx_Signal_Y = Constellation(Rx_YI_eye, Rx_YQ_eye, parameter.pamorder)
    Histogram2D('Rx', Rx_Signal_X[0:32767])
    
    
    cma = CMA(Rx_Signal_X[:200000], Rx_Signal_Y[:200000])
    # cma = CMA(Rx_Signal_X, Rx_Signal_Y)
    print('CMA Batch Size={}'.format(cma.batchsize), 'CMA Stepsize={}'.format(cma.stepsize),
          'CMA OverHead={}%'.format(cma.overhead * 100))
    # CMA Single
    # cma.run_single()
    # Rx_X_CMA, Rx_Y_CMA = Downsample(cma.rx_x_single, 2), Downsample(cma.rx_y_single, 2)
    # print(np.shape(Rx_X_CMA), cma.costfunx, cma.costfuny)
    # Histogram2D(Rx_X_CMA[cma.center:32767-cma.center+1])
    # CMA Batch
    cma.run_16qam()
    Rx_X_CMA, Rx_Y_CMA = Downsample(cma.rx_x_cma, 2, cma.center), Downsample(cma.rx_y_cma, 2, cma.center)
    print(cma.costfunx[0][0:10])
    Histogram2D('CMA', Rx_X_CMA[0])
    Rx_X_iqimba = IQimbaCompensator(Rx_X_CMA, 1e-4)
    Histogram2D("IQimba", Rx_X_iqimba[0])
    
    
    phaserec = Phaserecovery(Rx_X_iqimba)
    # Rx_X_recovery, B, C = phaserec.PLL_(0.01, 0.707)
    Rx_X_recovery = phaserec.DD_PLL()
    Histogram2D('DD-PLL', Rx_X_recovery[0])

    ph=KENG_phaserecovery()
    # PLL_Rx=ph.QAM_4(Rx_X_iqimba[0],c1_radius=math.sqrt(2)+math.sqrt(2)/2,c2_radius=(math.sqrt(10)+math.sqrt(18))/2) 
    # PLL_Rx=ph.QAM(Rx_X_iqimba[0],c1_radius=math.sqrt(2),c2_radius=math.sqrt(10))
    PLL_Rx=ph.QAM_4(Rx_X_iqimba[0],c1_radius=math.sqrt(2),c2_radius=math.sqrt(10))
    Histogram2D('KENG_PLL_ML',PLL_Rx[:,0])   
    
    
    Correlation=KENG_corr(window_length=150)
    Rx_real, Tx_real=Correlation.calculate_Rx(np.real(PLL_Rx[0:60000,0]),TxXI[0:60000])   
    Rx_imag, Tx_imag=Correlation.calculate_Rx(np.imag(-PLL_Rx[0:60000,0]),TxXQ[0:60000])   
    # Rx_real, Tx_real=Correlation.calculate_Rx(np.real(Rx_X_recovery[0,0:80000]),TxXI[0:80000])   
    # Rx_imag, Tx_imag=Correlation.calculate_Rx(np.imag(-Rx_X_recovery[0,0:80000]),TxXQ[0:80000])  


    # Rx_corr,Tx_corr=Correlation.calculate_Rx(PLL_Rx[0:90000,0],Tx_Signal_X[0:90000])   
    # snr, evm = SNR(np.array(Rx_corr[0:10000,0]), np.array(Tx_corr[0:10000,0]))
    # print(snr, evm)
    # Rx_real, Tx_real = corr(np.real(PLL_Rx[:,0]), TxXI[:,0], parameter.Prbsnum)
    # Rx_imag, Tx_imag = corr(np.imag(PLL_Rx[:,0]), TxXQ[:,0], parameter.Prbsnum)
   
    Rx_corr = Rx_real[0:35000] + 1j * Rx_imag[0:35000]
    Tx_corr = Tx_real[0:35000] + 1j * Tx_imag[0:35000]
   
    
    # Rx_corr, Tx_corr = corr(PLL_Rx[:,0], Tx_Signal_X, parameter.Prbsnum)
    # Rx_real, Tx_real = corr(np.real(Rx_X_recovery[0]), np.real(Tx_Signal_X), parameter.Prbsnum)
    # Rx_imag, Tx_imag = corr(np.imag(Rx_X_recovery[0]), np.imag(Tx_Signal_X), parameter.Prbsnum)
    # Rx_corr = Rx_real + 1j * Rx_imag
    # Tx_corr = Tx_real + 1j * Tx_imag
    # snr, evm = SNR(Rx_corr,Tx_corr)
    snr, evm = SNR(np.array(Rx_corr.T[0,:]), np.array(Tx_corr.T[0,:]))
    # bercount = BERcount(np.array(Tx_corr), np.array(Rx_corr), parameter.pamorder)
    # print(bercount)
    print(snr, evm)
    # snrscan[eyepos] = snr
    # evmscan[eyepos] = evm
# print(np.max(snrscan), np.argmax(snrscan))
#---
equalizer_real = Equalizer(np.real(np.array(Tx_corr.T)[0,:]), np.real(np.array(Rx_corr.T)[0,:]), 3, [11, 31, 31], 0.5)
equalizer_imag = Equalizer(np.imag(np.array(Tx_corr.T)[0,:]), np.imag(np.array(Rx_corr.T)[0,:]), 3, [11, 31, 31], 0.5)
Tx_volterra_real, Rx_volterra_real = equalizer_real.realvolterra()
Tx_volterra_imag, Rx_volterra_imag = equalizer_imag.realvolterra()
Tx_real_volterra = Tx_volterra_real + 1j * Tx_volterra_imag
Rx_real_volterra = Rx_volterra_real + 1j * Rx_volterra_imag
#---
# equalizer_complex = Equalizer(Tx_corr, Rx_corr, 3, [11, 3, 3], 0.5)
# equalizer_complex = Equalizer( np.array(Tx_corr.T)[0,:], np.array(Rx_corr.T)[0,:], 3, [21, 3, 1], 0.1)

# equalizer_real = Equalizer(np.real(Tx_corr), np.real(Rx_corr), 3, [11, 11, 11])
# equalizer_imag = Equalizer(np.imag(Tx_corr), np.imag(Rx_corr), 3, [11, 11, 11])
# Tx_volterra_real, Rx_volterra_real = equalizer_real.realvolterra()
# Tx_volterra_imag, Rx_volterra_imag = equalizer_imag.realvolterra()
# Tx_real_volterra = Tx_volterra_real + 1j * Tx_volterra_imag
# Rx_real_volterra = Rx_volterra_real + 1j * Rx_volterra_imag
# Tx_complex_volterra, Rx_complex_volterra = equalizer_complex.complexvolterra()
snr_volterra, evm_volterra = SNR(Tx_real_volterra, Rx_real_volterra)
# snr_volterra, evm_volterra = SNR(Rx_complex_volterra, Tx_complex_volterra)
# bercount = BERcount(Rx_complex_volterra, Tx_complex_volterra, parameter.pamorder)
# bercount = BERcount(Tx_complex_volterra, Rx_complex_volterra, parameter.pamorder)
# print(bercount)
print(snr_volterra, evm_volterra)
Histogram2D("ComplexVolterra", Rx_real_volterra, snr_volterra, evm_volterra)
# if __name__ == '__main__':
#     main()
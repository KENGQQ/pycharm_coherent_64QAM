import scipy.io
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
from subfunction.excelrecord import *

from CMA_64QAM import *

from KENG_Tx2Bit import *
from KENG_downsample import *
from KENG_Parameter_64QAM import *
from KENG_phaserecovery_64QAM import *
from KENG_correlation import *
from KENG_Volterra import *
from KENG_64QAM_LogicTx import *

from Equalizer import *
# from Phaserecovery import *
# import torch
# from Equalizer_RBF import *
# from Rolling_window import  rolling_window
# from torch.utils import data as Data
#
from tqdm import tqdm
from CD_compensator import *

address = r'C:\Users\keng\Google 雲端硬碟 (keng.eo08g@nctu.edu.tw)\OptsimData_coherent\QAM64_data/'
# address = r'C:\Users\kengw\Google 雲端硬碟 (keng.eo08g@nctu.edu.tw)\OptsimData_coherent\QAM64_data/'
folder = '20210604_DATA_Bire/100KLW_1GFO_50GBW_0dBLO_sample32_1000ns_CD-1280_EDC0_TxO-2dBm_RxO-08dBm_OSNR34dB_LO00dBm_fiber_PMD_Bire/'
address += folder

Imageaddress = address + 'image3'
parameter = Parameter(address, symbolRate=56e9, pamorder=8,simulation=True)
# open_excel(address)
##################### control  #####################
isplot = 1
iswrite = 0
xpart, ypart = 1, 1
eyestart, eyeend, eyescan = 29, 30, 1
tap1_start, tap1_end ,tap1_scan= 15, 17, 2 ;tap2_start, tap2_end, tap2_scan = 43, 45, 2;tap3_start, tap3_end, tap3_scan = 15, 17, 2
cma_stage= [1, 2, 3]; cma_iter = [30, 10, 10]
isrealvolterra = 0
iscomplexvolterra = 0
##################### intialize  #####################
window_length = 7000
# correlation_length = 110000
# correlation_length = 27000
correlation_length = 55000
final_length = correlation_length - 9000

CMAstage1_tap, CMAstage1_stepsize_x, CMAstage1_stepsize_x, CMAstage1_iteration = 0, 0, 0, 0
CMAstage2_tap, CMAstage2_stepsize_x, CMAstage2_stepsize_x, CMAstage2_iteration = 0, 0, 0, 0
CMAstage3_tap, CMAstage3_stepsize_x, CMAstage3_stepsize_x, CMAstage3_iteration = 0, 0, 0, 0
CMA_cost_X1, CMA_cost_X2, CMA_cost_X3 = 0, 0, 0
CMA_cost_Y1, CMA_cost_Y2, CMA_cost_Y3= 0, 0, 0
XIshift, XQshift, XI_corr, XQ_corr, SNR_X, EVM_X, bercount_X= 0, 0, 0, 0, 0, 0, 0
YIshift, YQshift, YI_corr, YQ_corr, SNR_Y, EVM_Y, bercount_Y= 0, 0, 0, 0, 0, 0, 0
##############################################################################################################################

print("symbolrate = {}Gbit/s\npamorder = {}\nresamplenumber = {}".format(parameter.symbolRate / 1e9, parameter.pamorder, parameter.resamplenumber))
Tx2Bit = KENG_Tx2Bit(PAM_order=parameter.pamorder)
downsample_Tx = KENG_downsample(down_coeff=parameter.resamplenumber)
downsample_Rx = KENG_downsample(down_coeff=parameter.resamplenumber)

Rx_XI, Rx_XQ = DataNormalize(parameter.RxXI, parameter.RxXQ, parameter.pamorder)
Rx_YI, Rx_YQ = DataNormalize(parameter.RxYI, parameter.RxYQ, parameter.pamorder)

cd_compensator = CD_compensator(Rx_XI + 1j * Rx_XQ, Rx_YI + 1j * Rx_YQ, Gbaud=56e9 * 32, KM=80)  # 39.62
CD_X, CD_Y = cd_compensator.overlap_save(Nfft=len(parameter.RxXI), NOverlap=4096)  # 4096
Rx_XI = np.real(CD_X); Rx_XQ = np.imag(CD_X)
Rx_YI = np.real(CD_Y); Rx_YQ = np.imag(CD_Y)

XSNR, XEVM, YSNR, YEVM = np.zeros(parameter.resamplenumber), np.zeros(parameter.resamplenumber), np.zeros(parameter.resamplenumber), np.zeros(parameter.resamplenumber)

LogTxXI_LSB = downsample_Tx.return_value(parameter.LogTxXI_LSB)
LogTxXI_CSB = downsample_Tx.return_value(parameter.LogTxXI_CSB)
LogTxXI_MSB = downsample_Tx.return_value(parameter.LogTxXI_MSB)

LogTxXQ_LSB = downsample_Tx.return_value(parameter.LogTxXQ_LSB)
LogTxXQ_CSB = downsample_Tx.return_value(parameter.LogTxXQ_CSB)
LogTxXQ_MSB = downsample_Tx.return_value(parameter.LogTxXQ_MSB)

LogTxYI_LSB = downsample_Tx.return_value(parameter.LogTxYI_LSB)
LogTxYI_CSB = downsample_Tx.return_value(parameter.LogTxYI_CSB)
LogTxYI_MSB = downsample_Tx.return_value(parameter.LogTxYI_MSB)

LogTxYQ_LSB = downsample_Tx.return_value(parameter.LogTxYQ_LSB)
LogTxYQ_CSB = downsample_Tx.return_value(parameter.LogTxYQ_CSB)
LogTxYQ_MSB = downsample_Tx.return_value(parameter.LogTxYQ_MSB)

TxXI, TxXQ = QAM64_LogicTx(LogTxXI_LSB, LogTxXI_CSB, LogTxXI_MSB, LogTxXQ_LSB, LogTxXQ_CSB, LogTxXQ_MSB)
TxYI, TxYQ = QAM64_LogicTx(LogTxYI_LSB, LogTxYI_CSB, LogTxYI_MSB, LogTxYQ_LSB, LogTxYQ_CSB, LogTxYQ_MSB)
# Histogram2D('Tx_X_normalized', Tx_Signal_X, Imageaddress)
# Histogram2D('Tx_X_normalized', Tx_Signal_Y, Imageaddress)

for eyepos in range(eyestart, eyeend, 1):
    down_num = eyepos

    # n = 1
    # RxXI = signal.resample_poly(Rx_XI[down_num:], up=1, down=parameter.resamplenumber / n)
    # RxXQ = signal.resample_poly(Rx_XQ[down_num:], up=1, down=parameter.resamplenumber / n)
    # RxYI = signal.resample_poly(Rx_YI[down_num:], up=1, down=parameter.resamplenumber / n)
    # RxYQ = signal.resample_poly(Rx_YQ[down_num:], up=1, down=parameter.resamplenumber / n)
    # Rx_Signal_X = RxXI[:, 0] + 1j * RxXQ[:, 0]
    # Rx_Signal_Y = RxYI[:, 0] + 1j * RxYQ[:, 0]

    RxXI = downsample_Rx.return_value(Rx_XI[down_num:])
    RxXQ = downsample_Rx.return_value(Rx_XQ[down_num:])
    RxYI = downsample_Rx.return_value(Rx_YI[down_num:])
    RxYQ = downsample_Rx.return_value(Rx_YQ[down_num:])
    Rx_Signal_X = RxXI + 1j * RxXQ
    Rx_Signal_Y = RxYI + 1j * RxYQ
    # Histogram2D('Rx_X_origin_{}'.format(eyepos), Rx_Signal_X, Imageaddress)
    # Histogram2D('Rx_Y_origin_{}'.format(eyepos), Rx_Signal_Y, Imageaddress)
    Histogram2D_thesis('Rx_X_origin_{}'.format(eyepos), Rx_Signal_X, Imageaddress)
    Histogram2D_thesis('Rx_Y_origin_{}'.format(eyepos), Rx_Signal_Y, Imageaddress)
    #########IQimba################
    # Rx_Signal_X = np.reshape(Rx_Signal_X,(1,-1))
    # Rx_Signal_Y = np.reshape(Rx_Signal_Y,(1,-1))
    # Rx_X_iqimba = IQimbaCompensator(Rx_Signal_X, 1e-4)
    # Rx_Y_iqimba = IQimbaCompensator(Rx_Signal_Y, 1e-4)
    # # Histogram2D("IQimba", Rx_X_iqimba[0])
    # Rx_Signal_X = np.reshape(Rx_X_iqimba, [Rx_X_iqimba.size,])
    # Rx_Signal_Y = np.reshape(Rx_Y_iqimba, [Rx_Y_iqimba.size,])
    ##########IQimba################
    for tap_1 in range(tap1_start, tap1_end, tap1_scan):
        print("eye : {} ,tap : {}".format(eyepos, tap_1))

        cma = CMA_single(Rx_Signal_X, Rx_Signal_Y, taps=tap_1, iter=cma_iter[0], mean=0)
        cma.stepsize_x = cma.stepsizelist[3]  ; CMAstage1_stepsize_x = cma.stepsize_x;
        cma.stepsize_y = cma.stepsizelist[3]  ; CMAstage1_stepsize_y = cma.stepsize_y;
        cma.qam_4_butter_RD(stage=cma_stage[0])
        # cma.qam_4_side_RD(stage=cma_stage[0])
        Rx_X_CMA_stage1 = cma.rx_x_cma[cma.rx_x_cma != 0]
        Rx_Y_CMA_stage1 = cma.rx_y_cma[cma.rx_y_cma != 0]
        CMAstage1_tap, CMAstage1_iteration, CMA_cost_X1, CMA_cost_Y1 = tap_1, cma_iter[0], np.round(cma.costfunx[0][-1], 4), np.round(cma.costfuny[0][-1], 4)
        if isplot == True:
            Histogram2D('CMA_X_{}_stage1 taps={} {}'.format(eyepos, cma.cmataps, cma.type), Rx_X_CMA_stage1, Imageaddress)
            Histogram2D('CMA_Y_{}_stage1 taps={} {}'.format(eyepos, cma.cmataps, cma.type), Rx_Y_CMA_stage1, Imageaddress)
        Rx_X_CMA = Rx_X_CMA_stage1
        Rx_Y_CMA = Rx_Y_CMA_stage1

        # cma = CMA_single(Rx_X_CMA_stage1, Rx_Y_CMA_stage1, taps=tap2_end, iter=cma_iter[1], mean=0)
        # cma.stepsize_x = cma.stepsizelist[9]  ; CMAstage2_stepsize_x = cma.stepsize_x;
        # cma.stepsize_y = cma.stepsizelist[9]  ; CMAstage2_stepsize_y = cma.stepsize_y;
        # cma.qam_4_butter_RD(stage=cma_stage[1])
        # # cma.qam_4_side_RD(stage=cma_stage[1])
        # Rx_X_CMA_stage2 = cma.rx_x_cma[cma.rx_x_cma != 0]
        # Rx_Y_CMA_stage2 = cma.rx_y_cma[cma.rx_y_cma != 0]
        # CMAstage2_tap, CMAstage2_iteration, CMA_cost_X2, CMA_cost_Y2 = tap2_end, cma_iter[1], np.round(cma.costfunx[0][-1], 4), np.round(cma.costfuny[0][-1], 4)
        # if isplot == True:
        #     Histogram2D('CMA_X_{}_stage1 taps={} {}'.format(eyepos, cma.cmataps, cma.type), Rx_X_CMA_stage1, Imageaddress)
        #     Histogram2D('CMA_Y_{}_stage1 taps={} {}'.format(eyepos, cma.cmataps, cma.type), Rx_Y_CMA_stage1, Imageaddress)
        # Rx_X_CMA = Rx_X_CMA_stage2
        # Rx_Y_CMA = Rx_Y_CMA_stage2
        #
        for taps_3 in range(tap3_start, tap3_end, tap3_scan):
            cma = CMA_single(Rx_X_CMA_stage1, Rx_Y_CMA_stage1, taps=taps_3, iter=cma_iter[2], mean=0)
            cma.stepsize_x = cma.stepsizelist[12]  ; CMAstage3_stepsize_x = cma.stepsize_x;
            cma.stepsize_y = cma.stepsizelist[12]  ; CMAstage3_stepsize_y = cma.stepsize_y;
            cma.qam_4_butter_RD(stage=cma_stage[2])
            # cma.qam_4_side_RD(stage = cma_stage[2])
            Rx_X_CMA_stage3 = cma.rx_x_cma[cma.rx_x_cma != 0]
            Rx_Y_CMA_stage3 = cma.rx_y_cma[cma.rx_y_cma != 0]
            CMAstage3_tap, CMAstage3_iteration, CMA_cost_X3, CMA_cost_Y3 = taps_3, cma_iter[2], np.round(cma.costfunx[0][-1], 4), np.round(cma.costfuny[0][-1], 4)
            if isplot == True:
                Histogram2D('CMA_X_{}_stage3 taps={} {}'.format(eyepos, cma.cmataps, cma.type), Rx_X_CMA_stage3, Imageaddress)
                Histogram2D('CMA_Y_{}_stage3 taps={} {}'.format(eyepos, cma.cmataps, cma.type), Rx_Y_CMA_stage3, Imageaddress)
            Rx_X_CMA = Rx_X_CMA_stage3
            Rx_Y_CMA = Rx_Y_CMA_stage3

#--------------------------- X PART-------------------------------------------------
        if xpart == True:
            print('================================')
            print('X part')
            ph = KENG_phaserecovery()
            FOcompen_X = ph.FreqOffsetComp(Rx_X_CMA, fsamp=56e9, fres=1e5)
            if isplot == True: Histogram2D('KENG_FOcompensate_X', FOcompen_X, Imageaddress)

            # DDPLL_RxX = ph.PLL(FOcompen_X)
            # PLL_BW = ph.bandwidth
            # if isplot == True: Histogram2D('KENG_FreqOffset_X', DDPLL_RxX[0, :], Imageaddress)

            DDPLL_RxX = FOcompen_X
            phasenoise_RxX = np.reshape(DDPLL_RxX, -1)
            PN_RxX = ph.QAM_64QAM_1(phasenoise_RxX, r1_o=1.55, r3_i=3.1, r3_o=3.47, r9_i=7.96)
            # PN_RxX = ph.QAM_64QAM_1(phasenoise_RxX, r1_o=2.1, r3_i=19, r3_o=19.1, r9_i=19.2)
            PN_RxX = PN_RxX[PN_RxX != 0]
            if isplot == True: Histogram2D('KENG_PhaseNoise_X', PN_RxX, Imageaddress)

            Rx_RA_X = ph.Rotation_algorithm(PN_RxX)
            if isplot == True: Histogram2D('KENG_PhaseNoise_X_RAstage', Rx_RA_X, Imageaddress)
            PN_RxX = Rx_RA_X

            Normal_ph_RxX_real, Normal_ph_RxX_imag = DataNormalize(np.real(PN_RxX), np.imag(PN_RxX), parameter.pamorder)
            Normal_ph_RxX = Normal_ph_RxX_real + 1j * Normal_ph_RxX_imag
            # if isplot == True: Histogram2D('KENG_PLL_Normalized_X', Normal_ph_RxX, Imageaddress)

            Correlation = KENG_corr(window_length=window_length)
            TxX_real, RxX_real, p = Correlation.corr_ex(TxXI, Normal_ph_RxX[0:correlation_length], 13); XIshift = Correlation.shift; XI_corr = Correlation.corr; XI_vec = Correlation.cal_vec;
            Correlation = KENG_corr(window_length=window_length)
            TxX_imag, RxX_imag, p = Correlation.corr_ex(TxXQ, Normal_ph_RxX[0:correlation_length], 13); XQshift = Correlation.shift; XQ_corr = Correlation.corr; XQ_vec = Correlation.cal_vec;
            RxX_corr = RxX_real[0:final_length] + 1j * RxX_imag[0:final_length]
            TxX_corr = TxX_real[0:final_length] + 1j * TxX_imag[0:final_length]
            # if isplot == True: Histogram2D('KENG_Corr_X', RxX_corr, Imageaddress)

            SNR_X, EVM_X = SNR(RxX_corr, TxX_corr)
            bercount_X = BERcount(np.array(TxX_corr), np.array(RxX_corr), parameter.pamorder)
            print('BERcount_X = {} \nSNR_X = {} \nEVM_X = {}'.format(bercount_X, SNR_X, EVM_X))
            # XSNR[eyepos] ,XEVM[eyepos] = SNR_X, EVM_X
            if isplot == True: Histogram2D('KENG_X_beforeVol', RxX_corr, Imageaddress, SNR_X, EVM_X, bercount_X)

        # --------------------------- Y PART------------------------
        if ypart == True:
            print('================================')
            print('Y part')
            ph = KENG_phaserecovery()
            FOcompen_Y = ph.FreqOffsetComp(Rx_Y_CMA, fsamp=56e9, fres=1e5)
            if isplot == True: Histogram2D('KENG_FOcompensate_Y', FOcompen_Y, Imageaddress)

            DDPLL_RxY = ph.PLL(FOcompen_Y)
            PLL_BW = ph.bandwidth
            if isplot == True: Histogram2D('KENG_FreqOffset_Y', DDPLL_RxY[0, :], Imageaddress)
            #
            phasenoise_RxY = np.reshape(DDPLL_RxY, -1)
            PN_RxY = ph.QAM_64QAM_1(phasenoise_RxY, r1_o=1.55, r3_i=3.1, r3_o=3.47, r9_i=7.96)
            PN_RxY = PN_RxY[PN_RxY != 0]
            if isplot == True: Histogram2D('KENG_PhaseNoise_Y', PN_RxY, Imageaddress)

            Rx_RA_Y = ph.Rotation_algorithm(PN_RxY)
            if isplot == True: Histogram2D('KENG_PhaseNoise_Y_RAstage', Rx_RA_Y, Imageaddress)
            PN_RxY = Rx_RA_Y

            Normal_ph_RxY_real, Normal_ph_RxY_imag = DataNormalize(np.real(PN_RxY), np.imag(PN_RxY), parameter.pamorder)
            Normal_ph_RxY = Normal_ph_RxY_real + 1j * Normal_ph_RxY_imag
            # if isplot == True: Histogram2D('KENG_PLL_Normalized_Y', Normal_ph_RxY, Imageaddress)

            Correlation = KENG_corr(window_length=window_length)
            TxY_real, RxY_real, p = Correlation.corr_ex(TxYI, Normal_ph_RxY[0:correlation_length], 13); YIshift = Correlation.shift; YI_corr = Correlation.corr; YI_vec = Correlation.cal_vec;
            Correlation = KENG_corr(window_length=window_length)
            TxY_imag, RxY_imag, p = Correlation.corr_ex(TxYQ, Normal_ph_RxY[0:correlation_length], 13); YQshift = Correlation.shift; YQ_corr = Correlation.corr; YQ_vec = Correlation.cal_vec;
            RxY_corr = RxY_real[0:final_length] + 1j * RxY_imag[0:final_length]
            TxY_corr = TxY_real[0:final_length] + 1j * TxY_imag[0:final_length]
            # if isplot == True: Histogram2D('KENG_Corr_Y', RxY_corr, Imageaddress)

            SNR_Y, EVM_Y = SNR(RxY_corr, TxY_corr)
            bercount_Y = BERcount(np.array(TxY_corr), np.array(RxY_corr), parameter.pamorder)
            print('BER_Y = {} \nSNR_Y = {} \nEVM_Y = {}'.format(bercount_Y, SNR_Y, EVM_Y))
            # YSNR[eyepos], YEVM[eyepos] = SNR_Y, EVM_Y
            if isplot == True: Histogram2D('KENG_Y_beforeVol', RxY_corr, Imageaddress, SNR_Y, EVM_Y, bercount_Y)

        # if iswrite == True:
        #     print('----------------write excel----------------')
        #     parameter_record = [eyepos,
        #                         str([cma.mean, cma.type, cma.overhead, cma.earlystop, cma.stepsizeadjust]), \
        #                         str([CMAstage1_tap, CMAstage1_stepsize_x, CMAstage1_stepsize_x, CMAstage1_iteration, [CMA_cost_X1, CMA_cost_Y1]]), \
        #                         str([CMAstage2_tap, CMAstage2_stepsize_x, CMAstage2_stepsize_x, CMAstage2_iteration, [CMA_cost_X2, CMA_cost_Y2]]), \
        #                         str([CMAstage3_tap, CMAstage3_stepsize_x, CMAstage3_stepsize_x, CMAstage3_iteration, [CMA_cost_X3, CMA_cost_Y3]]), \
        #                         PLL_BW, str([1.55, 3.1, 3.47, 7.96]), \
        #                         str([(XIshift, XQshift), (XI_corr, XQ_corr)]), str([SNR_X, EVM_X, bercount_X]), \
        #                         str([(YIshift, YQshift), (YI_corr, YQ_corr)]), str([SNR_Y, EVM_Y, bercount_Y]), \
        #                         str([(XI_vec, XQ_vec), (YI_vec, YQ_vec)])]
        #
        #     write_excel(address, parameter_record)

    # Correlation = KENG_corr(window_length=7000)
    # Rx_real, Tx_real = Correlation.calculate_Rx(np.real(Normal_ph_RxX), -TxXI[0:Normal_ph_RxX.size])
    # Rx_imag, Tx_imag = Correlation.calculate_Rx(-np.imag(Normal_ph_Rx[10000:50000]), TxXQ[10000:50000])
    # Rx_corr = np.array(Rx_real[0:40000].T) + 1j * np.array(Rx_imag[0:40000].T)
    # Tx_corr = np.array(Tx_real[0:40000].T) + 1j * np.array(Tx_imag[0:40000].T)
    # Rx_corr = np.reshape(Rx_corr, (-1))
    # Tx_corr = np.reshape(Tx_corr, (-1))
    # Histogram2D('KENG_Corr', Rx_corr)
    #
    # np.save(address + 'XSNR', XSNR)
    # np.save(address + 'YSNR', YSNR)
# ===========================================volterra=========================================================
if isrealvolterra == 1:
    equalizer_real = Equalizer(np.real(np.array(TxX_corr)), np.real(np.array(RxX_corr)), 3, [31, 31, 31], 0.2)
    equalizer_imag = Equalizer(np.imag(np.array(TxX_corr)), np.imag(np.array(RxX_corr)), 3, [31, 31, 31], 0.2)
    Tx_volterra_real, Rx_volterra_real = equalizer_real.realvolterra()
    Tx_volterra_imag, Rx_volterra_imag = equalizer_imag.realvolterra()
    Tx_real_volterra = Tx_volterra_real + 1j * Tx_volterra_imag
    Rx_real_volterra = Rx_volterra_real + 1j * Rx_volterra_imag

    snr_realvolterra, evm_realvolterra = SNR(Tx_real_volterra, Rx_real_volterra)
    realvol_bercount = BERcount(Tx_real_volterra, Rx_real_volterra, parameter.pamorder)
    print("BERcount_realvol : {}".format(realvol_bercount))
    print("SNR_realvol : {}, EVM_realvol : {}".format(snr_realvolterra, evm_realvolterra))
    Histogram2D("RealVolterra", Rx_real_volterra, Imageaddress, snr_realvolterra, evm_realvolterra, realvol_bercount)

# ------
if iscomplexvolterra == 1:
    equalizer_complex = Equalizer(np.array(TxY_corr), np.array(RxY_corr), 3, [31, 25, 25], 0.1)
    Tx_complex_volterra, Rx_complex_volterra = equalizer_complex.complexvolterra()

    snr_complexvolterra, evm_complexvolterra = SNR(Tx_complex_volterra, Rx_complex_volterra)
    print("SNR_cmplvol : {}, EVM_cmplvol : {}".format(snr_complexvolterra, evm_complexvolterra))
    # Histogram2D("complexVolterra", Rx_complex_volterra, Imageaddress, snr_complexvolterra, evm_complexvolterra)

    compvol_bercount = BERcount(Tx_complex_volterra, Rx_complex_volterra, parameter.pamorder)
    print("BERcount_cmplvol : {}".format(compvol_bercount))

# # #-----
# # vol=KENG_volterra(Rx_corr,Tx_corr,[21,1,3],25000)
# # vol_Rx=vol.first_third_order()

# kengvol_snr,kengvol_evm=SNR(vol.Rx_vol_test[0:5000,0], vol.Tx_vol_test[0:5000,0])
# print(kengvol_snr,kengvol_evm)
# Histogram2D('KENGvol',vol.Rx_vol_test[0:25000,0],kengvol_snr,kengvol_evm)


# Histogram2D_tseng("complexVolterra_tseng", Rx_complex_volterra)
# Histogram2D_tseng("CMA_tseng", Rx_X_CMA)
# Histogram2D_tseng("Focomoen_tseng", FOcompen_Y)
# Histogram2D_tseng("V-V_tseng", PN_RxX)
# Histogram2D_tseng("Rx_tseng", Rx_Signal_Y)




# NN equalizer
    # parameter
    # taps = 31
    # batch_size = 500
    # LR = 1e-3
    # EPOCH = 300
    # overhead = 0.5
    # trainnum = int(len(RxX_corr) * overhead)
    # device = torch.device("cuda:0")
    # train_inputr = rolling_window(torch.Tensor(RxX_corr.real), taps)[:trainnum]
    # train_inputi = rolling_window(torch.Tensor(RxX_corr.imag), taps)[:trainnum]
    # train_targetr = torch.Tensor(TxX_corr.real[taps // 2:-taps // 2 + 1])[:trainnum]
    # train_targeti = torch.Tensor(TxX_corr.imag[taps // 2:-taps // 2 + 1])[:trainnum]
    # train_tensor = Data.TensorDataset(train_inputr, train_inputi, train_targetr, train_targeti)
    # train_loader = Data.DataLoader(train_tensor, batch_size=batch_size, shuffle=False)
    #
    # val_inputr = rolling_window(torch.Tensor(RxX_corr.real), taps)
    # val_inputi = rolling_window(torch.Tensor(RxX_corr.imag), taps)
    # val_targetr = torch.Tensor(TxX_corr.real[taps // 2:-taps // 2 + 1])
    # val_targeti = torch.Tensor(TxX_corr.imag[taps // 2:-taps // 2 + 1])
    # val_tensor = Data.TensorDataset(val_inputr, val_inputi, val_targetr, val_targeti)
    # val_loader = Data.DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
    #
    # layer_widths = [16, 16, 1]
    # layer_centres = [16, 16]
    # basis_func = gaussian
    # final_modelr, final_modeli = Network(layer_widths, layer_centres, basis_func, taps).to(device), Network(
    #     layer_widths, layer_centres, basis_func, taps).to(device)
    # final_optr = torch.optim.Adam(final_modelr.parameters(), lr=LR)
    # final_opti = torch.optim.Adam(final_modeli.parameters(), lr=LR)
    # # modelx = conv1dResNet(Residual_Block, [2, 2, 2, 2]).to(device)
    # # lossxr = nn.MSELoss()
    # # lossxi = nn.MSELoss()
    # # lossxc = nn.CrossEntropyLoss()
    # # opty = torch.optim.Adam(modely.parameters(), weight_decay=1e-2, lr=LR)
    # L = []
    # val_L = []
    # for epoch in tqdm(range(EPOCH)):
    #     for i, (dr, di, txr, txi) in enumerate(train_loader):
    #         final_modelr.train()
    #         final_modeli.train()
    #         outr, outi = final_modelr(dr.to(device)), final_modeli(di.to(device))
    #         # outr,outi= final_modelr(dr.unsqueeze(1).to(device)),final_modeli(di.unsqueeze(1).to(device))
    #         # trr,tri = harddecision(txr,txi)
    #         Lossr = nn.MSELoss()(outr.squeeze().cpu(), torch.Tensor(txr))
    #         Lossi = nn.MSELoss()(outi.squeeze().cpu(), torch.Tensor(txi))
    #         Loss = Lossr + Lossi
    #         final_optr.zero_grad()
    #         Lossr.backward(retain_graph=True)
    #         final_optr.step()
    #         final_opti.zero_grad()
    #         Lossi.backward()
    #         final_opti.step()
    #         L.append(Loss.detach().cpu().numpy())
    #         # modely.eval()
    #     print("Train Loss:{:.3f}".format(Loss),
    #           '||' "Train Bercount:{:.3E}".format(BERcount(txr + 1j * txi, outr.cpu() + 1j * outi.cpu(), parameter.pamorder)))
    #     # print('\n training Accx: %f\n training Accy: %f\n' % (np.mean(Accx), np.mean(Accy)))
    #     # Accx = []
    #     # # Accy = []
    #     predictr, predicti = [], []
    # final_modelr.eval()
    # final_modeli.eval()
    # for i, (dr, di, txr, txi) in enumerate(val_loader):
    #     outr, outi = final_modelr(dr.to(device)), final_modeli(di.to(device))
    #     predictr.extend(outr.cpu().detach().numpy())
    #     predicti.extend(outi.cpu().detach().numpy())
    #     # Lossxr = lossxr(outr.cpu(), txr)
    #     # Lossxi = lossxi(outi.cpu(), txi)
    #     # Lossyr = lossyr(outy[:, 0].cpu(), tyr)
    #     # Lossyi = lossyi(outy[:, 1].cpu(), tyi)
    #     # Lossxc = lossxc(outcx.cpu(), tgx)
    #     # Lossyc = lossyc(outcy.cpu(), tgy)
    #     # xacc = (tgx.eq(torch.max(outcx.cpu(), 1)[1])).sum() / outcx.shape[0]
    #     # yacc = (tgy.eq(torch.max(outcy.cpu(), 1)[1])).sum() / outcy.shape[0]
    #     print("Val BERcount:{:.3E}".format(BERcount(txr + 1j * txi, outr.cpu() + 1j * outi.cpu(), 4)))
    # predictr = np.array(predictr).squeeze()
    # predicti = np.array(predicti).squeeze()
    # # snr_RBF, evm_RBF = SNR(TxX_corr[taps // 2:-taps // 2 + 1], (np.array(predictr) + 1j * np.array(predicti)).squeeze())
    # bercount = BERcount(TxX_corr[taps // 2:-taps // 2 + 1], (np.array(predictr) + 1j * np.array(predicti)).squeeze(), parameter.pamorder)
    # # print(snr_RBF, evm_RBF)
    # # Histogram2D("RBF-Net", (np.array(predictr) + 1j * np.array(predicti)).squeeze(), Imageaddress, snr_RBF, evm_RBF)
    # Histogram2D("RBF-Net", (np.array(predictr) + 1j * np.array(predicti)).squeeze(), Imageaddress)



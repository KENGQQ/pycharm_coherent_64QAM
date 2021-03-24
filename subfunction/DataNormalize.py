import numpy as np
import math

def DataNormalize(seqr , seqi, PAM_order):

    if len(seqi) == 0:
        seqr = np.array(seqr)
        rmean = np.mean(seqr)
        rlen = len(seqr)
        normalseqr = []
        r_shift = seqr - rmean
        factor_amplitude_r = np.mean(abs(r_shift))*2/PAM_order
        normalseq = r_shift / factor_amplitude_r
        return normalseq
    else :
        seqr = np.array(seqr)
        seqi = np.array(seqi)
        rmean = np.mean(seqr)
        imean = np.mean(seqi)
        rlen = len(seqr)
        ilen = len(seqi)
        assert rlen == ilen, 'Length of I & Q should equal'
        normalseqr, normalseqi = [], []
        r_shift = seqr - rmean
        i_shift = seqi - imean
        factor_amplitude_r = np.mean(abs(r_shift)) * 2 / PAM_order
        factor_amplitude_i = np.mean(abs(i_shift)) * 2 / PAM_order
        # factor_amplitude_r = np.mean(abs(r_shift)) / math.log(PAM_order,2)
        # factor_amplitude_i = np.mean(abs(i_shift)) / math.log(PAM_order,2)
        normalseqr = r_shift / factor_amplitude_r
        normalseqi = i_shift / factor_amplitude_i
        return normalseqr, normalseqi
        


import numpy as np
import math
def SNR2BER(SNRdB,data_format):
    SNR_ratio = 10**(SNRdB/10)
    un = np.sqrt(1.5*SNR_ratio/(data_format-1))
    BER=2*(1-data_format**(-0.5))*(math.erfc(un)+math.erfc(3*un))/math.log2(data_format)
    return BER


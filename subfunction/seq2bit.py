import numpy as np
from tensorflow.keras.utils import to_categorical
#from keras.utils import to_categorical
def seq2bit(seqr,seqi,PAM_order):
    if PAM_order == 2:
        r0 = 0#np.mean(seqr)
        i0 = 0#np.mean(seqi)
        bitseq = [] #象限(3,4,1,2)
        for r, i in zip(seqr, seqi):
            if r>r0 and i>i0:
                bitseq.append(2)
            elif r>r0 and i<i0:
                bitseq.append(3)
            elif r<r0 and i>i0:
                bitseq.append(1)
            elif r<r0 and i<i0:
                bitseq.append(0)
            else:
                print('unsolve')
        return to_categorical(bitseq, num_classes=4)
    elif PAM_order == 4:
        d2  = 2
        d0  = 0
        dm2 = -2
     
        bitseq = []
        for r, i in zip(seqr, seqi):
            if i<dm2:
                if r<dm2:
                    bitseq.append(0)
                elif r>dm2 and r<d0:
                    bitseq.append(1)
                elif r>d0 and r<d2:
                    bitseq.append(2)
                elif r>d2:
                    bitseq.append(3)
            elif i>dm2 and i<d0:
                if r<dm2:
                    bitseq.append(4)
                elif r>dm2 and r<d0:
                    bitseq.append(5)
                elif r>d0 and r<d2:
                    bitseq.append(6)
                elif r>d2:
                    bitseq.append(7)
            elif i>d0 and i<d2:
                if r<dm2:
                    bitseq.append(8)
                elif r>dm2 and r<d0:
                    bitseq.append(9)
                elif r>d0 and r<d2:
                    bitseq.append(10)
                elif r>d2:
                    bitseq.append(11)
            elif i>d2:
                if r<dm2:
                    bitseq.append(12)
                elif r>dm2 and r<d0:
                    bitseq.append(13)
                elif r>d0 and r<d2:
                    bitseq.append(14)
                elif r>d2:
                    bitseq.append(15)
            else:
                bitseq.append(-1)
                print('unsolve')
        return to_categorical(bitseq, num_classes=16)
    else:
        assert False, 'Wrong PAM_order'
    
    
    
    
    
    
    
    
    
    
    
    
    
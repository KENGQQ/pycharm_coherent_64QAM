import numpy as np

def bit2seq(bit,PAM_order):
    seq = []
    if PAM_order == 2:    
        for b in bit:
            if b == 0:
                seq.append([-1,-1])
            elif b == 1:
                seq.append([-1,1])
            elif b ==2:
                seq.append([1,1])
            else:
                seq.append([1,-1])
    elif PAM_order == 4:
            for b in bit:
                if b == 0:
                    seq.append([-3,-3])
                elif b ==1:
                    seq.append([-1,-3])
                elif b ==2:
                    seq.append([1,-3])
                elif b ==3:
                    seq.append([3,-3])
                elif b ==4:
                    seq.append([-3,-1])
                elif b ==5:
                    seq.append([-1,-1])
                elif b ==6:
                    seq.append([1,-1])
                elif b ==7:
                    seq.append([3,-1])
                elif b ==8:
                    seq.append([-3,1])
                elif b ==9:
                    seq.append([-1,1])
                elif b ==10:
                    seq.append([1,1])  
                elif b ==11:
                    seq.append([3,1])
                elif b ==12:
                    seq.append([-3,3])    
                elif b ==13:
                    seq.append([-1,3])  
                elif b ==14:
                    seq.append([1,3])
                elif b ==15:
                    seq.append([3,3])  
    else:
        assert False, 'Wrong PAM_order'
    return np.array(seq)
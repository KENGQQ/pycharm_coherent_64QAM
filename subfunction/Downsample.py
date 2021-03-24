import numpy as np

def Downsample(isig, downsample, downnum=0):

    isig = np.array(isig)
    if np.shape(isig) == (len(isig), ):
        rx_downsample = isig[downnum::downsample]
        return rx_downsample
    else:
        rx_downsample = isig[:, downnum::downsample]
        return rx_downsample


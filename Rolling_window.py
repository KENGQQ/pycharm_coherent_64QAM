def rolling_window(x, window_size, step_size=1):
    # unfold dimension to make our rolling window
    tmp = x
    return tmp.unfold(0,window_size,step_size)
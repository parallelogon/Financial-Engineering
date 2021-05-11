import numpy as np

# option functions, all take a stock path and return a price
def S(St):
    return(St)

def DOBC(St, K, H):
    if np.min(St[-1]) <= H:
        return 0
    else:
        return(max(0, St[-1] -K))

def DIBC(St, K, H):
    if np.min(St) > H:
        return 0
    else:
        return(max(0, St[-1] - K))

def DIBP(St, K, H):
    if np.min(St) > H:
        return 0
    else:
        return(max(0, K - St[-1]))

def DOBP(St, K, H):
    if np.min(St) <= H:
        return 0
    else:
        return(max(0, K - St[-1]))

def EC(St, K):
    return(np.maximum(0.0, St[-1] - K))

def EP(St, K):
    return(np.maximum(0.0, K - St[-1]))
import numpy as np
def hestfftcf(u, p, r, t, params):
    v0, kappa, eta, lbda, rho = params
    #Here they are of the form [p], in bsfftcf2 they are of the form [[p]]
    #might be a [[p]]roblem
    q = 0
    d = ((rho*theta*u*1j - kappa)**2 - theta**2(-1j*u -u^2))**0.5
    g = (kappa - 1j*rho*theta*u - d)/(kappa - 1j*rho*theta*u + d)


    step1 = 1j*u*np.log(p) + (r-q)*t
    step2 = eta*kappa*(theta**-2)*((kappa - 1j*u*rho*theta - d)*t - 2*np.log((1 - g*np.exp(-d*t))/(1-g)))
    step3 = sig**2 * theta**-2 * (kappa - 1j*u*rho*theta - d)*(1 - np.exp(-d*t))/(1-g*np.exp(-d*t))

    return(np.exp(step1 + step2 + step3))


def bsfftcf2(u,p,r,t,x):
    sig = x
    print(x)
    y = np.exp(1j * u * (np.log(p) + r * t - 1/2 * sig**2*t))*np.exp(-1/2*sig**2*u**2*t)
    return(y)
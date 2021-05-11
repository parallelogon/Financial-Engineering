import numpy as np
from scipy import interpolate
import torch
from scipy.fftpack import fft

#TODO: 
# loss functions which are more robust at middle easy - later during assignment

"""
Model class:
    The idea of this class is to define an arbitrary characteristic function
    (many of which should be implemented), and be able to fit it with an
    arbitrary loss function.  It will need to keep track of model parameters
    and weights

    To fit the model, fit with a list of strikes, prices, and times
"""


# Black-Scholes-Merton Characteristic Function for pricing in the carr-madan
# formula
def bsfftcf2(u,p,r,t,x):
    sig = list(map(float,x))[0]
    mu = np.log(p) + (r - 0.5*sig**2)*t
    var = t*sig**2

    y = np.exp(1j*u*mu - 0.5*u**2*var)
    return(y)

# Heston CF for carr-madan has a stochastic volatility term
def hestfftcf(u, p, r, t, params):
    sig, kappa, eta, theta, rho = list(map(float,params))
    kappa = abs(kappa)
    eta = abs(eta)
    theta = abs(theta)
    rho = min(max(-1,rho),1)

    q = 0
    d = ((rho*theta*u*1j - kappa)**2 - (theta**2)*(-1j*u - u**2))**0.5
    g = (kappa - 1j*rho*theta*u - d)/(kappa - 1j*rho*theta*u + d)


    if theta == 0.0:
        print("Problem val", sig, kappa, eta, theta, rho)

    step1 = 1j*u*np.log(p) + (r-q)*t
    step2 = eta*kappa*(theta**-2)*((kappa - 1j*u*rho*theta - d)*t - 2*np.log((1 - g*np.exp(-d*t))/(1-g)))
    step3 = sig**2 * theta**-2 * (kappa - 1j*u*rho*theta - d)*(1 - np.exp(-d*t))/(1-g*np.exp(-d*t))

    return(np.exp(step1 + step2 + step3))


# Bates CF Heston with Jumps
def batesfftcf(u, p, r, t, params):
    eta, kappa, theta, rho, sig, lbda, muJ, sigJ = list(map(float,params))

    kappa = abs(kappa)
    eta = abs(eta)
    theta = abs(theta)
    rho = min(max(-1,rho),1)
    sigJ = abs(sigJ)
    sig = abs(sig)
    lbda = max(0, lbda)


    #d = ((rho*theta*u*1j - kappa)**2 - (theta**2)*(-1j*u - u**2))**0.5
    d = ((rho*theta*u*1j - kappa)**2 + (theta**2)*(1j*u + u**2))**0.5

    g = (kappa - 1j*rho*theta*u - d)/(kappa - 1j*rho*theta*u + d)

    q = 0

    step1 = 1j*u*(np.log(p) + (r - q)*t)

    step2 = eta*kappa*theta**(-2)*((kappa - rho*theta*u*1j - d)*t - \
        2*np.log((1 - g*np.exp(-d*t))/(1-g)))

    step3 = (sig**2)*(theta**(-2))*(kappa - rho*theta*1j*u - d)*(1 - np.exp(-d*t))/(1 - g*np.exp(-d*t))

    step4 = -lbda*muJ*1j*u*t + lbda*t*((1 + muJ)**(1j*u)*np.exp(sigJ**2 * (1j*u/2)*(1j*u - 1))-1)

    return(np.exp(step1 + step2 + step3 + step4))


"""
Loss functions, each one takes inputs yHat, y, and weights
"""
def MSE(yHat, y, weights):
    y = y.reshape(yHat.shape)
    return np.mean(weights*(yHat - y)**2)

def RMSE(yHat,y, weights):
    y = y.reshape(yHat.shape)
    return np.sqrt(np.mean(weights*(yHat - y)**2))

def L1(yHat,y, weights):
    y = y.reshape(yHat.shape)
    return(1/len(y)*np.linalg.norm((yHat - y)*weights, ord = 1))

def ARPE(yHat,y, weights):
    y = y.reshape(yHat.shape)
    return np.mean(np.abs(weights*(yHat - y))/y)


# central difference derivative calculation for use in minimization
def J(f, theta, h = 0.000001):
    I = np.identity(np.size(theta))
    f0 = f(theta)

    valType = type(f0)

    if (valType == np.ndarray) or (valType == list):
        J = np.empty((np.shape(f0)[0],np.size(theta)))
    else:
        J = np.empty((1,np.size(theta)))

    for i,direction in enumerate(I.T):
        direction = direction.reshape(-1,1)*h
        a  = f(theta + direction) - f(theta - direction)
        if np.size(a) == 1:
           J[:,i] = a/(2*h)
        else:
            J[:,i] = a.T/(2*h)

    return(J)

"""
Line search algorithms, first one is basic line search used with newton-rapheson
second for BFGS - 2nd order approximation of the hessian

"""
""" def lineSearch(f,x,d,fx,Jd, alpha = 1, sigma = .0001, beta = 0.5):
    a = alpha
    while (f(x + a*d) - fx > sigma*a*Jd):
        if a <= 1e-15:
            return(a)
        
        a = a*beta
    return(a)
 """

def lineSearch(f,x,d,fx, Jd, alpha = 1, a_low = 0, a_hi = 100000000, sigma = 10e-4, eta = 0.9):
    aMax = a_hi
    while(True):
        if (f(x + alpha*d) > fx + sigma*alpha*Jd):
            a_hi = alpha
            alpha = 0.5*(a_hi + a_low)
        elif ( J(f, x + alpha*d) @ d < eta*Jd):
            a_low = alpha
            if a_hi >= aMax:
                alpha = 2*a_low
            else:
                alpha = 0.5*(a_low + a_hi)
        else:
            break
    return(alpha)


# Pricing with carr-madan
def CallPrice(cf, s0, K, t, r, params):
    N = 4096

    alpha = 1.5
    eta = 0.25
    p = s0
    strikes = K
    q = 0

    lbda = 2*np.pi/N/eta
    b = np.pi/eta
    k = -b + lbda*np.arange(0,N)
    KK = np.exp(k)
    v = np.arange(0, N*eta, eta)
    
    pm_one = np.empty((N,))
    pm_one[::2] = -1
    pm_one[1::2] = 1
    sw = 3 + pm_one
    sw[0] = 1
    sw = sw/3

    rhoTop = cf(v - (alpha + 1)*1j, p, r, t, params)
    rhoBottom = alpha**2 + alpha - v**2 + 1j*(2*alpha + 1)*v
    rho = np.exp(-r*t)*rhoTop/rhoBottom


    A = rho*np.exp(1j*v*b)*sw*eta
    Z = np.real(fft(A))
    CallPricesBS = np.real(np.exp(-alpha*k)/np.pi * Z)

    tck = interpolate.splrep(KK, CallPricesBS)
    price = interpolate.splev(strikes,tck)

    return(price)


# Class object called using
# x = Model()
# x.cf(choice) choice = {'hest', 'bs', 'bates'}
# x.fit(strikes, times, prices, weights)
# x.predict(strikes, times) gives model predictions
# x.priceMC(function, maturity, n, M) gives a MC simulation of the function
#  function is formatted such that it takes only a stock price (use a lambda 
#  function)
class Model:
    def __init__(self):
        self.parameters = np.array([])
        self.weights = 1 
        self.r = 0
        self.s0 = 0


    def cf(self,choice):
        if choice =='bs':
            self.parameters = np.random.rand(1,1)
            self.cf = bsfftcf2

        elif choice == 'hest':
            self.parameters = np.random.rand(5,1)
            self.cf = hestfftcf

        elif choice == 'bates':
            self.parameters = np.random.rand(8,1)
            self.cf = batesfftcf

        elif callable(choice):
            self.cf = choice


    def objective(self,choice):
        if choice == 'RMSE':
            self.lossFn = RMSE
        elif choice == "MSE":
            self.lossFn = MSE
        elif choice == "L1":
            self.lossFn = L1
        elif choice == "ARPE":
            self.lossFn = ARPE
        else:
            print("No Loss Function, pick one")


    def price(self,strikes,times,params):
        prices = []
        for time in np.unique(times):

            if np.size(strikes) > 1:
                strike = strikes[np.where(times == time)]
            else:
                strike = strikes

            a = CallPrice(self.cf, self.s0, strike, time, self.r, params)
            a = a.reshape(-1)
            prices = prices + list(a)
        return(np.array(prices).reshape(-1,1))


    def predict(self,strikes,time):
        return(CallPrice(self.cf, self.s0, strikes, time, self.r, self.parameters).reshape(-1,1))

    def fit(self, strikes, times, prices, max_itrs = 30, tol = 1e-4, weights = []):
        f = lambda x: self.lossFn(self.price(strikes, times, x), prices, self.weights)

        if np.size(weights) > 0:
            if np.size(weights) == np.size(strikes):
                self.weights = weights
            else:
                print("Weights not same size as strikes")
                return(0)
        else:
            print("Unweighted")
            self.weights = np.ones((np.size(strikes), 1))

        # Set x vector to minimize
        x = self.parameters

        #Initialize B
        B = np.eye(np.size(x))

        #Evaluate initial gradient
        fT = J(f,x)
        grad = fT.T
        fx = f(x)

        for i in range(max_itrs):

            print("iteration: ",i)

            #print(x)

            gnorm = np.linalg.norm(grad, ord = np.inf)
            print("Grad: ", gnorm,"\n")
            if gnorm <= tol:
                print("Terminating at iteration: ",i,"\n")
                break
            # Find the search direction
            d = - np.linalg.inv(B) @ grad

            # line search for the step size
            Jd = fT @ d
            a = lineSearch(f, x, d, fx, Jd)

            #update x
            xOld = x
            x = x + a*d

            fx = f(x)
            print("Objective: ", fx, "\n")

            # # update BFTS hessian approximation
            gradOld = grad
            fT = J(f,x)
            grad = fT.T


            s = x - xOld
            y = grad - gradOld
            Bs = B @ s + 1e-8

            aa = bb = 0

            if np.linalg.norm(s, ord = np.inf) <= tol:
                print("Terminated at iteration: ", i)
                break

            elif np.linalg.norm(y, ord = np.inf) <- tol:
                print("Terminated at iteration: ", i)
                break

            aa = Bs @ (Bs.T / s.T @ Bs )
            bb = y @ (y.T / s.T @ y)

            B = B - aa + bb

        self.parameters = x
        print("Final parameter value: ", self.parameters)
        return(f(x))
    
    def priceMC(self, optionFunc, maturity, n = 365, M = 10000):
        dt = maturity/n

        # pricing under the black-scholes-merton model
        if self.cf == bsfftcf2:

            sig = self.parameters[0][0]
            r = self.r
            mu_dt = (r - sig**2/2)*dt
            var = sig*np.sqrt(dt)
            Z = np.random.randn(M,n)
            St = np.zeros((M,n))
            St[:,0] = self.s0

            for i in range(1,n):
                St[:,i] = St[:,i-1]*np.exp( mu_dt + var*Z[:,i])

            val = np.array(list(map(optionFunc, St)))
            expectation = np.mean(val)
            sd = np.std(val)/np.sqrt(M)

        # pricing under the heston model
        elif self.cf == hestfftcf:
            sig, kappa, eta, theta, rho = self.parameters
            
            r = self.r
            mu_dt = 1 + r*dt
            expectation = 0
            sq_expectation = 0

            St = np.zeros((M,n))
            Vt = np.zeros((M,n))

            St[:,0] = self.s0
            Vt[:,0] = sig**2

            e1 = np.random.randn(M,n)
            Zstar = np.random.randn(M,n)
            e2 = rho*e1 + np.sqrt(1-rho**2)*Zstar

            for i in range(1,n):
                St[:,i] = St[:,i-1]*(mu_dt + np.sqrt(Vt[:,i-1]*dt)*e1[:,i-1])
                Vt[:,i] = np.abs(Vt[:,i-1] + (kappa*(eta - Vt[:,i-1]) - theta**2/4)*dt + \
                    theta*np.sqrt(Vt[:,i-1]*dt)*e2[:,i-1] + ((theta**2)*dt*e2[:,i-1]**2)/4)

            val = np.array(list(map(optionFunc, St)))
            expectation = np.mean(val)
            sd = np.std(val)/np.sqrt(M)
            

        # pricing under bates
        elif self.cf == batesfftcf:
            eta, kappa, theta, rho, sig, lbda, muJ, sigJ = self.parameters



            r = self.r
            
            q = 0

            mu_dt = (r - q - lbda*muJ)*dt

            expectation = 0
            sq_expectation = 0

            St = np.zeros((M,n))
            St[:,0] = self.s0

            sig2t = np.zeros((M,n))
            sig2t[:,0] = sig**2

            dNt = np.random.poisson(lbda*dt, (M,n))
            NJ = sigJ*np.random.randn(M,n) + np.log(1 + muJ) - sigJ**2/2
            Jt = np.exp(NJ) - 1

            e1 = np.random.randn(M,n)
            Zstar = np.random.randn(M,n)
            e2 = rho*e1 + np.sqrt(1-rho**2)*Zstar

            for i in range(1,n):
                St[:,i] = St[:,i-1] + St[:,i-1] * (mu_dt + np.sqrt(sig2t[:,i-1]*dt)*e1[:,i-1]) + \
                    Jt[:,i-1]*dNt[:,i-1]

                sig2t[i] = np.abs(kappa*(eta - sig2t[:,i-1])*dt + theta*np.sqrt(sig2t[:,i-1]*dt)*e2[:,i-1])

            val = np.array(list(map(optionFunc, St)))
            expectation = np.mean(val)
            sd = np.std(val)/np.sqrt(M)          


        return(np.exp(-r*maturity)*expectation, np.exp(-r*maturity)*sd)
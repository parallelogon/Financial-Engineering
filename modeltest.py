#Simulated annealing for initial values and then MCMC

from yahoo_fin import options
import matplotlib.pyplot as plt
import yahoo_fin.stock_info as si
import pandas as pd
import datetime
import numpy as np
from datetime import date, timedelta, datetime
from scipy import interpolate
import torch 
from scipy import stats
import pso
import model as md 
from scipy.stats import norm
from optionfuncs import *

# Get stock information from FSLY
ticker = 'FSLY'
stock = si.get_data(ticker)
s0 =si.get_live_price(ticker)


"""
treasury_rate = si.get_live_price("^TNX")/100
if np.isnan(treasury_rate):
    r = 1.67/100
else:
    r = si.get_live_price("^TNX")/100


dates = options.get_expiration_dates(ticker)

today = date.today()

dfCalls = pd.DataFrame()

T = []



# get live information
for day in dates:

    print(day)

    try: 
        calls = options.get_calls(ticker,day)

    except:
        pass


    maturity = datetime.strptime(day,"%B %d, %Y" ).date()

    dt = (maturity - today).total_seconds()/60/60/24/365

    # out of the money calls
    tempdf = calls[["Last Price", "Strike"]][calls['Strike'] > s0]

    tempdf["Bid-Ask"] = -calls["Bid"] + calls["Ask"]

    tempdf["Maturity"] = dt

    T.append(dt)

    dfCalls = dfCalls.append(tempdf)
"""

# save data for may 6
dfCalls = pd.read_csv('dfCallsMay6')
dateAccess = datetime(2021,5,6).date()
s0 = si.get_data(ticker).loc[str(dateAccess)]['close']
r = si.get_data("^TNX").loc[str(dateAccess)]['close']/100

# normalize prices and strikes by the stock value
Prices = dfCalls['Last Price'].to_numpy()/s0
Strikes = dfCalls['Strike'].to_numpy()/s0
Times = dfCalls['Maturity'].to_numpy()

# Inverse bid-ask weighting
Weights = dfCalls['Bid-Ask'].to_numpy()/s0

# remove zeros from the dataset 
# (bid-ask can have zero values if timing is incorect)
# Important to only run this after market has opened
valid = np.where(Weights != 0)

if np.size(valid) > 0:
    Weights = Weights[valid]
    Strikes = Strikes[valid]
    Times = Times[valid]
    Prices = Prices[valid]

Weights = 1/Weights.reshape(-1,1)
Weights = Weights/np.linalg.norm(Weights)

# a new model object
M = md.Model()

# normalized stock and interest rate
M.s0 = s0/s0
M.r = r

# RMSE objective function
M.objective('RMSE')

# heston model
M.cf("hest")



"""
# lambda function for the loss to minimize in a particle swarm
funcMin = lambda x: M.lossFn(M.price(Strikes, Times, x), Prices, Weights)


initial=np.random.rand(5,1)   
# bounds of possibility for sig, kappa, eta, theta, rho 
initial[0] = np.random.rand()
initial[1] = np.maximum(0.001, np.random.rand()*10)
initial[2] = np.maximum(0.001, np.random.rand()*10)
initial[3] = np.random.rand()
initial[4] = -1 + 2*np.random.rand()
bounds=[(0.01,1),(0.01,10),(0.01,10),(0.01,1),(-1.0,1.0)]

# PSO for a good nonlinear fit
swarm = pso.PSO(funcMin,initial,bounds,num_particles=1000,maxiter=10)

# after PSO one can use a regular optimization routine to insure a local
# minima
M.parameters = np.array(swarm.best_g).reshape(-1,1)

M.fit(Strikes, Times, Prices, weights = Weights)
M.parameters[4] = np.minimum(1, np.maximum(-1, M.parameters[4]))
# this is done to insure that |rho| <= 1, as numerical inconsistincies can
# cause nontrivial problems
"""

# Use previously found parameter values
M.parameters = np.array([[ 0.84516848],[ 3.23138087],[ 0.57471116],[ 0.77461603],[-0.6163664 ]])


# graph the results as a model check
predictions = []
for t in np.unique(Times):
    stri = Strikes[np.where(Times == t)]
    c = np.random.rand(3,)
    plt.scatter(stri, M.predict(stri, t), marker = '+', color = c)

    pri = Prices[np.where(Times ==t)]
    #plt.scatter(stri, pri, color = 'none', edgecolors = 'purple')
    plt.scatter(stri, pri, color = 'none', edgecolors=c)
    
plt.show()



K = 1
M.s0 = 1
fDOBC = lambda x: DOBC(x, 1, .7)
fDIBC = lambda x: DIBC(x, 1, .3)

fPC = lambda x: np.array([EP(x, K), EC(x, K)])

fP = lambda x: EP(x, K)
fC = lambda x: EC(x, K)

# ndays is measured in days/10 so ndays = 36 is 360 days
ndays = 36


# Find the prices of down and in barrier put options given various maturities and
# barrier values graphically
"""
Hs = np.arange(0.5, 0.95, .05)
means = np.zeros((len(Hs), ndays))
sds = np.zeros((len(Hs), ndays))
for j in range(0,len(Hs)):
    H = Hs[j]
    fDIBP = lambda x: DIBP(x, K, H)
    for i in range(1,ndays+1):
        print(j,i)
        t = i*10/365
        # M is set too low here, increase value for a precise graph (reccomended
        # at 100000)
        a,b = M.priceMC(fDIBP, t, n = ndays+i, M = 10000)
        means[j, i-1] = a
        sds[j, i-1] = b

# Code to make a plot of the above
times = [i/ndays for i in range(1,ndays+1)]
fig, ax = plt.subplots()
for i in range(0, len(Hs)):
    label_i = str(np.round(Hs[i],2))
    ax.errorbar(times, means[i,:], sds[i,:], c=np.random.rand(3,), ecolor = 'gray', alpha = 0.8, label = label_i)

ax.legend(bbox_to_anchor=(1, 1), loc="upper right")
ax.set_xticks([round(t,1) for t in times])
#ax.set_xticklabels([i*10 for i in range(1,ndays+1)])
plt.show()
"""
H = 0.75
K = 1


# MC Simulation should have results approximately normal by the CLT, find 
# mean value of several simulations, also very costly reduce M for noisier
# approximation

"""
fDIBP = lambda x: DIBP(x, K, H)
M.priceMC(fDIBC, .5, 180, M = 10000)
E = []
V = []
for i in range(1000):
    x, s = M.priceMC(fDIBP, .5, 400, M = 10000)
    E.append(x)
    V.append(s)

plt.hist(E, bins = 20, density = True)
mu = np.mean(E)
sig = np.std(E)
x = np.arange(mu - 4.0*sig,mu + 4.0*sig, 0.001)
y = np.sqrt(1/(2*np.pi*sig**2))*np.exp(-1/2*(x - mu)**2/sig**2)
plt.plot(x,y)
plt.show()

# Arbitrary margin value
margin = 0.01
Coupon = mu + (1 - np.exp(-r/2)) - margin

# number of payments
npay = 6
rate = Coupon / sum(1/npay*np.exp(-r*1/2/npay * k) for k in range(1,npay+1))

"""
# use finite difference method to find value of down and in barrier call delta
# this is used for hedging.  
h = 0.00001
H = 0.75
fDIBCh = lambda x: DIBC(x + h, K, H)
fDIBCmh = lambda x: DIBC(x - h, K, H)
fDelta = lambda x: (DIBP(x + h, K, H) - DIBP(x - h, K, H))/(2*h)


"""
generates a graph of deltas by stock price.  expensive to do without much noise
stocks = [i/10 for i in range(1, 21, 1)]
Delta = 0
reps = 20
for j in range(reps):
    delta_S = np.zeros(len(stocks))
    for i in range(len(stocks)):
        M.s0 = stocks[i]
        # make sure value of M is set higher than 10000
        delta_S[i],sd = M.priceMC(fDelta, 0.5, n = 400, M = 20000)

    Delta += delta_S[i]


plt.plot(stocks,delta_S/reps)
"""

M.s0 = s0
K = s0



# selected value of barrier for final product
H = 0.75*s0
h = .001
fDIBP = lambda x: DIBP(x, K, H)
M.s0 = s0 + h
high, sdH = M.priceMC(fDIBP, 0.5, n = 1000, M = 20000)
M.s0 = s0 - h
low, sdL = M.priceMC(fDIBP, 0.5, n = 1000, M = 20000)


fDelta = lambda x: (DIBP(x + h, K, H) - DIBP(x - h, K, H))/(2*h)
delta, sdelta = M.priceMC(fDelta, 0.5, n = 1000, M = 10000)
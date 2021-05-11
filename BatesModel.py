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


ticker = 'FSLY'

stock = si.get_data(ticker)

s0 =si.get_live_price(ticker)

treasury_rate = si.get_live_price("^TNX")/100
if np.isnan(treasury_rate):
    r = 1.67/100
else:
    r = si.get_live_price("^TNX")/100

dates = options.get_expiration_dates(ticker)
today = date.today()
dfCalls = pd.DataFrame()
T = []

count = 0
for day in dates:
    print(day)
    try: 
        calls = options.get_calls(ticker,day)
    except:
        pass

for day in dates:
    print(day)
    try: 
        puts = options.get_puts(ticker, day)
    except:
        pass


    maturity = datetime.strptime(day,"%B %d, %Y" ).date()
    dt = (maturity - today).total_seconds()/60/60/24/365

    tdf = calls[['Last Price', 'Strike']]
    tempdf = calls[['Last Price', 'Strike']][calls['Strike'] > s0]
    tempdf["Bid-Ask"] = -calls["Bid"] + calls["Ask"]
    tempdf["Maturity"] = dt
    T.append(dt)
    dfCalls = dfCalls.append(tempdf)


Prices = dfCalls['Last Price'].to_numpy()/s0
Strikes = dfCalls['Strike'].to_numpy()/s0
Times = dfCalls['Maturity'].to_numpy()

Weights = dfCalls['Bid-Ask'].to_numpy()/s0

valid = np.where(Weights != 0)

Weights = Weights[valid]
Strikes = Strikes[valid]
Times = Times[valid]
Prices = Prices[valid]

Weights = 1/Weights.reshape(-1,1)
Weights = Weights/np.linalg.norm(Weights)
M.weights = Weights
#N.priceMC(f, 1)

M = md.Model()
M.cf('bates')
M.objective('RMSE')
M.s0 = s0/s0
M.r = r

funcMin = lambda x: M.lossFn(M.price(Strikes, Times, x), Prices, M.weights)


# order of the parameter array: eta, kappa, theta, rho, sig, lbda, muJ, sigJ
bound_eta = (0.001, 2)
bound_kappa = (0.001, 10)
bound_lbda = (0.001, 10)
bound_rho = (-1.0, 1.0)
bound_theta = (0.001, 10)
bound_muJ = (-1.0, 10.0)
bound_sig = (0.001, 10)
bound_sigJ = (0.001, 10)

initial = np.zeros((8,1))
initial[0] = np.maximum(0.001,np.random.rand()*2)
initial[1] = np.maximum(0.001, np.random.rand()*10)
initial[2] = np.maximum(0.001, np.random.rand()*10)
initial[3] = -1 + np.random.rand()*2
initial[4] = np.maximum(0.001, np.random.rand()*1)
initial[5] = -1 + np.random.rand()*11
initial[6] = np.maximum(0.001, np.random.rand()*1)
initial[7] = np.maximum(0.001, np.random.rand()*1)

print(initial)

bounds=[bound_eta, bound_kappa, bound_theta, bound_rho, bound_sig, bound_lbda, bound_muJ, bound_sigJ]
swarm = pso.PSO(funcMin,initial,bounds,num_particles=100,maxiter=100)
M.parameters = np.array(swarm.best_g).reshape(-1,1)

M.s0 = 1
M.fit(Strikes, Times, Prices,weights = M.weights)
predictions = []
for t in np.unique(Times):
    stri = Strikes[np.where(Times == t)]
    plt.scatter(stri, M.predict(stri, t), marker = '+')

est =  M.price(Strikes,Times, M.parameters)
plt.scatter(Strikes, Prices, color = 'none', edgecolors = 'purple')
plt.show()

def S(St):
    return(St[-1])

def EC(St, K):
    out = np.maximum(0, St[-1] - K)
    return(np.maximum(0, St[-1] - K))

def DOBC(St, K, H):
    if min(St) <= H:
        return 0
    else:
        return(np.maximum(0, St[-1] -K))

def DIBC(St, K, H):
    if min(St) > H:
        return 0
    else:
        return(np.maximum(0, St[-1] - K))

def EP(St, K):
    out = np.maximum(0, K - St[-1])
    return(out)


K = 1

fDOBC = lambda x: DOBC(x, K, 30)
fDIBC = lambda x: DIBC(x, K, 30)

fP = lambda x: EP(x, K)
fC = lambda x: EC(x, K)
fPC = lambda x: np.array([EP(x, K), EC(x, K)])

# with open("HestonParams.txt", "w") as fp:
#     for param in M.parameters:
#         print(param)
#         fp.write(str(param))

M.s0 = s0
M.priceMC(fPC, 3/12,n = 10, M = 100)
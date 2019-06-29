import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, optimize

def sir(u,parms,t):
    bet,gamm,iota,N,dt=parms
    S,I,R,Y=u
    lambd = bet*(I+iota)/N
    ifrac = 1.0 - math.exp(-lambd*dt)
    rfrac = 1.0 - math.exp(-gamm*dt)
    infection = np.random.binomial(S,ifrac)
    recovery = np.random.binomial(I,rfrac)
    return [S-infection,I+infection-recovery,R+recovery,Y+infection]

def simulate(tf,tl): 
    #tf = 2.5
    #tl = 11
    dt = tf/(tl-1)
    parms = [3.0, 0.5, 0.01, 100.0, dt]
    t = np.linspace(0,tf,tl)
    S = np.zeros(tl)
    I = np.zeros(tl)
    R = np.zeros(tl)
    Y = np.zeros(tl)
    u = [95,5,0,0]
    S[0],I[0],R[0],Y[0] = u
    for j in range(1,tl):
        u = sir(u,parms,t[j])
        S[j],I[j],R[j],Y[j] = u
    return {'t':t,'S':S,'I':I,'R':R,'Y':Y}

def sir_model(y, x, beta, gamma):
    S = -beta * y[0] * y[1] / N
    R = gamma * y[1]
    I = -(S + R)
    return S, I, R

def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:,1]



if __name__ == '__main__':
    tf=10 #final time in the simulation i.e. runs from 0-->tf
    tl=11 #number of data points in the 0-->tf interval
    numSims=20 #number of simulation runs generated
    
    
    sim_array = np.empty([numSims,tl])
    
    plt.style.use("ggplot")
    plt.subplot(141)
    
    for i in range(numSims):
        sir_out = pd.DataFrame(simulate(tf,tl))
        # 
        sim_array[i,:]=sir_out['I'].values
        t_array=sir_out['t'].values
        #sline = plt.plot("t","S","",data=sir_out,color="lightcoral",linewidth=2)
        iline = plt.plot("t","I","",data=sir_out,color="palegreen",linewidth=2)
        #rline = plt.plot("t","R","",data=sir_out,color="paleturquoise",linewidth=2)
        #iline = plt.plot(t_array,sim_array[i,:],color="palegreen",linewidth=2)
    
    #1st set of data for fitting
    xdata1=t_array[0:tl-6]
    ydata1=sim_array[1,0:tl-6]
    #2nd set of data for fitting
    xdata2=t_array[0:tl-4]
    ydata2=sim_array[1,0:tl-4]
    #3rd set of data for fitting
    xdata3=t_array[0:tl-2]
    ydata3=sim_array[1,0:tl-2]
    
    
    N = 100.0
    I0 = ydata1[0]
    S0 = N - I0
    R0 = 0.0
    
    popt1, pcov1 = optimize.curve_fit(fit_odeint, xdata1, ydata1)
    popt2, pcov2 = optimize.curve_fit(fit_odeint, xdata2, ydata2)
    popt3, pcov3 = optimize.curve_fit(fit_odeint, xdata3, ydata3)
    #Confidence intervals and mid-point model fit
    
    # Plotting the full epidemic curve even if the amount of data is < the full curve
    xModel=np.linspace(0,tf,tl)
    fitted1 = fit_odeint(xModel, *popt1)
    fitted2 = fit_odeint(xModel, *popt2)
    fitted3 = fit_odeint(xModel, *popt3)
    #upper confidence interval
    #poptUpper= popt + 50*np.array([math.sqrt(pcov[0,0]), math.sqrt(pcov[1,1])])
    #fitted2 = fit_odeint(xdata, *poptUpper)
    ##lower confidence interval 
    #poptLower= popt - 5*np.array([math.sqrt(pcov[0,0]), math.sqrt(pcov[1,1])])
    #fitted3 = fit_odeint(xdata, *poptLower)
    
    #print(sir_out)    
    print(popt1)
    #plt.figure()
    
    a1=np.transpose(sim_array)
    
    plt.subplot(142)
    plt.plot(t_array,a1,color="palegreen",linewidth=2)
    plt.plot(xModel, fitted1)
    plt.plot(xdata1, ydata1, 'o')
    
    plt.subplot(143)
    plt.plot(t_array,a1,color="palegreen",linewidth=2)
    plt.plot(xModel, fitted2)
    plt.plot(xdata2, ydata2, 'o')
    
    plt.subplot(144)
    plt.plot(t_array,a1,color="paleblue",linewidth=2)
    plt.plot(xModel, fitted3)
    plt.plot(xdata3, ydata3, 'o')
    
    plt.xlabel("Time",fontweight="bold")
    plt.ylabel("Number",fontweight="bold")
    #legend = plt.legend(title="Population",loc=4,bbox_to_anchor=(1.25,0.5))
    #plt.xticks(range(0,xmax))
    #plt.plot(xdata, fitted2)
    #plt.plot(xdata, fitted3)
    plt.show()
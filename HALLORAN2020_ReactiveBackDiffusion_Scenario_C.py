# -*- coding: utf-8 -*-
"""
HALLORAN2020_ReactiveBackDiffusion_Scenario_C.py
Landon Halloran, 2020
www.ljsh.ca 
github.com/lhalloran

Python script to process and analyse model output for Scenario C in Halloran & Hunkeler (2020) paper.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
#%% 
#################################### TO BE DEFINED BY USER #######################################
cutOffC = 0.00001    # Cut-off normalised concentration (should be >=1E-5).
pointNumber = 4      # Observation well for analysis. Above aquitard starts at 0, moving L to R (i.e., well at 100m is # 4)
##################################################################################################
#%% DEFINE CUSTOM FUNCTIONS
# text colour chooser
def choose_colour(val,maxval):
    if np.isnan(val):
        colour = 'black'
    elif val>maxval/2:
        colour = 'black'
    else: 
        colour = 'white'
    return colour

# t,x to unitless spacetime (n pore volumes)
def tx_to_unitless(v,t,x):
    return v*t/x

# convert to velocity
def to_velocity(K,epsilon,dHdx):
    return (1/epsilon)*K*dHdx

#%%

fileName = 'ScenarioC.csv'
theFontSize=14
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

nt = 66 # number of time steps
nPoints = 5
nParams = 6
nParamsCombo = 288
dHdx = 0.01 # horizontal hydraulic gradient 
epsilonAquitard = 0.4 # porosity in aquitard!
epsilonAquifer = 0.35 # porosity in aquifer!

dataIn = pd.read_csv(fileName, header=4)
data = np.empty((nParamsCombo, nt, nPoints))
params = np.empty((nParamsCombo, nParams))
paramNames = np.array(dataIn.columns.values[1:1+nParams])
t = dataIn.values[0:nt,0]

# convert to numpy matrix...
for i in np.arange(nParamsCombo):
    startInd = i*nt
    data[i,:,:] = dataIn.values[startInd:startInd+nt, nParams+1:]
    params[i,:] = dataIn.values[startInd, 1:1+nParams]

#%% some general plotting parameters
thefigsize=10,10

#%% initial plots of data (note: these are not exported to pdf)
xloc=np.arange(0,101,20)

plt.figure()
for i in np.arange(nParamsCombo):
    clrnow=(0.0,i/nParamsCombo,1-i/nParamsCombo,0.25)
    plt.plot(xloc,np.append(np.array(1),data[i,-1,:]),c=clrnow)
plt.title('Concentration at 1000 years')
plt.xlabel('distance from source (m)')

plt.figure()
for i in np.arange(nParamsCombo):
    clrnow=(0.0,i/nParamsCombo,1-i/nParamsCombo,0.25)
    plt.plot(t,data[i,:,4],c=clrnow)
plt.title('Concentration at 100 m')
plt.xlabel('time (years)')
#%% determine spatial dependence at equilibrium (t = 1000 years)
def expdec(x, b):
    return np.exp(-x/b)
# Exponential decay parameter b (1 value per model run):
equilExpParam=np.zeros((nParamsCombo))-1.0
for i in np.arange(nParamsCombo):
    datanow = np.append(np.array(1),data[i,-1,:])
    pOpt, pCov = curve_fit(expdec, xloc, datanow, p0=(20))
    equilExpParam[i] = pOpt
#%% As data time-step is not uniform, plot 

def delayedExpRise(t, a, b, t0):
    return np.max([0,a*(1 - np.exp(-(t-t0)/b))])
delayedExpRise_vec = np.vectorize(delayedExpRise) #vectorize so you can use func with array
def delayedExpRise_vec_self(x,a,b,t0):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i]=delayedExpRise(x[i],a,b,t0)
    return y

temporalParams=np.zeros((nParamsCombo,nPoints,3))
for n in np.arange(nPoints):
    print("Processing observation well #"+str(n)+'...')
    for i in np.arange(nParamsCombo):
        datanow = data[i,:,n]
        try:
            if np.max(datanow)>cutOffC:
                pOpt, pCov = curve_fit(delayedExpRise_vec_self, t, datanow, p0=(np.max(datanow),1,15),bounds=([np.max(datanow)*0.99,0.1,0.1],[1,1E5,60]))
                #print(str(pOpt))
                if pOpt[1]<1E4:
                    temporalParams[i,n,:] = pOpt
                else:
                    temporalParams[i,n,:] = [0,np.NaN,np.NaN]
            else:
                temporalParams[i,n,:] = [0,np.NaN,np.NaN]
        except RuntimeError:
            print("# Error - curve_fit failed @ n="+str(n)+", i="+str(i))
            temporalParams[i,n,:] = [0,np.NaN,np.NaN]#3*[np.NaN]
print('# Fits complete.')    
xPoint = np.remainder(pointNumber,5)*20 + 20

#%% prep for plots
nParamUnique=[]
i=0
for row in params.transpose():
    nParamUnique.append(np.unique(row).size)
    print('# '+paramNames[i]+' has '+str(nParamUnique[i])+' unique values')
    i = i+1

indsOrder = np.array([2,3,0,4,5])   # indices of params for x axis, y-axis, nx plots, ny plots, param for each plot
                                    # should be: r_1, D_F, K_aquifer, z+aquifer, f_retardation
nx,ny=nParamUnique[indsOrder[2]],nParamUnique[indsOrder[3]]

#%% when there are 5 parameters, must do everything n times, where n=# of parameter values for the 5th parameter
valsExtraParam = np.unique(params[:,indsOrder[4]])
dtNow = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
fileOutName='out/'+fileName+'_OUT_'+dtNow+'.pdf'
pp = PdfPages(fileOutName)

# begin loop
for lastParamVal in valsExtraParam: # here, this should be relaxation factor
    fig=plt.figure(figsize=(thefigsize))
    plt.plot()
    fig.patch.set_visible(False)
    plt.axis('off')
    textForFig = fileOutName+ '\n' + '$f_{re} =$ ' + str(lastParamVal) +'\n' + 'point # = ' +str(pointNumber+1) +' (@ '+str(xPoint)+' m)\n'+'\n\n order of figures:\n'+'equilibrium concentration \n value of $b$ parameter (years) \n value of $b$ in terms of pore volumes \n x_0 for steady-state dcay with length'
    plt.text(0,0,textForFig,multialignment='center',ha='center')
    pp.savefig(fig)
    plt.close()

    # make plot of maximum concentration value:
    CMin,CMax = 0,1
    fig, axes = plt.subplots(nx, ny,figsize=(thefigsize)) 
    for ix in np.arange(nx):
        valix = np.unique(params[:,indsOrder[2]])[ix]
        for iy in np.arange(ny):
            valiy = np.unique(params[:,indsOrder[3]])[iy]
            indsNow = np.where(np.logical_and(np.logical_and(params[:,indsOrder[2]]==valix,params[:,indsOrder[3]]==valiy),
                                              params[:,indsOrder[4]]==lastParamVal))
            dataNow = temporalParams[indsNow,pointNumber,0]
            yNow=params[indsNow,indsOrder[0]]
            xNow=params[indsNow,indsOrder[1]]
            sizeNow=np.array([nParamUnique[indsOrder[0]],nParamUnique[indsOrder[1]]])
            if ny==1:
                plt.sca(axes[ix])
                axNow=axes[ix]
            else:
                plt.sca(axes[ix,iy])
                axNow=axes[ix,iy]
            axNow.imshow(dataNow.reshape(sizeNow),vmax=CMax,vmin=CMin)
            xTickLabels = [ '{:.2e}'.format(l) for l in xNow[0,0:np.unique(params[:,indsOrder[0]]).size] ]
            yTickLabels = yNow[0,0::np.unique(params[:,indsOrder[1]]).size]
            plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            if ix==0: # ix and iy are backwards, I think....
                titlestr=paramNames[indsOrder[3]]+' = ' +str(valiy)
                plt.title(titlestr,fontsize=theFontSize,weight='bold')
            if iy==ny-1:
                titlestr=paramNames[indsOrder[2]]+' = ' +str(valix)
                axNow.yaxis.set_label_position("right")
                plt.ylabel(titlestr,rotation=270,fontsize=theFontSize,ha='center', va='bottom')
            if ix==nx-1:
                plt.xticks(np.arange(np.unique(params[:,indsOrder[0]]).size), xTickLabels,rotation=15)
            else:
                plt.xticks(np.arange(np.unique(params[:,indsOrder[0]]).size),[])
            if iy==0:
                plt.yticks(np.arange(np.unique(params[:,indsOrder[1]]).size), yTickLabels)
            else:
                plt.yticks(np.arange(np.unique(params[:,indsOrder[1]]).size), [])            
            # add value labels to plot
            x_positions = np.arange(nParamUnique[indsOrder[0]])
            y_positions = np.arange(nParamUnique[indsOrder[1]])
            for y_index, y in enumerate(y_positions):
                for x_index, x in enumerate(x_positions):
                    CNow=dataNow.reshape(sizeNow)[y_index,x_index]
                    if CNow>1:
                        label='1.000' 
                    elif CNow<0.0005:
                        label='0.000' # for when concentration never exceeds cutoff
                    else:
                        label = "{:2.3f}".format(CNow)
                    text_x = x
                    text_y = y
                    thecolour=choose_colour(CNow,1)
                    axNow.text(text_x, text_y, label, color=thecolour, ha='center', va='center', name='ITC Avant Garde Gothic',rotation=45)
    fig.text(0.5, 0.05, paramNames[indsOrder[1]], ha='center',fontsize=theFontSize)
    fig.text(0.05, 0.5, paramNames[indsOrder[0]], ha='center',fontsize=theFontSize,rotation='vertical')
    pp.savefig(fig)

    #make plot of b parameter vs. 4 parameters
    bMin,bMax = 0,1000
    fig, axes = plt.subplots(nx, ny,figsize=(thefigsize))
    for ix in np.arange(nx):
        valix = np.unique(params[:,indsOrder[2]])[ix]
        for iy in np.arange(ny):
            valiy = np.unique(params[:,indsOrder[3]])[iy]
            indsNow = np.where(np.logical_and(np.logical_and(params[:,indsOrder[2]]==valix,params[:,indsOrder[3]]==valiy),
                                              params[:,indsOrder[4]]==lastParamVal))
            dataNow = temporalParams[indsNow,pointNumber,1]
            dataNow[temporalParams[indsNow,pointNumber,0]<0.0001]=np.NaN
            yNow=params[indsNow,indsOrder[0]]
            xNow=params[indsNow,indsOrder[1]]
            sizeNow=np.array([nParamUnique[indsOrder[0]],nParamUnique[indsOrder[1]]])
            if ny==1:
                plt.sca(axes[ix])
                axNow=axes[ix]
            else:
                plt.sca(axes[ix,iy])
                axNow=axes[ix,iy]
            axNow.imshow(dataNow.reshape(sizeNow),vmax=bMax,vmin=bMin,cmap='cividis')
            xTickLabels = [ '{:.2e}'.format(l) for l in xNow[0,0:np.unique(params[:,indsOrder[0]]).size] ]
            yTickLabels = yNow[0,0::np.unique(params[:,indsOrder[1]]).size]
            plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            if ix==0: # ix and iy are backwards, I think....
                titlestr=paramNames[indsOrder[3]]+' = ' +str(valiy)
                plt.title(titlestr,fontsize=theFontSize,weight='bold')
            if iy==ny-1:
                titlestr=paramNames[indsOrder[2]]+' = ' +str(valix)
                axNow.yaxis.set_label_position("right")
                plt.ylabel(titlestr,rotation=270,fontsize=theFontSize,ha='center', va='bottom')
            if ix==nx-1:
                plt.xticks(np.arange(np.unique(params[:,indsOrder[0]]).size), xTickLabels,rotation=15)
            else:
                plt.xticks(np.arange(np.unique(params[:,indsOrder[0]]).size),[])
            if iy==0:
                plt.yticks(np.arange(np.unique(params[:,indsOrder[1]]).size), yTickLabels)
            else:
                plt.yticks(np.arange(np.unique(params[:,indsOrder[1]]).size), [])            
            # add value labels to plot
            x_positions = np.arange(nParamUnique[indsOrder[0]])
            y_positions = np.arange(nParamUnique[indsOrder[1]])
            for y_index, y in enumerate(y_positions):
                for x_index, x in enumerate(x_positions):
                    bNow=dataNow.reshape(sizeNow)[y_index,x_index]
                    if bNow>bMax:
                        label='>'+str(int(bMax)) # for when attenuation takes longer than simulation results
                    elif np.isnan(bNow):
                        label='*' # for when concentration never exceeds cutoff
                    else:
                        label = "{:2.2f}".format(bNow)
                    text_x = x
                    text_y = y
                    thecolour=choose_colour(bNow,bMax)
                    axNow.text(text_x, text_y, label, color=thecolour, ha='center', va='center', name='ITC Avant Garde Gothic',rotation=45)
    fig.text(0.5, 0.05, paramNames[indsOrder[1]], ha='center',fontsize=theFontSize)
    fig.text(0.05, 0.5, paramNames[indsOrder[0]], ha='center',fontsize=theFontSize,rotation='vertical')    
    pp.savefig(fig)
    
    # plot of b parameter normalised to n volumes (normalised time) vs. 4 parameters    
    nParamUnique=[]
    i=0
    for row in params.transpose():
        nParamUnique.append(np.unique(row).size)
        #print(paramNames[i]+' has '+str(nParamUnique[i])+' unique values')
        i = i+1
    bnMin,bnMax = max(0.01,min(tx_to_unitless(365*24*3600*to_velocity(params[:,indsOrder[2]],epsilonAquifer,dHdx),temporalParams[:,pointNumber,1],xPoint))),max(tx_to_unitless(365*24*3600*to_velocity(params[:,indsOrder[2]],epsilonAquifer,dHdx),temporalParams[:,pointNumber,1],xPoint))
    fig, axes = plt.subplots(nx, ny,figsize=(thefigsize))    
    for ix in np.arange(nx):
        valix = np.unique(params[:,indsOrder[2]])[ix]
        for iy in np.arange(ny):
            valiy = np.unique(params[:,indsOrder[3]])[iy]
            indsNow = np.where(np.logical_and(np.logical_and(params[:,indsOrder[2]]==valix,params[:,indsOrder[3]]==valiy),
                                              params[:,indsOrder[4]]==lastParamVal))
            dataNowbDrop = temporalParams[indsNow,pointNumber,1]
            dataNow = tx_to_unitless(365*24*3600*to_velocity(valix,epsilonAquifer,dHdx),dataNowbDrop,xPoint) # option convert to n volumes
            dataNow[temporalParams[indsNow,pointNumber,0]<0.0001]=np.NaN
            yNow=params[indsNow,indsOrder[0]]
            xNow=params[indsNow,indsOrder[1]]
            sizeNow=np.array([nParamUnique[indsOrder[0]],nParamUnique[indsOrder[1]]])
            if ny==1:
                plt.sca(axes[ix])
                axNow=axes[ix]
            else:
                plt.sca(axes[ix,iy])
                axNow=axes[ix,iy]
            axNow.imshow(dataNow.reshape(sizeNow),vmax=bnMax,vmin=bnMin,cmap='cividis',norm=LogNorm(vmin=bnMin, vmax=bnMax))
            xTickLabels = [ '{:.2e}'.format(l) for l in xNow[0,0:np.unique(params[:,indsOrder[0]]).size] ]
            yTickLabels = yNow[0,0::np.unique(params[:,indsOrder[1]]).size]
            plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            if ix==0: # ix and iy swapped
                titlestr=paramNames[indsOrder[3]]+' = ' +str(valiy)
                plt.title(titlestr,fontsize=theFontSize,weight='bold')
            if iy==ny-1:
                titlestr=paramNames[indsOrder[2]]+' = ' +str(valix)
                axNow.yaxis.set_label_position("right")
                plt.ylabel(titlestr,rotation=270,fontsize=theFontSize,ha='center', va='bottom')
            if ix==nx-1:
                plt.xticks(np.arange(np.unique(params[:,indsOrder[0]]).size), xTickLabels,rotation=15)
            else:
                plt.xticks(np.arange(np.unique(params[:,indsOrder[0]]).size),[])
            if iy==0:
                plt.yticks(np.arange(np.unique(params[:,indsOrder[1]]).size), yTickLabels)
            else:
                plt.yticks(np.arange(np.unique(params[:,indsOrder[1]]).size), [])            
            # add value labels to plot
            x_positions = np.arange(nParamUnique[indsOrder[0]])
            y_positions = np.arange(nParamUnique[indsOrder[1]])
            for y_index, y in enumerate(y_positions):
                for x_index, x in enumerate(x_positions):
                    bnNow=dataNow.reshape(sizeNow)[y_index,x_index]
                    if bnNow>bnMax:
                        label='>'+str(int(bnMax)) # for when attenuation takes longer than simulation results
                    elif np.isnan(bnNow):
                        label='*' # for when concentration never exceeds cutoff
                    else:
                        label = "{:2.2f}".format(bnNow)
                    text_x = x
                    text_y = y
                    thecolour=choose_colour(np.log(bnNow/bnMin),np.log(bnMax))
                    axNow.text(text_x, text_y, label, color=thecolour, ha='center', va='center', name='ITC Avant Garde Gothic',rotation=45)
    fig.text(0.5, 0.05, paramNames[indsOrder[1]], ha='center',fontsize=theFontSize)
    fig.text(0.05, 0.5, paramNames[indsOrder[0]], ha='center',fontsize=theFontSize,rotation='vertical')
    pp.savefig(fig)
    
    # make plot of spatial decay parameter (x_0) for all (measure of plume length at equilibirum)   
    x0Min,x0Max = min(equilExpParam),max(equilExpParam)
    fig, axes = plt.subplots(nx, ny,figsize=(thefigsize))
    for ix in np.arange(nx):
        valix = np.unique(params[:,indsOrder[2]])[ix]
        for iy in np.arange(ny):
            valiy = np.unique(params[:,indsOrder[3]])[iy]
            indsNow = np.where(np.logical_and(np.logical_and(params[:,indsOrder[2]]==valix,params[:,indsOrder[3]]==valiy),
                                              params[:,indsOrder[4]]==lastParamVal))
            dataNow = equilExpParam[indsNow] # option convert to n volumes
            yNow=params[indsNow,indsOrder[0]]
            xNow=params[indsNow,indsOrder[1]]
            sizeNow=np.array([nParamUnique[indsOrder[0]],nParamUnique[indsOrder[1]]])
            if ny==1:
                plt.sca(axes[ix])
                axNow=axes[ix]
            else:
                plt.sca(axes[ix,iy])
                axNow=axes[ix,iy]
            axNow.imshow(dataNow.reshape(sizeNow),vmax=x0Max,vmin=x0Min,cmap='inferno',norm=LogNorm(vmin=x0Min, vmax=x0Max))
            xTickLabels = [ '{:.2e}'.format(l) for l in xNow[0,0:np.unique(params[:,indsOrder[0]]).size] ]
            yTickLabels = yNow[0,0::np.unique(params[:,indsOrder[1]]).size]
            plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            if ix==0: # ix and iy swapped
                titlestr=paramNames[indsOrder[3]]+' = ' +str(valiy)
                plt.title(titlestr,fontsize=theFontSize,weight='bold')
            if iy==ny-1:
                titlestr=paramNames[indsOrder[2]]+' = ' +str(valix)

                axNow.yaxis.set_label_position("right")

                plt.ylabel(titlestr,rotation=270,fontsize=theFontSize,ha='center', va='bottom')
            if ix==nx-1:

                plt.xticks(np.arange(np.unique(params[:,indsOrder[0]]).size), xTickLabels,rotation=15)
            else:
                plt.xticks(np.arange(np.unique(params[:,indsOrder[0]]).size),[])
            if iy==0:

                plt.yticks(np.arange(np.unique(params[:,indsOrder[1]]).size), yTickLabels)
            else:
                plt.yticks(np.arange(np.unique(params[:,indsOrder[1]]).size), [])            
            # add value labels to plot
            x_positions = np.arange(nParamUnique[indsOrder[0]])
            y_positions = np.arange(nParamUnique[indsOrder[1]])
            for y_index, y in enumerate(y_positions):
                for x_index, x in enumerate(x_positions):
                    x0Now=dataNow.reshape(sizeNow)[y_index,x_index]
                    if x0Now>x0Max:
                        label='>'+str(int(x0Max)) # for when attenuation takes longer than simulation results
                    elif np.isnan(bnNow):
                        label='*' # for when concentration never exceeds cutoff
                    else:
                        label = "{:2.1f}".format(x0Now)
                    text_x = x
                    text_y = y
                    thecolour=choose_colour(np.log(x0Now/x0Min),np.log(x0Max))
                    axNow.text(text_x, text_y, label, color=thecolour, ha='center', va='center', name='ITC Avant Garde Gothic',rotation=45)
    fig.text(0.5, 0.05, paramNames[indsOrder[1]], ha='center',fontsize=theFontSize)
    fig.text(0.05, 0.5, paramNames[indsOrder[0]], ha='center',fontsize=theFontSize,rotation='vertical')
    pp.savefig(fig)

#%% plots vs combined parameters
alpher=0.6

#%% x0 vs. pis
fig = plt.figure(figsize=(12,6))
ylimNow=[0,1]
legendNow = ['$z_{a}=1$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=1$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=1$m, $K_{a}=5$x$10^{-5}$ m/s',
             '$z_{a}=0.2$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=0.2$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=0.2$m, $K_{a}=5$x$10^{-5}$ m/s',
             '$z_{a}=5$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=5$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=5$m, $K_{a}=5$x$10^{-5}$ m/s']

# pis are from application of Buckingham Pi Theorem:
pi1=params[:,4]*np.sqrt(params[:,2]/(params[:,3]*epsilonAquitard**(4/3)))
pi2=params[:,0]*np.sqrt(params[:,3]*epsilonAquitard**(4/3))/params[:,2]**(3/2)

# x0 vs. pi1
pltNow=plt.subplot(1,2,1)
loopNow=zip(np.arange(0,nParamsCombo,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    plt.loglog(pi1[i:i+32],equilExpParam[i:i+32],c,alpha=alpher)#
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
plt.ylabel('$x_0$ [m]',fontsize=theFontSize)
plt.xlabel('$\Pi_1$ [-]',fontsize=theFontSize)
#plt.legend(['1 m','0.2 m','5 m'])

# x0 vs. pi2
pltNow=plt.subplot(1,2,2)
loopNow=zip(np.arange(0,nParamsCombo,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    plt.loglog(pi2[i:i+32],equilExpParam[i:i+32],c,alpha=alpher)#
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
#plt.ylabel('$x_0$ [m]',fontsize=theFontSize)
plt.xlabel('$\Pi_2$ [-]',fontsize=theFontSize)

pp.savefig(fig)
#%% C'_equil vs. pis
Cequil = temporalParams[:,pointNumber,0]
fig = plt.figure(figsize=(12,6))

# x0 vs. pi1
pltNow=plt.subplot(1,2,1)
loopNow=zip(np.arange(0,nParamsCombo,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    plt.semilogx(pi1[i:i+32],Cequil[i:i+32],c,alpha=alpher)#
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
plt.ylabel('$C\'_{eq}$',fontsize=theFontSize)
plt.xlabel('$\Pi_1$ [-]',fontsize=theFontSize)
#plt.legend(['1 m','0.2 m','5 m'])

# x0 vs. pi2
pltNow=plt.subplot(1,2,2)
loopNow=zip(np.arange(0,nParamsCombo,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    plt.semilogx(pi2[i:i+32],Cequil[i:i+32],c,alpha=alpher)# 
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
#plt.ylabel('$x_0$ [m]',fontsize=theFontSize)
plt.xlabel('$\Pi_2$ [-]',fontsize=theFontSize)

pp.savefig(fig)
#%% x0 & C'eq vs. eta
eta = np.sqrt(params[:,3]*params[:,2]*epsilonAquitard**(4/3))/(params[:,0]*params[:,4])

xlimNow=[3.5E-3,3.5E2]
fig = plt.figure(figsize=(12,6))
pltNow=plt.subplot(1,2,1)
loopNow=zip(np.arange(0,nParamsCombo,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
plt.semilogx(np.logspace(-3,3,200),44*np.logspace(-3,3,200)**-1,'k-',alpha=alpher) # this is the lower limit line
for i,c in loopNow:
    #plt.semilogx(chi[i:i+32],nvatt_masked[i:i+32,pointNumber],c,alpha=alpher)#
    #plt.semilogx(DHparam[i:i+32],Cequil[i:i+32],c,alpha=alpher)#
    plt.loglog(eta[i:i+32],equilExpParam[i:i+32],c,alpha=alpher)#
plt.xlim(xlimNow)
plt.ylim([3E-1,3E4]) # this is for the publication - not general.
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
plt.xlabel('$\eta$ $[m^{-1}]$',fontsize=theFontSize)
#plt.ylabel('$C\'_{eq}$ [-]',fontsize=theFontSize)
plt.ylabel('$x_0$ [m]',fontsize=theFontSize)

pltNow=plt.subplot(1,2,2)
loopNow=zip(np.arange(0,nParamsCombo,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    #plt.semilogx(chi[i:i+32],nvatt_masked[i:i+32,pointNumber],c,alpha=alpher)#
    plt.semilogx(eta[i:i+32],Cequil[i:i+32],c,alpha=alpher)#
    #plt.loglog(DHparam[i:i+32],equilExpParam[i:i+32],c,alpha=alpher)#
plt.xlim(xlimNow)
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
plt.xlabel('$\eta$ $[m^{-1}]$',fontsize=theFontSize)
plt.ylabel('$C\'_{eq}$',fontsize=theFontSize)
#plt.ylabel('$x_0$ [m]',fontsize=theFontSize)

pp.savefig(fig)

#%%
pp.close()
print('# Figures successfully written to file: '+fileOutName)
plt.close('all')
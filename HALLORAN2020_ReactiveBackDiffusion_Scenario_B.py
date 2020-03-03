# -*- coding: utf-8 -*-
"""
HALLORAN2020_ReactiveBackDiffusion_Scenario_B.py
Landon Halloran, 2020
www.ljsh.ca 
github.com/lhalloran

Python script to process and analyse model output for Scenario B in Halloran & Hunkeler (2020) paper.

"""

#%% Import necessary packages:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

#%% 
#################################### TO BE DEFINED BY USER #######################################
dropFactor = 1.0/10  # Attenuation factor (<1).
cutOffC = 0.00001    # Cut-off normalised concentration (should be >=1E-5).
pointNumber = 9      # Observation well for analysis. Above aquitard starts at 0, moving L to R (i.e., well at 100m is # 4)
                     # Below aquitard starts at 5, moving L to R (i.e., well at 100m is # 9)
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

#%% READ AND PRE-PROCESS DATA
fileName = 'ScenarioB.csv'
theFontSize=14
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

nt = 211 # number of time steps
nPoints = 10
nParams = 6
nParamsCombo = 288
dHdx = 0.01 # horizontal hydraulic gradient 
epsilonAquitard = 0.4 # porosity in aquitard
epsilonAquifer = 0.35 # porosity in aquifer
tRemove = 10

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

tDrop = np.zeros((nParamsCombo,nPoints))-1.0 # time where C drops by dropfactor from peak
maxC = np.zeros((nParamsCombo,nPoints))-1.0 # maximum concentration 
for i in np.arange(nParamsCombo):
    for j in np.arange(nPoints):
        dataNow = data[i,:,j]
        maxNow = np.max(dataNow)
        if maxNow>=cutOffC:
            maxC[i,j] = maxNow
        else:
            maxC[i,j] = cutOffC*0.1 # assign small value
        
        if params[i,0]==5E-5: # max value is basically constant over a certain period in many of of the fastest flow cases, so define as t=10 days
            indMaxNow = 10
        else:
            indMaxNow = np.argmax(dataNow)
            
        if maxNow<cutOffC:
            tDrop[i,j] = np.NaN # if no significant concentration is seen (i.e. down in rounding errors)
        elif np.argmax(dataNow[indMaxNow:]<=maxNow*dropFactor) == 0:
            tDrop[i,j] = t[-1]+1.0 # if dropFactor value is never seen, assign 200+1 years
        else:
            indDrop = np.argmax(dataNow[indMaxNow:]<=maxNow*dropFactor)+indMaxNow
            # this now does linear approximation to estimate the inter-year point 
            # at which the concentration drops below threshold...
            C2,t2 = dataNow[indDrop],t[indDrop]
            C1,t1 = dataNow[indDrop-1],t[indDrop-1]
            tDropNow = t2 - 1 + (C1-maxNow*dropFactor)/(C1-C2)
            tDrop[i,j] = tDropNow - t[indMaxNow] # tDrop is now time since max
nvatt=tDrop-tDrop

thefigsize=10,10 # general plotting parameters
xPoint = np.remainder(pointNumber,5)*20 + 20 # x distance of points

#%% prep for plots
nParamUnique=[]
i=0
for row in params.transpose():
    nParamUnique.append(np.unique(row).size)
    print(paramNames[i]+' has '+str(nParamUnique[i])+' unique values')
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
for lastParamVal in valsExtraParam: # here, this should be retardation factor
    fig=plt.figure(figsize=(thefigsize))
    plt.plot()
    fig.patch.set_visible(False)
    plt.axis('off')
    textForFig = fileOutName+ '\n' + '$f_{re} =$ ' + str(lastParamVal) +'\n' + 'point # = ' +str(pointNumber+1) +' (@ '+str(xPoint)+' m)\n'+'drop factor = ' + str(dropFactor) +'\n cutOffC = '+str(cutOffC)+'\n\n order of figures:\n'+'time (years) for attenuation after peak \n peak value (mol/m$^3$) \n n pore volumes for attenuation after peak'
    plt.text(0,0,textForFig,multialignment='center',ha='center')
    pp.savefig(fig)
    plt.close()
    
    # make plot of tDrop vs. 4 parameters
    tMin,tMax = 0,t[-1]-tRemove
    fig, axes = plt.subplots(nx, ny,figsize=(thefigsize))
    
    for ix in np.arange(nx):
        valix = np.unique(params[:,indsOrder[2]])[ix]
        for iy in np.arange(ny):
            valiy = np.unique(params[:,indsOrder[3]])[iy]
            indsNow = np.where(np.logical_and(np.logical_and(params[:,indsOrder[2]]==valix,params[:,indsOrder[3]]==valiy),
                                              params[:,indsOrder[4]]==lastParamVal))
            dataNow = tDrop[indsNow,pointNumber]
            yNow=params[indsNow,indsOrder[0]]
            xNow=params[indsNow,indsOrder[1]]
            sizeNow=np.array([nParamUnique[indsOrder[0]],nParamUnique[indsOrder[1]]])
            if ny==1:
                plt.sca(axes[ix])
                axNow=axes[ix]
            else:
                plt.sca(axes[ix,iy])
                axNow=axes[ix,iy]
            axNow.imshow(dataNow.reshape(sizeNow),vmax=tMax,vmin=tMin,cmap='cividis')
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
                    tNow=dataNow.reshape(sizeNow)[y_index,x_index]
                    if tNow>t[-1]-tRemove:
                        label='>'+str(int(t[-1]-tRemove)) # for when attenuation takes longer than simulation results
                    elif np.isnan(tNow):
                        label='*' # for when concentration never exceeds cutoff
                    else:
                        label = "{:2.2f}".format(tNow)
                    text_x = x
                    text_y = y
                    thecolour=choose_colour(tNow,t[-1]-tRemove)
                    axNow.text(text_x, text_y, label, color=thecolour, ha='center', va='center', name='ITC Avant Garde Gothic',rotation=45)
    fig.text(0.5, 0.05, paramNames[indsOrder[1]], ha='center',fontsize=theFontSize)
    fig.text(0.05, 0.5, paramNames[indsOrder[0]], ha='center',fontsize=theFontSize,rotation='vertical')    
    pp.savefig(fig)
    
    # make plot of maximum concentration value:
    CMin,CMax = np.min(maxC[:,pointNumber]),np.max(maxC[:,pointNumber])
    fig, axes = plt.subplots(nx, ny,figsize=(thefigsize))   
    for ix in np.arange(nx):
        valix = np.unique(params[:,indsOrder[2]])[ix]
        for iy in np.arange(ny):
            valiy = np.unique(params[:,indsOrder[3]])[iy]
            indsNow = np.where(np.logical_and(np.logical_and(params[:,indsOrder[2]]==valix,params[:,indsOrder[3]]==valiy),
                                              params[:,indsOrder[4]]==lastParamVal))
            dataNow = maxC[indsNow,pointNumber]
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
                    elif CNow<cutOffC:
                        label='*' # for when concentration never exceeds cutoff
                    else:
                        label = "{:2.3f}".format(CNow)
                    text_x = x
                    text_y = y
                    thecolour=choose_colour(CNow,CMax)
                    axNow.text(text_x, text_y, label, color=thecolour, ha='center', va='center', name='ITC Avant Garde Gothic',rotation=45)
    fig.text(0.5, 0.05, paramNames[indsOrder[1]], ha='center',fontsize=theFontSize)
    fig.text(0.05, 0.5, paramNames[indsOrder[0]], ha='center',fontsize=theFontSize,rotation='vertical')
    pp.savefig(fig)

    # make plot of n volumes (normalised time) vs. 4 parameters
    nParamUnique=[]
    i=0
    for row in params.transpose():
        nParamUnique.append(np.unique(row).size)
        print(paramNames[i]+' has '+str(nParamUnique[i])+' unique values')
        i = i+1
    tnMin,tnMax = min(tx_to_unitless(365*24*3600*to_velocity(params[:,indsOrder[2]],epsilonAquifer,dHdx),tDrop[:,pointNumber],xPoint)),max(tx_to_unitless(365*24*3600*to_velocity(params[:,indsOrder[2]],epsilonAquifer,dHdx),tDrop[:,pointNumber],xPoint))
    fig, axes = plt.subplots(nx, ny,figsize=(thefigsize))    
    for ix in np.arange(nx):
        valix = np.unique(params[:,indsOrder[2]])[ix]
        for iy in np.arange(ny):
            valiy = np.unique(params[:,indsOrder[3]])[iy]
            #print(str(valix)+','+str(valiy))
            indsNow = np.where(np.logical_and(np.logical_and(params[:,indsOrder[2]]==valix,params[:,indsOrder[3]]==valiy),
                                              params[:,indsOrder[4]]==lastParamVal))
            dataNowtDrop = tDrop[indsNow,pointNumber]
            dataNow = tx_to_unitless(365*24*3600*to_velocity(valix,epsilonAquifer,dHdx),dataNowtDrop,xPoint) # option convert to n volumes
            nvatt[indsNow,pointNumber] = dataNow
            yNow=params[indsNow,indsOrder[0]]
            xNow=params[indsNow,indsOrder[1]]
            sizeNow=np.array([nParamUnique[indsOrder[0]],nParamUnique[indsOrder[1]]])
            if ny==1:
                plt.sca(axes[ix])
                axNow=axes[ix]
            else:
                plt.sca(axes[ix,iy])
                axNow=axes[ix,iy]
            axNow.imshow(dataNow.reshape(sizeNow),vmax=tnMax,vmin=tnMin,cmap='cividis',norm=LogNorm(vmin=tnMin, vmax=tnMax))
            xTickLabels = [ '{:.2e}'.format(l) for l in xNow[0,0:np.unique(params[:,indsOrder[0]]).size] ]
            yTickLabels = yNow[0,0::np.unique(params[:,indsOrder[1]]).size]
            plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            if ix==0:
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
                    tnNow = dataNow.reshape(sizeNow)[y_index,x_index]
                    if dataNowtDrop.reshape(sizeNow)[y_index,x_index]>t[-1]-tRemove:
                        label='>'+"{:2.2f}".format(tnNow) # for when attenuation takes longer than simulation results
                    elif np.isnan(tnNow):
                        label='*' # for when concentration never exceeds cutoff
                    else:
                        label = "{:2.2f}".format(tnNow)
                    text_x = x
                    text_y = y
                    thecolour=choose_colour(np.log(tnNow/tnMin),np.log(tnMax))
                    axNow.text(text_x, text_y, label, color=thecolour, ha='center', va='center', name='ITC Avant Garde Gothic',rotation=45)
    fig.text(0.5, 0.05, paramNames[indsOrder[1]], ha='center',fontsize=theFontSize)
    fig.text(0.05, 0.5, paramNames[indsOrder[0]], ha='center',fontsize=theFontSize,rotation='vertical')
    pp.savefig(fig)
#%% 
fig=plt.figure(figsize=(thefigsize))
plt.plot()
fig.patch.set_visible(False)
plt.axis('off')
textForFig = fileOutName+ '\n Plots involving combined parameters $Pi_1$, $Pi_2$, and $\eta$...'
plt.text(0,0,textForFig,multialignment='center',ha='center')
pp.savefig(fig)
plt.close()

#%% plots vs combined parameters
tDropMasked = np.ma.masked_where(tDrop==201,tDrop) 
nDrop = tx_to_unitless(365*24*3600*to_velocity(params[:,0],epsilonAquifer,dHdx),tDropMasked[:,pointNumber],xPoint)

# pis are from application of Buckingham Pi Theorem:
pi1=params[:,4]*np.sqrt(params[:,2]/(params[:,3]*epsilonAquitard**(4/3)))
pi2=params[:,0]*np.sqrt(params[:,3]*epsilonAquitard**(4/3))/params[:,2]**(3/2)

alpher=0.6
nvatt_masked = np.ma.masked_where(tDrop==201,nvatt)

#%% n_v,att vs. Pis 
fig = plt.figure(figsize=(12,6))
ylimNow=[0.1,100]
legendNow = ['$z_{a}=1$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=1$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=1$m, $K_{a}=5$x$10^{-5}$ m/s',
             '$z_{a}=0.2$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=0.2$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=0.2$m, $K_{a}=5$x$10^{-5}$ m/s',
             '$z_{a}=5$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=5$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=5$m, $K_{a}=5$x$10^{-5}$ m/s']

# nvatt vs. Pi1
pltNow=plt.subplot(1,2,1)
loopNow=zip(np.arange(0,288,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    plt.loglog(pi1[i:i+32],nvatt_masked[i:i+32,pointNumber],c,alpha=alpher)#
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
plt.xlabel('$\Pi_1$ [-]',fontsize=theFontSize)
plt.ylabel('$n_{v,att}$',fontsize=theFontSize)
plt.ylim(ylimNow)
plt.legend(legendNow,fontsize=theFontSize-6)

# nvatt vs. Pi2
pltNow = plt.subplot(1,2,2)
loopNow=zip(np.arange(0,288,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    plt.loglog(pi2[i:i+32],nvatt_masked[i:i+32,pointNumber],c,alpha=alpher)#
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
plt.xlabel('$\Pi_2$ [-]',fontsize=theFontSize)
plt.ylabel('$n_{v,att}$',fontsize=theFontSize)
plt.ylim(ylimNow)

pp.savefig(fig)
#%% C'max vs. Pis
fig = plt.figure(figsize=(12,6))

ylimNow=[0,1]
legendNow = ['$z_{a}=1$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=1$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=1$m, $K_{a}=5$x$10^{-5}$ m/s',
             '$z_{a}=0.2$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=0.2$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=0.2$m, $K_{a}=5$x$10^{-5}$ m/s',
             '$z_{a}=5$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=5$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=5$m, $K_{a}=5$x$10^{-5}$ m/s']

# nvatt vs. Pi1
pltNow=plt.subplot(1,2,1)
loopNow=zip(np.arange(0,288,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    plt.semilogx(pi1[i:i+32],maxC[i:i+32,pointNumber],c,alpha=alpher)#
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
plt.xlabel('$\Pi_1$ [-]',fontsize=theFontSize)
plt.ylabel('$C\'_{max}$',fontsize=theFontSize)
plt.ylim(ylimNow)
plt.legend(legendNow,fontsize=theFontSize-6)

# nvatt vs. Pi2
pltNow = plt.subplot(1,2,2)
loopNow=zip(np.arange(0,288,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    plt.semilogx(pi2[i:i+32],maxC[i:i+32,pointNumber],c,alpha=alpher)#
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
plt.xlabel('$\Pi_2$ [-]',fontsize=theFontSize)
plt.ylabel('$C\'_{max}$',fontsize=theFontSize)
plt.ylim(ylimNow)

pp.savefig(fig)
#%% C'max vs. n_v,att
fig = plt.figure(figsize=(6,6))
pltNow=plt.subplot(1,1,1)
loopNow=zip(np.arange(0,288,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    plt.semilogx(nvatt_masked[i:i+32,pointNumber],maxC[i:i+32,pointNumber],c,alpha=alpher)#
plt.xlabel('$n_{v,att}$',fontsize=theFontSize)
plt.ylabel('$C\'_{max}$',fontsize=theFontSize)
plt.ylim([0,1])
plt.xlim([0.1,100])
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
#plt.legend(['1 m','0.2 m','5 m'])
legendNow = ['$z_{a}=1$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=1$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=1$m, $K_{a}=5$x$10^{-5}$ m/s',
             '$z_{a}=0.2$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=0.2$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=0.2$m, $K_{a}=5$x$10^{-5}$ m/s',
             '$z_{a}=5$m, $K_{a}=2$x$10^{-6}$ m/s','$z_{a}=5$m, $K_{a}=1$x$10^{-5}$ m/s','$z_{a}=5$m, $K_{a}=5$x$10^{-5}$ m/s']
plt.legend(legendNow,fontsize=theFontSize-6)

pp.savefig(fig)
#%% ETA PARAMETER
# n_v,att and C'max vs eta

eta = np.sqrt(params[:,3]*params[:,2]*epsilonAquitard**(4/3))/(params[:,0]*params[:,4])

fig = plt.figure(figsize=(12,6))
pltNow=plt.subplot(1,2,1)
loopNow=zip(np.arange(0,nParamsCombo,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    plt.loglog(eta[i:i+32],nvatt_masked[i:i+32,pointNumber],c,alpha=alpher)#
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
plt.xlabel('$\eta$ $[m^{-1}]$',fontsize=theFontSize)
plt.ylabel('$n_{v,att}$',fontsize=theFontSize)

pltNow=plt.subplot(1,2,2)
loopNow=zip(np.arange(0,nParamsCombo,32),['gv','go','g^','bv','bo','b^','rv','ro','r^'])
for i,c in loopNow:
    plt.semilogx(eta[i:i+32],maxC[i:i+32,pointNumber],c,alpha=alpher)#
pltNow.grid(True,which="both",ls=":",linewidth=0.5,c='grey')
plt.xlabel('$\eta$ $[m^{-1}]$',fontsize=theFontSize)
plt.ylabel('$C\'_{max}$',fontsize=theFontSize)

pp.savefig(fig)
#%% ...and we're done!
pp.close()
print('Figures successfully written to file: '+fileOutName)
plt.close('all')
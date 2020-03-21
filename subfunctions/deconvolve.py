#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:23:30 2019

Various Deconvolution approach used for the RF technique.
@author: pm
"""

import numpy as np
from subfunctions.nextPowerOf2 import nextPowerOf2
import subfunctions.SignalProcessingTB as sptb
# import spectrum


def gen_it(P,H,dt,mu,shift=0,width=2.5,e_min=5,omega_min=0.5,it_max=400,):#,swv,swv_width,it_max,shift,dt,mu):
    """ This function provides a generalised iterative deconvolution as given by Wang & Pavlis (2016)
Essentially, H will be deconvolved by P. When using this it is best to use normalised traces in order to not
having to adjust mu too much!

 INPUT parameters
 P - the source wavelet estimation, in receiver functions the P-, longitudinal, or vertical component (denominator)
 H - the horizontal component (i.e. Sv/Sh, Q/T, radial/transverse component) (enumerator)
 e_min - is the ratio of residual and original data after the nth iteration; given in per cent [5]
 omega_min - secondary control parameter (percentage improvement per iteration), given in per cent - used as criterion to test significance
 swv - kind of source wavelet (e.g. Gaussian, delta) that the result will be convolved with (not yet implemented)
 width - width of the source wavelet in seconds - the wider the more noise is supress but resolution decreases
 it_max - maximal number of iterations before the algorithm interrupts
 shift - time shift of the waveform, where t=0 is the theoretical arrival P-wave arrival, should be positive when the trace starts earlier!
 dt - sampling interval
 mu - the damping factor for a least-squares damped solution
"""
    # From per cent to decimal
    e_min = e_min/100
    omega_min = omega_min/100
    
    #create proper numpy arrays (allow functions on arrays as in Matlab)
    P = np.array(P,dtype=float)
    H = np.array(H,dtype=float)
    N = len(H)
    N2 = nextPowerOf2(N)
    
    # Pad input with zeros to the next power of 2
    P = np.append(P,(N2-N)*[0])
    H = np.append(H,(N2-N)*[0])
    
    # cast source wavelet estimation into frequency domain
    P_ft = np.fft.fft(P,n=N2)

    # damp source wavelet
    P_ftd = np.conj(P_ft)/(np.conj(P_ft)*P_ft*dt + mu) #dt
    P_d = np.real(np.fft.ifft(P_ftd,n=N2)) #back to time domain
    
    
    
            ##### PREPARING PARAMETERS FOR LOOP #####
    it = 0 #number of iteration
    e = 100 #start value
    r = H # the first residual value
    
    # Begin iterative process:
    IR_new = np.zeros(N2)
    IR = IR_new
    
    rej = 0 #Number of rejected peaks
    
            ##### WHILE LOOP FOR ITERATIVE DECON #####
    while e>e_min and it<it_max: #
        it = it+1
        a = sptb.convf(P_d,r,N2,dt) #convolve eestimated source pulse & residual - similiar to 
        

        k = np.argmax(abs(a[0:N2])) #Find position of maximum correlation, with k*dt
        
        
        #That's the scaling from the original paper, which doesn't make very much sense to me
        #s2 = a[k]*P
        #scale = np.sqrt(np.dot(s2,H))/np.linalg.norm(H)# scaling this scaling makes small results even smaller, why?
        
        scale = (mu + np.linalg.norm(P_ft))/(np.sum(P**2)) #scale akin to the original by Ligorria
        ak = a[k]*scale
        
        IR_new[k] = IR[k] + ak #construct estimated impulse response 
        
        # convolve impulse response and source wavelet to estimation of horizontal component:
        est = sptb.convf(IR_new,P,N2,dt)
        
        # compute residual for ongoing iteration it
        r_new = H - est 
        omega = abs((np.linalg.norm(r)**2-np.linalg.norm(r_new)**2)/np.linalg.norm(r)**2) #significance criterion
        
         # Calculate criterion for convergence
        e = np.linalg.norm(r_new)**2/np.linalg.norm(H)**2 #changed it too squared norm
        #e = np.linalg.norm(r_new)/np.linalg.norm(H) #That's the original from the paper
        # If convergence criteria weren't met and sigificance criterion was met
        #Else the significance criterion isn't met and the peak is most likely noise and thus discarded
        if omega>omega_min: 
            IR = IR_new
            r = r_new #set new residual value
        else: #reject peak
            IR_new = IR #discard insignifcant peak
            rej = rej+1
    
    # Create receiver function
    Gauss = sptb.gaussian(N2,dt,width)
    rf = sptb.filter(IR,Gauss,dt,N2)
    
    # shift and truncate RF
    if shift: #only if shift !=0
        rf = sptb.sshift(rf,N2,dt,shift)
    rf = rf[0:N] #truncate
            
    return rf,IR,it,rej
    
   
    

    
def it(P,H,dt,shift=0,width=2.5,omega_min=0.5,it_max=400):
    """ Iterative deconvolution after Ligorria & Ammon (1999)
Essentially, H will be deconvolved by P.
 INPUT parameters
 P - the source wavelet estimation, in P receiver functions the P-, longitudinal, or vertical component
 H - the horizontal component (i.e. Sv/Sh, Q/T, radial/transverse component)
 omega_min - convergence control parameter (percentage improvement per iteration), given in per cent [0.01]
 swv - kind of source wavelet (e.g. Gaussian, delta) that the result will be convolved with - not yet implemented
 width - width of the source wavelet in seconds - the wider the more noise is supress but resolution decreases
 it_max - maximal number of iterations before the algorithm interrupts [400]
 shift - time shift of the waveform, where t=0 is the theoretical arrival P-wave arrival, should be positive when the trace starts earlier!
 dt - sampling interval
"""   
    omega_min = omega_min/100 #change from per cent to decimal
    
    #create proper numpy arrays (allow functions on arrays as in Matlab)
    P = np.array(P,dtype=float)
    H = np.array(H,dtype=float)
    N = len(H)
    N2 = nextPowerOf2(N) 

    # Pad input with zeros to the next power of 2
    P = np.append(P,(N2-len(P))*[0])
    H = np.append(H,(N2-len(H))*[0])

            ##### PREPARING PARAMETERS FOR LOOP #####
    it = 0 #number of iteration
    r = H # the first residual value
    # Begin iterative process:
       
    IR = np.zeros(N2)
    omega = 1 # start value
    
            ##### WHILE LOOP FOR ITERATIVE DECON #####
    while omega>omega_min and it<it_max: # 
        it = it+1
        
        # find cross-correlation
        a = sptb.corrf(r,P,N2)
        
        k = np.argmax(abs(a[0:N2])) #Find position of maximum correlation, with k*dt
        
        
        # Find amplitude of point with highest correlation
        ak = a[k]/(np.sum(np.power(P,2))*dt) #That dt is here because I don't have it in the corrf function
        
        IR[k] = IR[k] + ak #construct estimated impulse response 
        
        # convolve impulse response and source wavelet:
        est = sptb.convf(IR,P,N2,dt)
        
        # compute residual for ongoing iteration it
        r_new = H - est
        omega = abs((np.linalg.norm(r)**2-np.linalg.norm(r_new)**2)/np.linalg.norm(r)**2)
        r = r_new
        
        
    # Create receiver function
    Gauss = sptb.gaussian(N2,dt,width)
    rf = sptb.filter(IR,Gauss,dt,N2)
    
    # shift and truncate RF
    if shift: #only if shift !=0
        rf = sptb.sshift(rf,N2,dt,shift)
    rf = rf[0:N]
    
    return rf,it,IR

   
def damped(P,H,mu=10):
    """ A simple damped deconvolution. I just used it to benchmark
    INPUT:
        P: denominator (t-domain)
        H: enumerator (t-domain)
        mu: damping factor"""  
    P = np.array(P, dtype=float)
    H = np.array(H, dtype=float)
    N = len(H)
    N2 = nextPowerOf2(N)
    # cast source wavelet estimation into frequency domain
    P_ft = np.fft.fft(P)#,n=N2)
    H_ft = np.fft.fft(H)#,n=N2)
    
    IR_ft = H_ft*np.conj(P_ft)/(np.conj(P_ft)*P_ft+mu)
    IR = np.real(np.fft.ifft(IR_ft))[0:N]
    return IR


def waterlevel(v, u, mu=.0025):
    N = len(v)
    N2 = nextPowerOf2(N)
    # fft
    v_ft = np.fft.fft(v)#, n=N2)
    u_ft = np.fft.fft(u)#, n=N2)
    denom = np.conj(v_ft)*v_ft
    # for ii, x in denom:
    #     if x < mu:
    #         denom[ii] = mu
    RF_ft = np.conj(v_ft)*u_ft/denom
    RF = np.real(np.fft.ifft(RF_ft))[0:N]
    return RF

def multitaper(P,H,mu,k=3,p=2.5):
    """ Multitaper deconvolution (Park & Levis (2000)). That still needs some work. For some reason,
    flips the plot. I should also implement the newer version from 2006
    INPUT:
        P: denominator (t-domain)
        H: enumerator (t-domain)
        p: time bandwidth product
        k: use kth Slepian sequence"""  
    # create Slepian taper
    P = np.array(P,float)
    H = np.array(H,float)
    N = len(H)
    N2 = nextPowerOf2(N)
    #P_ft = np.fft.fft(P,n=N2)
    #H_ft = np.fft.fft(H,n=N2)
    #slep= np.array([],float)
    Y_H = np.array([0]*N2,float)
    Y_P = np.array([0]*N2,float)
    
    slep = spectrum.dpss(N,p,k)      #create Slepian
    
    for ii in range(k):
        Y_H = np.vstack((Y_H,np.fft.fft(slep[0][:,ii]*H,n=N2)))
        Y_P = np.vstack((Y_P,np.fft.fft(slep[0][:,ii]*P,n=N2)))
        
    u =  sum(np.conj(Y_H)*Y_P) #enumerator
    v = np.sqrt(sum(np.conj(Y_H)*Y_H)*sum(np.conj(Y_P)*Y_P))  #denominator
    C = u/v #coherence estimate in f-domain, used for variance
    #
    # receiver function
    v = sum(np.conj(Y_P)*Y_P) + mu #mu is a spectrum damping factor and should be a function of f
    H = u/v
    
    h = np.real(np.fft.ifft(H,N2)) #back to time domain
    # h should technically be truncated
    
    # calculate variance
    var = (1-np.power(C,2))/((k-1)*np.power(C,2))*np.power(H,2)
        
    return h,var
    
    
    
    
    
    


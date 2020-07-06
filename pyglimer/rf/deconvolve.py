#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:23:30 2019

Various Deconvolution approaches used for the RF technique.
@author: pm
"""

import numpy as np
from scipy.signal.windows import dpss

from ..utils.nextpowof2 import nextPowerOf2
import pyglimer.utils.signalproc as sptb


def gen_it(P, H, dt, mu, shift=0, width=2.5, e_min=5, omega_min=0.5,
           it_max=400):
    """
    Has a bug in the normalisation!
    This function provides a generalised iterative deconvolution as given
    by Wang & Pavlis (2016)
    Essentially, P will be deconvolved from H.

    :param P: The source wavelet estimation (denominator).
    :type P: np.ndarray
    :param H: Impulse response convolved with source wavelet (enumerator).
    :type H: np.ndarray
    :param dt: Sampling interval [s]
    :type dt: float
    :param mu: Damping factor for least-square damping.
    :type mu: float
    :param shift: Time shift of theoretical arrival [s]. The default is 0.
    :type shift: float, optional
    :param width: Width parameter for Gaussian. The default is 2.5.
    :type width: float, optional
    :param e_min: Is the energy ratio of residual and original data
        after the nth iteration; given in per cent. The default is 5.
    :type e_min: float, optional
    :param omega_min: secondary control parameter (percentage improvement per
        iteration), given in per cent - used as criterion to test significance.
        The default is 0.5.
    :type omega_min: float, optional
    :param it_max: Maximum number of iterations. defaults to 400
    :type it_max: int, optional
    :return: rf : The receiver function.
        IR : The estimation of the medium's impulse response.
        it : Number of iterations
        rej : Number of peaks that did not meet the criterion of significance.
    :rtype: rf : 1D np.ndarray
        IR : 1D np.ndarray
        it : int
        rej : int
    """

    # From per cent to decimal
    e_min = e_min/100
    omega_min = omega_min/100

    # create proper numpy arrays (allow functions on arrays as in Matlab)
    P = np.array(P, dtype=float)
    H = np.array(H, dtype=float)
    N = len(H)
    N2 = nextPowerOf2(N)

    # Pad input with zeros to the next power of 2
    P = np.append(P, (N2-N)*[0])
    H = np.append(H, (N2-N)*[0])

    # cast source wavelet estimation into frequency domain
    P_ft = np.fft.fft(P, n=N2)

    # damp source wavelet
    P_ftd = np.conj(P_ft)/(np.conj(P_ft)*P_ft*dt + mu)  # dt
    P_d = np.real(np.fft.ifft(P_ftd, n=N2))  # back to time domain

    # PREPARING PARAMETERS FOR LOOP #####
    it = 0  # number of iteration
    e = 100  # start value
    r = H  # the first residual value

    # Begin iterative process:
    IR_new = np.zeros(N2)
    IR = IR_new

    rej = 0  # Number of rejected peaks

    # WHILE LOOP FOR ITERATIVE DECON #####
    while e > e_min and it < it_max:
        it = it+1
        # convolve eestimated source pulse & residual
        a = sptb.convf(P_d, r, N2, dt)

        # Find position of maximum correlation, with k*dt
        k = np.argmax(abs(a[0:N2]))

        # That's the scaling from the original paper,
        # which doesn't make very much sense to me
        # s2 = a[k]*P
        # scale = np.sqrt(np.dot(s2,H))/np.linalg.norm(H)

        # scale akin to the original by Ligorria
        scale = (mu + np.linalg.norm(P_ft))/(np.sum(P**2))
        ak = a[k]*scale

        IR_new[k] = IR[k] + ak  # construct estimated impulse response

        # convolve impulse response and source wavelet
        # to estimation of horizontal component:
        est = sptb.convf(IR_new, P, N2, dt)

        # compute residual for ongoing iteration it
        r_new = H - est
        omega = abs((np.linalg.norm(r)**2-np.linalg.norm(r_new)**2)
                    / np.linalg.norm(r)**2)  # significance criterion

        # Calculate criterion for convergence
        e = np.linalg.norm(r_new)**2/np.linalg.norm(H)**2
        # That's the original from the paper
        # e = np.linalg.norm(r_new)/np.linalg.norm(H)
        # If convergence criteria weren't met and sigificance criterion was met
        # Else the significance criterion isn't met and the peak is most
        # likely noise and thus discarded
        if omega > omega_min:
            IR = IR_new
            r = r_new  # set new residual value
        else:  # reject peak
            IR_new = IR  # discard insignifcant peak
            rej = rej+1

    # Create receiver function
    Gauss = sptb.gaussian(N2, dt, width)
    rf = sptb.filter(IR, Gauss, dt, N2)

    # shift and truncate RF
    if shift:  # only if shift !=0
        rf = sptb.sshift(rf, N2, dt, shift)
    rf = rf[0:N]  # truncate
    return rf, IR, it, rej


def it(P, H, dt, shift=0, width=2.5, omega_min=0.5, it_max=200):
    """
    Iterative deconvolution after Ligorria & Ammon (1999)
    Essentially, P will be deconvolved from H.

    :param P: The source wavelet estimation / denominator.
    :type P: 1D np.ndarray
    :param H: The enumerator (impulse response * source wavelet estimation).
    :type H: 1D np.ndarray
    :param dt: Sampling interval [s].
    :type dt: float
    :param shift: Time shift of main arrival [s]. The default is 0.
    :type shift: float, optional
    :param width: Gaussian width parameter. The default is 2.5.
    :type width: float, optional
    :param omega_min: Convergence control parameter (percentage improvement
        per iteration). The default is 0.5.
    :type omega_min: float, optional
    :param it_max: Maximal number of iterations before the algorithm
        interrupts, defaults to 400
    :type it_max: int, optional
    :return: rf : The receiver function.
        it : Number of iterations until algorithm converged.
        IR : Estimation of the medium's impulse response.
    :rtype: rf : np.ndarray
        it : int
        IR : np.ndarray
    """

    omega_min = omega_min/100  # change from per cent to decimal
    # create proper numpy arrays (allow functions on arrays as in Matlab)
    P = np.array(P, dtype=float)
    H = np.array(H, dtype=float)
    N = len(H)
    N2 = nextPowerOf2(N)

    # Pad input with zeros to the next power of 2
    P = np.append(P, (N2-len(P))*[0])
    H = np.append(H, (N2-len(H))*[0])

    # PREPARING PARAMETERS FOR LOOP #
    it = 0  # number of iteration
    r = H  # the first residual value

    # Begin iterative process:
    IR = np.zeros(N2)
    omega = 1  # start value

    # WHILE LOOP FOR ITERATIVE DECON #####
    while omega > omega_min and it < it_max:
        it = it+1

        # find cross-correlation
        a = sptb.corrf(r, P, N2)
        # Find position of maximum correlation, with k*dt
        k = np.argmax(abs(a[0:N2]))

        # Find amplitude of point with highest correlation
        # That dt is here because I don't have it in the corrf function
        ak = a[k]/(np.sum(np.power(P, 2))*dt)

        IR[k] = IR[k] + ak  # construct estimated impulse response

        # convolve impulse response and source wavelet:
        est = sptb.convf(IR, P, N2, dt)

        # compute residual for ongoing iteration it
        r_new = H - est
        omega = abs((np.linalg.norm(r)**2 - np.linalg.norm(r_new)**2) /
                    np.linalg.norm(r)**2)
        r = r_new

    # Create receiver function
    Gauss = sptb.gaussian(N2, dt, width)
    rf = sptb.filter(IR, Gauss, dt, N2)

    # shift and truncate RF
    if shift:  # only if shift !=0
        rf = sptb.sshift(rf, N2, dt, shift)
    rf = rf[0:N]

    return rf, it, IR


def damped(P, H, mu=10):
    """Outdated use function spectraldivision instead!
    A simple damped deconvolution. I just used it to benchmark
    INPUT:
        P: denominator (t-domain)
        H: enumerator (t-domain)
        mu: damping factor"""
    P = np.array(P, dtype=float)
    H = np.array(H, dtype=float)
    N = len(H)
    N2 = nextPowerOf2(N)
    # cast source wavelet estimation into frequency domain
    P_ft = np.fft.fft(P, n=N2)
    H_ft = np.fft.fft(H, n=N2)

    IR_ft = H_ft*np.conj(P_ft)/(np.conj(P_ft)*P_ft+mu)
    IR = np.real(np.fft.ifft(IR_ft))[0:N]
    return IR


# def waterlevel(v, u, mu=.0025):
#     N = len(v)
#     N2 = nextPowerOf2(N)
#     # fft
#     v_ft = np.fft.fft(v)#, n=N2)
#     u_ft = np.fft.fft(u)#, n=N2)
#     denom = np.conj(v_ft)*v_ft
#     # for ii, x in denom:
#     #     if x < mu:
#     #         denom[ii] = mu
#     RF_ft = np.conj(v_ft)*u_ft/denom
#     RF = np.real(np.fft.ifft(RF_ft))[0:N]
#     return RF


# def multitaper_bak(P, H, mu, k=3, p=2.5):
#     """Suspended, use multitaper instead.
#     Multitaper deconvolution (Park & Levis (2000)).
#     I should also implement the newer version from 2006
#     INPUT:
#         P: denominator (t-domain)
#         H: enumerator (t-domain)
#         p: time bandwidth product
#         k: use kth Slepian sequence"""
#     # create Slepian taper
#     P = np.array(P, float)
#     H = np.array(H, float)
#     N = len(H)
#     N2 = nextPowerOf2(N)
#     # P_ft = np.fft.fft(P,n=N2)
#     # H_ft = np.fft.fft(H,n=N2)
#     # slep= np.array([],float)
#     Y_H = np.array([0]*N2, float)
#     Y_P = np.array([0]*N2, float)
#
#     slep = spectrum.dpss(N, p, k)  # create Slepian
#
#     for ii in range(k):
#         Y_H = np.vstack((Y_H, np.fft.fft(slep[0][:, ii]*H, n=N2)))
#         Y_P = np.vstack((Y_P, np.fft.fft(slep[0][:, ii]*P, n=N2)))
#
#     u = sum(np.conj(Y_H)*Y_P)  # enumerator
#     v = np.sqrt(sum(np.conj(Y_H)*Y_H)*sum(np.conj(Y_P)*Y_P))  # denominator
#     C = u/v  # coherence estimate in f-domain, used for variance
#
#     # receiver function
#     # where mu is a spectrum damping factor and should be a function of g
#     v = sum(np.conj(Y_P)*Y_P) + mu
#     H = u/v
#
#     h = np.real(np.fft.ifft(H, n=N2))  # back to time domain
#     # h should technically be truncated
#
#     # calculate variance
#     var = (1-np.power(C, 2))/((k-1)*np.power(C, 2))*np.power(H, 2)
#
#     return h, var


def spectraldivision(v, u, ndt, tshift, regul, phase):
    """
    Function spectraldivision(v,u,ndt,tshift,regul) is a standard spectral
    division frequency domain deconvolution.

    :param v: Source wavelet estimation (denominator).
    :type v: 1D np.ndarray
    :param u: Impulse response convolved by v (enumerator).
    :type u: 1D np.ndarray
    :param ndt: Sampling interval [s].
    :type ndt: float
    :param tshift: Time shift of primary arrival [s].
    :type tshift: float
    :param regul: Regularization, can be chosen by the variable "regul", this
        can be 'con', 'wat', or 'fqd' for constant damping factor, waterlevel,
        or frequency-dependent damping, respectively
    :type regul: str
    :param phase: Phase either "P" for Ps or "S" for Sp.
    :type phase: str
    :raises Exception: for uknown regulisations
    :return: qrf : Receiver function.
        lrf : Output of deoncolution of source wavelet estimation from
        longitudinal component.
    :rtype: 1D np.ndarray
    """    

    N = len(v)

    # pre-event noise needed for regularisation parameter
    vn = np.zeros(N)
    if phase == "P":
        vn[:round((tshift-2)/ndt)] = v[:round((tshift-2)/ndt)]
    elif phase == "S":
        vn[:round((tshift/2)/ndt)] = v[:round((tshift/2)/ndt)]
        # vn[:round((tshift/6)/ndt)] = u[:round((tshift/6)/ndt)]

    # number of points in fft
    nf = nextPowerOf2(N)
    # nft = nf/2 + 1

    # fourier transform
    uf = np.fft.fft(u, n=nf)
    vf = np.fft.fft(v, n=nf)
    vnf = np.fft.fft(vn, n=nf)

    # denominator and regularisation
    den = np.multiply(vf, np.conj(vf))
    noise = np.multiply(vnf, np.conj(vnf))

    # den0 = den

    # which regularization do you want?
    freqdep = regul == 'fqd'
    const = regul == 'con'
    water = regul == 'wat'

    if not freqdep and not const and not water:
        raise Exception("Regularization not defined (your input: regul=",
                        regul, """). Use either 'fqd' for frequency-dependent
                        or 'con' for constant value regularization or 'wat'
                        for water-level.""")

    # constant damping factor regularization
    if const:
        eps = max(noise.real)
        den = den + eps

    # frequency-dependent damping
    if freqdep:
        den = den+noise

    # waterlevel regularization
    if water:
        eps = max(noise.real)
        # eps = 10
        # eps = 100  # constant wl to reproduce Rychert et al
        den[den.real < eps] = eps

    # numerator
    num = np.multiply(uf, np.conj(vf))
    numl = np.multiply(vf, np.conj(vf))

    # deconvolution, with scaling by 1/ndt to bring back
    # rfl peak close to unity, for any value of ndt
    rfq = num/den*(1/ndt)
    rfl = numl/den*(1/ndt)

    # SR: this is the cos^2 filter discussed in Park and Levin, 2000,
    # on page 1509 (top of right column), with fc set to 1 Hz - probably
    # comes from Helffrich's code.
    if freqdep:
        N2 = np.floor(np.size(rfq)/2)
        for i in range(int(N2)):
            fac = np.cos(np.pi/2*(i-1)/N2)**2
            rfq[i] = rfq[i]*fac

    # back transformation
    v = np.arange(0, nf/2+1)*2*np.pi/(nf*ndt)
    v = np.concatenate((v, -np.flip(v[1:-1])))
    x = np.exp(-1j*v*tshift)
    rfq = np.multiply(rfq, x)
    rfl = np.multiply(rfl, x)

    qrft = np.fft.ifft(rfq, n=nf)
    qrf = qrft.real

    qrf = qrf[:N]
    lrft = np.fft.ifft(rfl, n=nf)
    lrf = lrft[:N].real
    lrf = lrf[:N]
    return qrf, lrf


def multitaper(P, D, dt, tshift, regul):
    """
    Output has to be filtered (Noise appears in high-frequency)!
    multitaper: takes Ved-Kathrins code and changes inputs to
    make it run like Helffrich algorithm, minus normalization and with
    questionable calculation of pre-event noise (have to double check this
    and compare it to the regular FFT based estimate without any tapers).

    Findings: when P arrival is not in the center of the window, the
    amplitudes are not unity at the beginning and decreasing from there on.
    Instead they peak at the time shift which corresponds to the middle index
    in the P time window.

    TB  = time bandwidth product (usually between 2 and 4)
    NT  = number of tapers to use, has to be <= 2*TB-1
    TB = 4; NT = 7; #choice of TB = 4, NT = 3 is supposed to be optimal
    t0 = -5; t1 = max(time); # This defines the beginning and end of the lag
    times for the receiver function

    changed by KS June 2016, added coherence and variance estimate; output
    RF in time domain (rf), RF in freq. domain (tmpp), and variance of RF
    (var_rf); can be used as input for multitaper_weighting.m
    the input "regul" defines which type of regularization is used,
    regul='fqd' defines frequency dependent regularisation from pre-event
    noise, regul='con' defines adding a constant value (here maximum of
    pre-event noise) as regularisation

    Parameters
    ----------
    P : np.array
        Source time function estimation.
    D : np.array
        Component containing the converted wave.
    dt : float
        Sampling interval [s].
    tshift : float
        Time shift of primary arrival [s].
    regul : str
        Regularization, either 'fqd' for frequency dependentor 'con' for
        constant.

    Raises
    ------
    Exception
        For unknown input.

    Returns
    -------
    rf : np.array
        The receiver function.
    lrf : np.array
        The source-time wavelet convolved by itself.
    var_rf : np.array
        The receiver function's variance.
    tmpp : np.array
        Receiver function without variance weighting.

    """

    # wavelet always in the center of the window
    # Original from Ved's
    # win_len = tshift*2;
    # Modification to get P&L
    # win_len = length(P)*dt;
    # Modificaiton to get Helffrich
    win_len = 50

    Nwin = round(win_len/dt)

    # Fraction of overlap overlap between moving time windows. As your TB
    # increases, the frequency smearing gets worse, which means that the RFs
    # degrate at shorter and shorter lag times. Therefore, as you increase TB,
    # you should also increase Poverlap.
    Poverlap = 0.75

    # Length of waveforms;
    nh = len(P)

    # tapernumber, bandwith
    # is in general put to NT=3, and bandwidth to 2.5
    # TB=2.5;
    # #NT=2*TB-1; #4 tapers
    # NT=3;
    TB = 4
    # NT=2*TB-1; #4 tapers
    NT = 3

    # Create moving time windowed slepians
    starts = np.arange(0, nh-Nwin+1, round((1-Poverlap)*Nwin))

    # Construct Slepians
    Etmp, lambdas = dpss(Nwin, TB, Kmax=NT, return_ratios=True)
    E = np.zeros([len(starts)*NT, nh])
    n = 0

    NUM = np.zeros([NT, len(P)])
    DEN = np.zeros([NT, len(D)])
    DUM = np.zeros([NT, len(D)])

    ESTP = np.zeros([NT, len(P)])
    ESTD = np.zeros([NT, len(D)])

    # finding frequency dependent regularisation parameter DEN_noise
    # added: KS 26.06.2016
    Pn = np.zeros(np.shape(P))

    # Pn(3/dt:(tshift-5)/dt)=P(3/dt:(tshift-5)/dt)
    Pn[:round((tshift-2)/dt)] = P[:round((tshift-2)/dt)]
    # pre-event noise: starting 3s after trace start
    # stop 10s before theoretical start of P
    # wave to aviod including it

    DEN_noise = np.zeros([NT, len(P)])

    # Multitaper
    # SR: problem here is how the loop is done ... there's only a peak
    # at the pulse because the two windows of the num and den are moving
    # together. One should first comput the entire estimate of the wavelet
    # and the data for each valie of k, and then do the sum of products for
    # each k!!!

    for k in range(NT):
        for j in starts:
            E[n, j:(j+Nwin)] = Etmp[k, :].transpose()

            tmp1 = np.fft.fft(np.multiply(E[n, :], P))
            tmp2 = np.fft.fft(np.multiply(E[n, :], D))

            NUM[k, :] = NUM[k, :] + np.multiply(lambdas[k]*tmp1.conj(), tmp2)
            DEN[k, :] = DEN[k, :] + np.multiply(lambdas[k]*tmp1.conj(), tmp1)

            ESTP[k, :] = ESTP[k, :] + lambdas[k]*tmp1
            ESTD[k, :] = ESTD[k, :] + lambdas[k]*tmp2

            # DUM only from D trace (converted wave component) used in
            # coherence estimate
            DUM[k, :] = DUM[k, :] + np.multiply(lambdas[k]*tmp2.conj(), tmp2)

            # pre-event noise
            # always stick to first time window
            tmp1n = np.fft.fft(np.multiply(E[n, :], Pn))
            DEN_noise[k, :] = DEN_noise[k, :] +\
                np.multiply(lambdas[k]*tmp1n.conj(), tmp1n)
            n = n + 1

    # max_imag_ep = max(abs(np.imag(np.fft.ifft(sum(ESTP)))))
    # max_imag_ed = max(abs(np.imag(np.fft.ifft(sum(ESTD)))))
    # ep = np.real(np.fft.ifft(sum(ESTP)));
    # ed = np.real(np.fft.ifft(sum(ESTD)));

    # Calculate optimal RF with frequency-dependend regularisation
    freqdep = regul == 'fqd'
    const = regul == 'con'

    if freqdep:
        tmpp = np.divide(np.sum(np.multiply(ESTP.conj(), ESTD), axis=0),
                         np.sum(np.multiply(ESTP.conj(), ESTP), axis=0)
                         + sum(DEN_noise))*1/dt
        tmpp_l = np.divide(np.sum(np.multiply(ESTP.conj(), ESTP), axis=0),
                           np.sum(np.multiply(ESTP.conj(), ESTP), axis=0)
                           + sum(DEN_noise))*1/dt

        N2 = np.floor(nh/2) + 1
        for i in range(int(N2)):
            fac = np.cos(np.pi/2*i/N2)**2
            tmpp[i] = tmpp[i]*fac

    elif const:
        # ordinary regularisation with adding only a constant value
        eps = DEN_noise.real.max() + 1
        tmpp = np.divide(np.sum(np.multiply(ESTP.conj(), ESTD), axis=0),
                         np.sum(np.multiply(ESTP.conj(), ESTP)+eps,
                                axis=0))*1/dt
        tmpp_l = np.divide(np.sum(np.multiply(ESTP.conj(), ESTD), axis=0),
                           np.sum(np.multiply(ESTP.conj(), ESTP)+eps,
                                  axis=0))*1/dt

    else:
        raise Exception('Regularization not defined (your input: regul=',
                        regul, """). Use either "fqd" for frequency-dependent
                        or "con" for constant value regularization.""")

    # RF without variance weighting
    tmp1 = np.real(np.fft.ifft(tmpp))
    tmp1_l = np.fft.ifft(tmpp_l).real

    # Interpolate to desired
    N = len(P)
    rf = tmp1[round(N-tshift/dt):N]
    rf = np.append(rf, tmp1[:N-round(tshift/dt)])
    # rf[round(tshift/dt):N] = tmp1[:N-round(tshift/dt)]

    lrf = tmp1_l[round(N-tshift/dt):N]
    lrf = np.append(lrf, tmp1_l[:round(N-tshift/dt)])
    # lrf[tshift/dt:N] = tmp1_l[:N-tshift/dt]

    ####
    # Coherence and Variance of RF
    # added: KS 26.06.2016
    C_rf = np.divide(NUM, np.sqrt(np.multiply(DUM, DEN)))
    var_rf = np.zeros(len(C_rf))
    # for ii in range(len(C_rf)):
    #     var_rf[ii] = ((1-abs(C_rf[ii])**2)/((NT-1)*abs(C_rf[ii])**2))*(abs(tmpp[ii])**2)
    var_rf = None
    return rf, lrf, var_rf, tmpp

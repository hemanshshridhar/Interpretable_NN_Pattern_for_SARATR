import numpy as np
from numpy import pi
from numpy.linalg import norm
from scipy.io import loadmat
from scipy.stats import linregress
from fnmatch import fnmatch
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from numpy import dot, pi, exp, sqrt, inf
from numpy.linalg import norm
import matplotlib.pylab as plt
from scipy.stats import linregress
from matplotlib import cm
# from . import signal as sig
# from . import phsTools
from scipy.interpolate import interp1d
import multiprocessing as mp
import numpy as np
from numpy import pi, arccosh, sqrt, cos
from scipy.fftpack import fftshift, fft2, ifft2, fft, ifft
from scipy.signal import firwin, filtfilt
def AFRL(directory, pol, start_az, n_az=3):
##############################################################################
#                                                                            #
#  This function reads in the AFRL *.mat files from the user supplied        #
#  directory and exports both the phs and a Python dictionary compatible     #
#  with ritsar.                                                              #
#                                                                            #
##############################################################################

    #Check Python version
    version = sys.version_info

    #Get filenames
    walker = os.walk(directory+'/'+pol)
    if version.major < 3:
        w = walker.next()
    else:
        w = walker.__next__()
    prefix = '/'+pol+'/'+w[2][0][0:19]
    az_str = []
    fnames = []
    az = np.arange(start_az, start_az+n_az)
    [az_str.append(str('%03d_'%a))      for a in az]
    [fnames.append(directory+prefix+a+pol+'.mat') for a in az_str]

    #Grab n_az phase histories
    phs = []; platform = []
    for fname in fnames:
        #Convert MATLAB structure to Python dictionary
        MATdata = loadmat(fname)['data'][0][0]

        data =\
        {
        'fp'    :   MATdata[0],
        'freq'  :   MATdata[1][:,0],
        'x'     :   MATdata[2].T,
        'y'     :   MATdata[3].T,
        'z'     :   MATdata[4].T,
        'r0'    :   MATdata[5][0],
        'th'    :   MATdata[6][0],
        'phi'   :   MATdata[7][0],
        }

        #Define phase history
        phs_tmp     = data['fp'].T

        phs.append(phs_tmp)

        #Transform data to be compatible with ritsar
        c           = 299792458.0
        nsamples    = int(phs_tmp.shape[1])
        npulses     = int(phs_tmp.shape[0])
        freq        = data['freq']
        pos         = np.hstack((data['x'], data['y'], data['z']))
        k_r         = 4*pi*freq/c
        B_IF        = data['freq'].max()-data['freq'].min()
        delta_r     = c/(2*B_IF)
        delta_t     = 1.0/B_IF
        t           = np.linspace(-nsamples/2, nsamples/2, nsamples)*delta_t


        chirprate, f_0, r, p, s = linregress(t, freq)
        T_p         = B_IF / chirprate

        #Vector to scene center at synthetic aperture center
        if np.mod(npulses,2)>0:
            R_c = pos[npulses // 2]  # Use floor division to ensure an integer index
        else:
            R_c = np.mean(pos[npulses // 2 - 1 : npulses // 2 + 1], axis=0)


        #Save values to dictionary for export
        platform_tmp = \
        {
            'f_0'       :   f_0,
            'freq'      :   freq,
            'chirprate' :   chirprate,
            'B_IF'      :   B_IF,
            'nsamples'  :   nsamples,
            'npulses'   :   npulses,
            'pos'       :   pos,
            'delta_r'   :   delta_r,
            'R_c'       :   R_c,
            't'         :   t,
            'k_r'       :   k_r,
            'T_p'       : T_p,
        }
        platform.append(platform_tmp)

    #Stack data from different azimuth files
    phs = np.vstack(phs)
    npulses = int(phs.shape[0])

    pos = platform[0]['pos']
    for i in range(1, n_az):
        pos = np.vstack((pos, platform[i]['pos']))

    if np.mod(npulses,2)>0:
        R_c = pos[npulses//2]
    else:
        R_c = np.mean(pos[npulses // 2 - 1 : npulses // 2 + 1], axis=0)

    #Replace Dictionary values
    platform = platform_tmp
    platform['npulses'] =   npulses
    platform['pos']     =   pos
    platform['R_c']     =   R_c

    #Synthetic aperture length
    L = norm(pos[-1]-pos[0])

    #Add k_y
    platform['k_y'] = np.linspace(-npulses/2,npulses/2,npulses)*2*pi/L

    return(phs, platform)
def img_plane_dict(platform, res_factor=1.0, n_hat = np.array([0,0,1]), aspect = 1, upsample = True):
##############################################################################
#                                                                            #
#  This function defines the image plane parameters.  The user specifies the #
#  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
#  image plane whose pixels are sized at the theoretical resolution limit    #
#  of the system (derived using delta_r which in turn was derived using the  #
#  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
#  defaults to nsamples/npulses.                                             #
#                                                                            #
#  'n_hat' is a user specified value that defines the image plane            #
#  orientation w.r.t. to the nominal ground plane.                           #
#                                                                            #
##############################################################################

    nsamples = platform['nsamples']
    npulses = platform['npulses']

    #Import relevant platform parameters
    R_c = platform['R_c']

    #Define image plane parameters
    if upsample:
        nu= 2**int(np.log2(nsamples)+bool(np.mod(np.log2(nsamples),1)))
        nv= 2**int(np.log2(npulses)+bool(np.mod(np.log2(npulses),1)))
    else:
        nu= nsamples
        nv= npulses

    #Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r']*res_factor*nsamples/nu
    dv = aspect*du

    #Define range and cross-range locations
    u = np.arange(-nu/2, nu/2)*du
    v = np.arange(-nv/2, nv/2)*dv

    #Derive image plane spatial frequencies
    k_u = 2*pi*np.linspace(-1.0/(2*du), 1.0/(2*du), nu)
    k_v = 2*pi*np.linspace(-1.0/(2*dv), 1.0/(2*dv), nv)

    #Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c)/norm(np.cross(n_hat, R_c))
    u_hat = np.cross(v_hat, n_hat)/norm(np.cross(v_hat, n_hat))

    #Represent u and v in (x,y,z)
    [uu,vv] = np.meshgrid(u,v)
    uu = uu.flatten(); vv = vv.flatten()

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T
            )))
    b = np.asmatrix(np.vstack((uu,vv)))
    pixel_locs = np.asarray(A*b)
    print(f'no of coordibaes {b.size}')
    #Construct dictionary and return to caller
    img_plane =\
    {
    'n_hat'     :   n_hat,
    'u_hat'     :   u_hat,
    'v_hat'     :   v_hat,
    'du'        :   du,
    'dv'        :   dv,
    'u'         :   u,
    'v'         :   v,
    'k_u'       :   k_u,
    'k_v'       :   k_v,
    'pixel_locs':   pixel_locs # 3 x N_pixel array specifying x,y,z location
                               # of each pixel
    }

    return(img_plane)
import numpy as np
from numpy import pi, arccosh, sqrt, cos
from scipy.fftpack import fftshift, fft2, ifft2, fft, ifft
from scipy.signal import firwin, filtfilt

#all FT's assumed to be centered at the origin
def ft(f, ax=-1):
    F = fftshift(fft(fftshift(f), axis = ax))

    return F

def ift(F, ax = -1):
    f = fftshift(ifft(fftshift(F), axis = ax))

    return f

def ft2(f, delta=1):
    F = fftshift(fft2(fftshift(f)))*delta**2

    return(F)

def ift2(F, delta=1):
    N = F.shape[0]
    f = fftshift(ifft2(fftshift(F)))*(delta*N)**2

    return(f)

def RECT(t,T):
    f = np.zeros(len(t))
    f[(t/T<0.5) & (t/T >-0.5)] = 1

    return f

def taylor1(nsamples, S_L=43):
    xi = np.linspace(-0.5, 0.5, nsamples)
    A = 1.0/pi*arccosh(10**(S_L*1.0/20))
    n_bar = int(2*A**2+0.5)+1
    sigma_p = n_bar/sqrt(A**2+(n_bar-0.5)**2)

    #Compute F_m
    m = np.arange(1,n_bar)
    n = np.arange(1,n_bar)
    F_m = np.zeros(n_bar-1)
    for i in m:
        num = 1
        den = 1
        for j in n:
            num = num*\
            (-1)**(i+1)*(1-i**2*1.0/sigma_p**2/(\
                            A**2+(j-0.5)**2))
            if i!=j:
                den = den*(1-i**2*1.0/j**2)

        F_m[i-1] = num/den

    w = np.ones(nsamples)
    for i in m:
        w += F_m[i-1]*cos(2*pi*i*xi)

    w = w/w.max()
    return(w)

def upsample(f,size):
    x_pad = size[1]-f.shape[1]
    y_pad = size[0]-f.shape[0]

    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0

    if np.mod(y_pad,2):
        y_off = 1
    else:
        y_off = 0

    F = ft2(f)
    F_pad = np.pad(F, ((y_pad/2,y_pad/2+y_off),(x_pad/2, x_pad/2+x_off)),
                   mode = 'constant')
    f_up = ift2(F_pad)

    return(f_up)

def upsample1D(f, size):
    x_pad = size-f.size

    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0

    F = ft(f)
    F_pad = np.pad(F, (x_pad/2, x_pad/2+x_off),
                   mode = 'constant')
    f_up = ift(F_pad)

    return(f_up)

def pad1D(f, size):
    x_pad = size-f.size

    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0


    f_pad = np.pad(f, (x_pad/2, x_pad/2+x_off),
                   mode = 'constant')

    return(f_pad)

def pad(f, size):
    x_pad = size[1]-f.shape[1]
    y_pad = size[0]-f.shape[0]

    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0

    if np.mod(y_pad,2):
        y_off = 1
    else:
        y_off = 0

    f_pad = np.pad(f, ((y_pad//2,y_pad//2+y_off),(x_pad//2, x_pad//2+x_off)),
                   mode = 'constant')

    return(f_pad)

def cart2sph(cart):
    x = np.array([cart[:,0]]).T
    y = np.array([cart[:,1]]).T
    z = np.array([cart[:,2]]).T
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    sph = np.hstack([azimuth, elevation, r])
    return sph

def sph2cart(sph):
    azimuth     = np.array([sph[:,0]]).T
    elevation   = np.array([sph[:,1]]).T
    r           = np.array([sph[:,2]]).T
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    cart = np.hstack([x,y,z])
    return cart

def decimate(x, q, n=None, axis=-1, beta = None, cutoff = 'nyq'):
    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n == None:
        n = int(np.log2(x.shape[axis]))

    if x.shape[axis] < n:
        n = x.shape[axis]-1

    if beta == None:
        beta = 1.*n/8

    padlen = n/2

    if cutoff == 'nyq':
        eps = np.finfo(float).eps  # ✅ Correct fix
        cutoff = 1.-eps

    window = ('kaiser', beta)
    a = 1.

    b = firwin(n, cutoff / q, window=window)
    y = filtfilt(b, [a], x, axis=axis, padlen=int(padlen))  # ✅ Ensure padlen is an integer

    sl = [slice(None)] * y.ndim
    sl[axis] = slice(None, None, q)


    return y[tuple(sl)]
def taylor1(nsamples, S_L=43):
    xi = np.linspace(-0.5, 0.5, nsamples)
    A = 1.0/pi*arccosh(10**(S_L*1.0/20))
    n_bar = int(2*A**2+0.5)+1
    sigma_p = n_bar/sqrt(A**2+(n_bar-0.5)**2)

    #Compute F_m
    m = np.arange(1,n_bar)
    n = np.arange(1,n_bar)
    F_m = np.zeros(n_bar-1)
    for i in m:
        num = 1
        den = 1
        for j in n:
            num = num*\
            (-1)**(i+1)*(1-i**2*1.0/sigma_p**2/(\
                            A**2+(j-0.5)**2))
            if i!=j:
                den = den*(1-i**2*1.0/j**2)

        F_m[i-1] = num/den

    w = np.ones(nsamples)
    for i in m:
        w += F_m[i-1]*cos(2*pi*i*xi)

    w = w/w.max()
    return(w)
def backprojection(phs, platform, img_plane, taylor = 43, upsample = 6, prnt = True):
##############################################################################
#                                                                            #
#  This is the Backprojection algorithm.  The phase history data as well as  #
#  platform and image plane dictionaries are taken as inputs.  The (x,y,z)   #
#  locations of each pixel are required, as well as the size of the final    #
#  image (interpreted as [size(v) x size(u)]).                               #
#                                                                            #
##############################################################################

    #Retrieve relevent parameters
    nsamples    =   platform['nsamples']
    npulses     =   platform['npulses']
    k_r         =   platform['k_r']
    pos         =   platform['pos']
    delta_r     =   platform['delta_r']
    u           =   img_plane['u']
    v           =   img_plane['v']
    r           =   img_plane['pixel_locs']

    #Derive parameters
    nu = u.size
    nv = v.size
    k_c = k_r[nsamples//2]

    #Create window
    win_x = taylor1(nsamples,taylor)
    win_x = np.tile(win_x, [npulses,1])

    win_y = taylor1(npulses,taylor)
    win_y = np.array([win_y]).T
    win_y = np.tile(win_y, [1,nsamples])

    win = win_x*win_y

    #Filter phase history
    filt = np.abs(k_r)
    phs_filt = phs*filt*win

    #Zero pad phase history
    N_fft = 2**(int(np.log2(nsamples*upsample))+1)
    phs_pad = pad(phs_filt, [npulses,N_fft])

    #Filter phase history and perform FT w.r.t t
    Q = ft(phs_pad)
    dr = np.linspace(-nsamples*delta_r/2, nsamples*delta_r/2, N_fft)

    #Perform backprojection for each pulse
    img = np.zeros(nu*nv)+0j
    for i in range(npulses):
        if prnt:
            print("Calculating backprojection for pulse %i" %i)
        r0 = np.array([pos[i]]).T
        dr_i = norm(r0)-norm(r-r0, axis = 0)

        Q_real = np.interp(dr_i, dr, Q[i].real)
        Q_imag = np.interp(dr_i, dr, Q[i].imag)

        Q_hat = Q_real+1j*Q_imag
        img += Q_hat*np.exp(-1j*k_c*dr_i)

    r0 = np.array([pos[npulses//2]]).T
    dr_i = norm(r0)-norm(r-r0, axis = 0)
    img = img*np.exp(1j*k_c*dr_i)
    img = np.reshape(img, [nv, nu])[::-1,:]
    return(img)
def DSBP(phs, platform, img_plane, center=None, size=None, derate = 1.05, taylor = 20, n = 32, beta = 4, cutoff = 'nyq', factor_max = 6, factor_min = 0):
##############################################################################
#                                                                            #
#  This is the Digital Spotlight Backprojection algorithm based on K. Dungan #
#  et. al.'s 2013 SPIE paper.                                                #
#                                                                            #
##############################################################################

    #Retrieve relevent parameters
    c           =   299792458.0
    pos         =   platform['pos']
    freq        =   platform['freq']
    u           =   img_plane['u']
    v           =   img_plane['v']
    du          =   img_plane['du']
    dv          =   img_plane['dv']
    p           =   img_plane['pixel_locs']

    #Derive parameters
    if center is None:
        empty_arg = True
        size = [0,0];
        size[1] = len(u)
        size[0] = len(v)
        Vx = u.max()-u.min()
        Vy = v.max()-v.min()
        center = np.mean(p, axis=-1)
        phs = reMoComp(phs, platform, center)
        pos = pos-center
    else:
        empty_arg=False
        Vx = size[1]*du
        Vy = size[0]*dv
        phs = reMoComp(phs, platform, center)
        pos = pos-center

    phsDS       = phs
    platformDS  = dict(platform)
    img_planeDS = dict(img_plane)

    #calculate decimation factor along range
    deltaF = abs(np.mean(np.diff(freq)))
    deltaFspot = c/(2*derate*norm([Vx, Vy]))
    N = int(np.floor(deltaFspot/deltaF))

    #force the decimation factor if specified by the user
    if N > factor_max:
        N = factor_max
    if N < factor_min:
        N = factor_min

    #decimate frequencies and phase history
    if N > 1:
        freq = decimate(freq, N, n = n, beta = beta, cutoff = cutoff)
        phsDS = decimate(phsDS, N, n = n, beta = beta, cutoff = cutoff)

    #update platform
    platformDS['nsamples'] = freq.size
    platformDS['freq']     = freq
    deltaF = freq[freq.size//2]-freq[freq.size//2-1] #Assume sample spacing can be determined by difference between last two values (first two are distorted by decimation filter)
    freq   = freq[freq.size//2]+np.arange(-freq.size//2,freq.size//2)*deltaF
    platformDS['k_r'] = 4*pi*freq//c

    #interpolate phs and pos using uniform azimuth spacing
    sph = cart2sph(pos)
    sph[:,0] = np.unwrap(sph[:,0])
    RPP = sph[1:,0]-sph[:-1,0]
    abs_RPP = abs(RPP)
    I = np.argsort(abs_RPP); sort_RPP = abs_RPP[I]
    im=I[4]; RPPdata = sort_RPP[4]#len(I)/2

    az_i = np.arange(sph[0,0], sph[-1,0], RPP[im])
    sph_i = np.zeros([az_i.size, 3])
    sph_i[:,0] = az_i
    sph_i[:,1:3] = interp1d(sph[:,0], sph[:,1:3], axis = 0)(sph_i[:,0])
    phsDS = interp1d(sph[:,0], phsDS, axis = 0)(sph_i[:,0])

    sph = sph_i
    pos = sph2cart(sph)

    #decimate slowtime positions and phase history
    fmax = freq[-1]
    PPRspot = derate*2*norm([Vx, Vy])*fmax*np.cos(sph[:,1].min())/c
    PPRdata = 1.0/RPPdata
    M = int(np.floor(PPRdata/PPRspot))

    #force the decimation factor if specified by the user
    if M > factor_max:
        M = factor_max
    if M < factor_min:
        M = factor_min

    if M > 1:
        FilterScale = np.array([decimate(np.ones(sph.shape[0]), M, n = n, beta = beta, cutoff = cutoff)]).T
        phsDS = decimate(phsDS, M, axis=0, n = n, beta = beta, cutoff = cutoff)/FilterScale
        sph = decimate(sph, M, axis=0, n = n, beta = beta, cutoff = cutoff)/FilterScale

    platformDS['npulses'] = int(phsDS.shape[0])
    platformDS['pos']     = sph2cart(sph)

    #Update platform
    if empty_arg:
        img_planeDS['pixel_locs']     = p-np.array([center]).T
    else:
        #Find cordinates of center pixel
        p = img_plane['pixel_locs'].T
        center_index = np.argsort(norm(p-center, axis = -1))[0]
        center_index = np.array(np.unravel_index(center_index, [v.size, u.size]))

        #Update u and v
        img_planeDS['u']    = np.arange(-size[1]//2,size[1]//2)*du
        img_planeDS['v']    = np.arange(-size[0]//2,size[0]//2)*dv

        #get pixel locs for sub_image
        u_index = np.arange(center_index[1]-size[1]//2,center_index[1]+size[1]//2)
        v_index = np.arange(center_index[0]-size[0]//2,center_index[0]+size[0]//2)
        uu,vv = np.meshgrid(u_index,v_index)
        locs_index = np.ravel_multi_index((vv.flatten(),uu.flatten()),(v.size,u.size))
        img_planeDS['pixel_locs']     = img_plane['pixel_locs'][:,locs_index]-np.array([center]).T

    #Backproject using spotlighted data
    img = backprojection(phsDS, platformDS, img_planeDS, taylor = taylor, prnt = False)

    return(img)
def reMoComp(phs, platform, center = np.array([0,0,0])):
##############################################################################
#                                                                            #
#  This is the re-motion compensation algorithm.  It re-motion compensates   #
#  the phase history to a new scene center.  The "center" argument is the    #
#  3D vector (in meters) that points to the new scene center using the old   #
#  scene center as the origin.                                               #
#                                                                            #
##############################################################################

    #Retrieve relevent parameters
    k_r         =   platform['k_r']
    pos         =   platform['pos']

    R0 = np.array([norm(pos, axis = -1)]).T
    RC = np.array([norm(pos-center, axis = -1)]).T
    dr = R0-RC
    remocomp = np.exp(-1j*k_r*dr)

    phs = phs*remocomp


    return(phs)
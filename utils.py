import tensorflow as tf
import numpy as np
import numba

def median_filter(df, window=3):
    '''
    applies a rolling window filter to our data frame
    '''
    return df.rolling(center=True,window=window).mean()

def sum_norm(fv):
    '''
    sum normalize the feature vector
    '''
    return fv/fv.sum(axis=1)[:,None]

@numba.njit(fastmath=True)
def morletConjFT(w, omega0):
    '''
    Equation C2 Berman et al. 2014
    '''
    return np.pi**(-1/4)*np.exp(-.5*(w-omega0)**2) 


def berman2014_wavelet(x, f, omega0, dt):
    '''
    Calculate the modified morlet wavelet from Berman et al. 2014
    On the GPU with tensorflow
    '''
    N = len(x)
    L = len(f)
            
    amp = np.zeros((L,N)) #make amplitude container
    if N % 2 == 1: #make the signal divisible by 2
        x = np.append(x, 0)
        N += 1
        test = True
    else:
        test = False

    s = x.shape
    if s[0]!=1:
        x = x.T
    x = np.concatenate([np.zeros(int(N/2)), x, np.zeros(int(N/2))]) #pad the signal for convolution
    M = N 
    N = len(x)
    scales = np.divide([omega0 + np.sqrt(2+omega0**2)],4*np.pi*f) #eq. C3
    Omegavals = np.divide([2*np.pi*np.arange(-N/2,N/2,1,dtype='int8')],N*dt)
    
    with tf.Graph().as_default():
        sess = tf.Session()
        x_tensor = tf.cast(x, dtype='complex32')
        xHat = tf.fft(x_tensor)

        if test == True:
            idx = np.arange((M-1)/2+1, M/2+M-1, 1, dtype='uint8')
        else:
            idx = np.arange((M-1)/2+1, M/2+M, 1, dtype='uint8')


        for i in range(0,L): 
            m = morletConjFT(-Omegavals*scales[i], omega0)

            #
            m_tensor = tf.cast(m, dtype='complex32')
            sig_tensor = tf.multiply(m_tensor,xHat)
            q = tf.ifft(sig_tensor)
            q = sess.run(q)

            q = q[:,idx]
            amp[i,:] = np.abs(q)*((np.pi**-.25)*np.exp(.25*(omega0-np.sqrt(omega0**2+2)**2)))/np.sqrt(2*scales[i]) #eq. C5 

        sess.close()
    return amp

def compute_wavelet(df, window=3, min_freq=1., sampling_rate=90, n_octaves=25)
    '''
    TODO: add doc-string
    '''
    #smooth and fill in na_values
    df = df.rolling(center=True,window=window).mean()
    df = df.bfill()
    df = df.ffill()

    #wavelet computation
    print('calculating berman_2014 wavelet')
    channels = [column for column in df]
    f = np.linspace(min_freq,sampling_rate/2,num=n_octaves)
    amps = []
    for chan in channels:
        print(chan)
        x = df[chan].as_matrix(columns=None)
        amp = berman2014_wavelet(x, f, omega0=5, dt=1/config.sampling_rate)
        amps.append(amp)
    fv = np.vstack(fv)
    fv = sum_norm(fv)
    return fv


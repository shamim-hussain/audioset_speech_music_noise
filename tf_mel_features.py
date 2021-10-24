
import tensorflow as tf
import tensorflow.signal as tfs
import numpy as np

@tf.function
def preemphasize(sig, coeff):
    return tf.concat([sig[...,0:1], sig[...,1:]-coeff*sig[...,:-1]], axis=-1)

@tf.function
def spectrogram(sig, fs=16000, winlen=.025, winstep=.010, 
                nfft=512, wfunc=None, preemph=0.97):
    
    sig=preemphasize(sig,preemph)
    wl_samp=round(winlen*fs)
    ws_samp=round(winstep*fs)
    
    siglen=tf.shape(sig)[-1]
    missing=(ws_samp-(siglen-wl_samp) % ws_samp)%ws_samp+tf.maximum(wl_samp-siglen,0)
    
    paddings=[(0,0)]*(sig.shape.rank-1)+[(0,missing)]
    sig=tf.pad(sig, paddings, 'CONSTANT')

    stft=tfs.stft(sig, wl_samp, ws_samp, nfft, wfunc)#, pad_end=True
    spec=(tf.abs(stft)**2)/nfft
    return spec


@tf.function
def log_spectrogram(sig=None, fs=16000, winlen=.025, winstep=.010, 
                    nfft=512, wfunc=None, preemph=0.97, spec=None):
    spec = spec if not spec is None else spectrogram(sig, fs, winlen, winstep,
                                                     nfft, wfunc, preemph)
    return tf.math.log(spec+1e-12)

@tf.function
def mfb(sig=None, fs=16000, nfilt=26, winlen=.025, winstep=.010,
         lowfreq=0, highfreq=None, nfft=512, wfunc=None, preemph=0.97, spec=None):
    highfreq=highfreq if not highfreq is None else fs//2
    spec=spec if not spec is None else spectrogram(sig, fs, winlen, winstep,
                                                   nfft, wfunc, preemph)
    
    bins=spec.shape[-1]
    melmat=tfs.linear_to_mel_weight_matrix(nfilt, bins, fs, 
                                           lowfreq, highfreq)
    mel_spec=tf.tensordot(spec, melmat,1)
    mel_spec.set_shape(spec.shape[:-1].concatenate(melmat.shape[-1:]))
    energy = tf.reduce_sum(spec, axis=-1)
    return mel_spec, energy

@tf.function
def log_mfb(sig=None, fs=16000, nfilt=26, winlen=.025, winstep=.010,
         lowfreq=0, highfreq=None, nfft=512, wfunc=None, preemph=0.97, 
            spec=None):
    highfreq=highfreq if not highfreq is None else fs//2
    mel_spec, energy=mfb(sig,fs,nfilt,winlen, winstep, lowfreq, highfreq,
                 nfft, wfunc, preemph, spec)
    log_mel_spec=tf.math.log(mel_spec+1e-12)
    return log_mel_spec,energy

@tf.function
def lifter(cepstra, L=22):
    if L > 0:
        ncoeff = cepstra.shape[-1].value
        n = tf.range(ncoeff, dtype=cepstra.dtype)
        lift = 1 + (L/2.)*tf.sin(3.14159265358979*n/L)
        return lift*cepstra
    else:
        return cepstra

@tf.function
def mfcc(sig=None, fs=16000, numcep=13, nfilt=26, winlen=.025, winstep=.010,
         lowfreq=0, highfreq=None, nfft=512, wfunc=None, preemph=0.97, ceplifter=22,include_energy=True,
         log_mel_spec=None, energy=None):
    highfreq=highfreq if not highfreq is None else fs//2
    lmfb, energy=(log_mel_spec, energy) if not log_mel_spec is None else log_mfb(sig,fs,nfilt,winlen, winstep, 
                                 lowfreq, highfreq, nfft, wfunc, preemph)
    mfc=tfs.mfccs_from_log_mel_spectrograms(lmfb)[...,:numcep]
    mfc=lifter(mfc,ceplifter)
    if include_energy:
        mfc=tf.concat([tf.log(energy+1e-12)[...,None],mfc[...,1:]],axis=-1)
    return mfc


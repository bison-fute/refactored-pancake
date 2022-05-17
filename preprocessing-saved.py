            
# downsample step at the end
class Preprocessing:
    """
    desc:
    """
    def __init__(self, train_set = None, validation_set = None, test_set = None):
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.datasets = [self.train_set, self.validation_set, self.test_set]
        self.labels = sorted(list(set(datapoint[2] for datapoint in self.train_set)))

        
    def cepstrum(self, input, nceps=13):
        """
        desc: calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform
        inputs:
            input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
                   number of frames and nmelfilters the length of the filterbank
            nceps: number of output cepstral coefficients
        outputs:
            array of Cepstral coefficients [N x nceps]
        """
        out_dct = dct(input, n=None) # n affects the lenght of the transform => diff. results
        out = self.lifter(out_dct)
        return np.array(out_dct[:,0:13])
        
        
    def enframe(self, samples, winlen=400, winshift=200):
        """
        desc: slices the input samples into overlapping windows.
        inputs:
            winlen: window length in samples.
            winshift: shift of consecutive windows in samples
        returns:
            numpy array [N x winlen], where N is the number of windows that fit
            in the input signal
        """
        out, i = [], 0
        # test that both sides of the win. are inside the signal
        while (i < len(samples)) and (i+int(winlen) < len(samples)):
            out.append(samples[i:i+int(winlen)])
            i += int(winshift)
        return np.vstack(out)

    
    def lifter(self, mfcc, lifter=22):
        """
        desc: applies liftering to improve the relative range of MFCC coefficients.
        inputs:
           mfcc: NxM matrix where N is the number of frames and M the number of MFCC coefficients
           lifter: lifering coefficient
        returns:
           NxM array with lifeterd coefficients
        """
        nframes, nceps = mfcc.shape
        cepwin = 1.0 + lifter/2.0 * np.sin(np.pi * np.arange(nceps) / lifter)
        return np.multiply(mfcc, np.tile(cepwin, nframes).reshape((nframes,nceps)))
    
    
    def logMelSpectrum(self, input, samplingrate):
        """
        desc: calculates the log output of a Mel filterbank when the input is the power spectrum
        inputs:
            input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
                   nfft the length of each spectrum
            samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
        returns:
            array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
            of filters in the filterbank
        """
        nfft = input.shape[1]
        mel_filters = self.trfbank(samplingrate, nfft)
        out = input@mel_filters.T
        return np.log(out)


    def mspec(self, waveform, samplingrate=16000, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512):
        """
        desc: computes Mel Filterbank features. See lab 1. Conserving same default parameters as waveforms' sampling rate & length are similar
        inputs:
            wave: array of speech samples with shape (N,)
            winlen: lenght of the analysis window
            winshift: number of samples to shift the analysis window at every time step
            preempcoeff: pre-emphasis coefficient
            nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
            samplingrate: sampling rate of the original signal
        returns:
            N x nfilters array with mel filterbank features (see trfbank for nfilters)
        """
        frames = self.enframe(waveform, winlen, winshift)
        preemph = self.preemp(frames, preempcoeff)
        windowed = self.windowing(preemph)
        spec = self.powerSpectrum(windowed, nfft)
        return self.logMelSpectrum(spec, samplingrate)
    
    
    def mfcc(self, mspecs,  nceps=13, liftercoeff=22):
        """
        desc: computes Mel Frequency Cepstrum Coefficients.
        inputs:
            samples: array of speech samples with shape (N,)
            winlen: lenght of the analysis window
            winshift: number of samples to shift the analysis window at every time step
            preempcoeff: pre-emphasis coefficient
            nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
            nceps: number of cepstrum coefficients to compute
            samplingrate: sampling rate of the original signal
            liftercoeff: liftering coefficient used to equalise scale of MFCCs
        returns:
            N x nceps array with lifetered MFCC coefficients
        """
        ceps = self.cepstrum(mspecs, nceps)
        return self.lifter(ceps, liftercoeff)
    
    
    def powerSpectrum(self, input, nfft=512):
        """
        desc: calculates the power spectrum of the input signal, that is the square of the modulus of the FFT
        inputs:
            input: array of speech samples [N x M] where N is the number of frames and
                   M the samples per frame
            nfft: length of the FFT
        output:
            array of power spectra [N x nfft]
        """
        out_fft = fft(input, nfft, axis=1)
        return np.square(np.abs(out_fft))
    
    
    def preemp(self, input, p=0.97):
        """
        desc: pre-emphasis filter.
        inputs:
            input: array of speech frames [N x M] where N is the number of frames and
                   M the samples per frame
            p: preemhasis factor (defaults to the value specified in the exercise)
        returns:
            output: array of pre-emphasised speech samples
        """
        b = [1, -p]
        a = [1]
        out = lfilter(b, a, input, zi=None)
        return out
    
    
    def process_audio(self, audio):
        """
        desc: wrapper, compute waveform lmfs and lmfcc
        input:
            audio: one sampple of dataset
        return:
            same format as input array with added features (lmfs and lmfcc) in first element
        """
        waveform = audio
        lmfs = self.mspec(waveform, samplingrate = 16000, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512)
        lmfcc = self.mfcc(lmfs, nceps=13, liftercoeff=22)
        return lmfcc
        #return [waveform, lmfs, lmfcc], sampling_rate, label, speaker_id, utterance_number
        
 
    def process_dataset(self, dataset):
        """
        desc: iterate function process audio over whole dataset
        input:
            dataset
        return:
            processed dataset list of processed audios
        """
        processed_dataset = []
        for audio in dataset:
            processed_audio = self.process_audio(audio)
            processed_dataset.append(processed_audio)  
        return processed_dataset
            
    
    def trfbank(self, fs, nfft=512, lowfreq=133.33, linsc=200/3., logsc=1.0711703, 
                nlinfilt=13, nlogfilt=27, equalareas=False):
        """
        desc: compute triangular filterbank for MFCC computation.
        inputs:
            fs:         sampling frequency (rate)
            nfft:       length of the fft
            lowfreq:    frequency of the lowest filter
            linsc:      scale for the linear filters
            logsc:      scale for the logaritmic filters
            nlinfilt:   number of linear filters
            nlogfilt:   number of log filters
        returns:
            res:  array with shape [N, nfft], with filter amplitudes for each column.
        """
        # Total number of filters
        nfilt = nlinfilt + nlogfilt

        #------------------------
        # Compute the filter bank
        #------------------------
        # Compute start/middle/end points of the triangular filters in spectral
        # domain
        freqs = np.zeros(nfilt+2)
        freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
        freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
        if equalareas:
            heights = np.ones(nfilt)
        else:
            heights = 2./(freqs[2:] - freqs[0:-2])

        # Compute filterbank coeff (in fft domain, in bins)
        fbank = np.zeros((nfilt, nfft))
        # FFT bins (in Hz)
        nfreqs = np.arange(nfft) / (1. * nfft) * fs
        for i in range(nfilt):
            low = freqs[i]
            cen = freqs[i+1]
            hi = freqs[i+2]

            lid = np.arange(np.floor(low * nfft / fs) + 1,
                            np.floor(cen * nfft / fs) + 1, dtype=np.int)
            lslope = heights[i] / (cen - low)
            rid = np.arange(np.floor(cen * nfft / fs) + 1,
                            np.floor(hi * nfft / fs) + 1, dtype=np.int)
            rslope = heights[i] / (hi - cen)
            fbank[i][lid] = lslope * (nfreqs[lid] - low)
            fbank[i][rid] = rslope * (hi - nfreqs[rid])

        return fbank
    
    
    def windowing(self, input):
        """
        desc: applies hamming window to the input frames.
        inputs:
            input: array of speech samples [N x M] where N is the number of frames and
                   M the samples per frame
        returns:
            array of windoed speech samples [N x M]
        """
        win = hamming(input.shape[1], sym=0)
        out = []
        for frame in input:
            out.append(frame * win)
        return np.vstack(out)
            
            
            
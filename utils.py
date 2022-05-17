from libraries import *





def downsample(dataset, new_sr=8000):
    """ 
    desc: downsample audio for faster audio preprocessing, losing a bit of information
    inputs:
        dataset: dataset to be down sampled
        new_sr: new sampling rate
    return:
        transformed_dataset: array of transformed inputs
    """
    transformed_dataset = []
    for waveform, sample_rate, label, speeker_id, utterance_number in tqdm(dataset):
        # transform = Resample(orig_freq=sample_rate, new_freq=new_sr)
        # new_waveform = transform(waveform)
        # transformed = new_waveform, new_sr, label, speeker_id, utterance_number
        waveform = (waveform - waveform.mean())/waveform.std()
        transformed = waveform, sample_rate, label, speeker_id, utterance_number
        transformed_dataset.append(transformed)
    return transformed_dataset


def enframe(samples, winlen=400, winshift=200):
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


def pad_sequence(batch):
    """ 
    desc: make all tensor in a batch the same length as the longest sequence by padding with zeros 
    """
    batch = [torch.from_numpy(item).t() for item in batch]
    print(np.array(batch).shape)
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)  # in each tensor, target first, tensor second

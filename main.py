# import libraries
import IPython.display as ipd
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

# check if a CUDA GPU available and select it to run code on it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SubsetSC(SPEECHCOMMANDS):
    """
    desc: create a subclass that splits it into standard training, validation, testing subsets.
    """
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# create training and validation split
train_set = SubsetSC("training")
validation_set = SubsetSC("validation")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]


# dispprint(f"label_last: {label_last}")lay information on the waveform
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))
plt.plot(waveform.t().numpy());

# display the list of labels available in the dataset
labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
print(labels)


# first samples of train_set saying "backward"
waveform_first, _, label_first, *_ = train_set[0]
print(f"label_first: {label_first}")
# display(ipd.Audio(waveform_first.numpy(), rate=sample_rate))
waveform_second, _, label_second, *_ = train_set[1]
print(f"label_second: {label_second}")
# display(ipd.Audio(waveform_second.numpy(), rate=sample_rate))

# last sample of train_set saying "visual"
waveform_last, _, label_last, *_ = train_set[-1]
print(f"label_last: {label_last}")

# downsample audio for faster processing, losing a bit of information
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)
# display(ipd.Audio(transformed.numpy(), rate=new_sample_rate))
print(transformed)
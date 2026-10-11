# EEG Motor Imagery Classification using Deep Learning

Classifying imagined hand, feet, and tongue movements from raw EEG signals
using a convolutional neural networ.

---

## The problem

When you imagine moving your hand without actually moving it, your brain fires
the same motor circuits it would for real movement. This produces measurable
electrical patterns on the scalp — specifically a suppression of activity in the
mu (8–12 Hz) and beta (13–30 Hz) bands over the motor cortex, known as
event-related desynchronization (ERD).

For patients with paralysis or ALS, this signal is often intact even when no
muscle movement is possible. BCIs aim to decode it in real time and translate
it into device control. The hard part is not detecting the signal — it is
translating it reliably across different people, whose ERD patterns vary
significantly.

---

## Dataset

**BCI Competition IV Dataset 2a**
- 9 subjects performing motor imagery tasks
- 22 EEG channels, 250 Hz sampling rate
- 4 classes: Left Hand, Right Hand, Feet, Tongue
- 288 trials per subject per session

---

## Pipeline

Preprocessing is done with MNE-Python. The STFT converts each trial into a
time-frequency spectrogram per channel, giving the CNN a (22, freqs, time)
input that captures both spatial and spectral structure.

---

## Model

A two-block 2D CNN with double convolutions per block, batch normalisation,
ELU activations, and dropout. The 22 EEG channels act as the input channel
dimension, and the frequency × time spectrogram is the spatial input.

---

## Results

There is a gap between binary and multi-class performance which is expected — left vs right hand imagery produces a clean contralateral ERD pattern that the model can
learn. Feet and tongue are harder to separate, particularly in a cross-subject
setting where between-subject variability dominates.
`
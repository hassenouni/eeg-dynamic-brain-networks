# Dynamic graph sequences

`dynamic_connectivity` computes Pearson correlation or PLV in complete sliding
windows. It returns matrices, actual rounded window duration, starts and centers.
The incomplete tail is dropped. Graph changes use the Frobenius norm of consecutive
matrix differences after clearing diagonals, aligned to the later window center.

Overlapping windows introduce temporal dependence. Window length and stride are
analysis parameters, not validated clinical defaults. PLV requires narrowband
inputs as in the connectivity package. These dynamic graphs are descriptive:
there is no temporal graph neural network and no seizure prediction model here.
For machine learning, split by patient before constructing graph sequences.

## References

- [Method or implementation reference](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
- [CHB-MIT 1.0.0](https://physionet.org/content/chbmit/1.0.0/)
- [Siena 1.0.0](https://physionet.org/content/siena-scalp-eeg/1.0.0/)

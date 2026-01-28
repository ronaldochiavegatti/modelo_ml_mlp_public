#ifndef MFCC_H
#define MFCC_H

#include <stddef.h>

typedef struct {
    int sample_rate;
    int fft_size;
    int n_mels;
    int n_mfcc;
    int n_bins;
    float *mel_filters;
} MfccBank;

// Allocate and precompute mel filterbank weights.
int mfcc_init(MfccBank *bank, int sample_rate, int fft_size, int n_mels, int n_mfcc);
// Free internal buffers for a bank.
void mfcc_free(MfccBank *bank);
// Compute MFCCs from a power spectrum.
void mfcc_compute(const MfccBank *bank, const float *power_spectrum, float *mfcc_out);

#endif

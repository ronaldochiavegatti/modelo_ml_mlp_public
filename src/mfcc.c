// MFCC computation (mel filterbank + log energies + DCT).
#include "mfcc.h"

#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

int mfcc_init(MfccBank *bank, int sample_rate, int fft_size, int n_mels, int n_mfcc) {
    if (!bank || sample_rate <= 0 || fft_size <= 0 || n_mels <= 0 || n_mfcc <= 0) {
        return -1;
    }

    bank->sample_rate = sample_rate;
    bank->fft_size = fft_size;
    bank->n_mels = n_mels;
    bank->n_mfcc = n_mfcc;
    bank->n_bins = fft_size / 2 + 1;
    bank->mel_filters = (float *)calloc((size_t)n_mels * (size_t)bank->n_bins, sizeof(float));
    if (!bank->mel_filters) {
        return -1;
    }

    // Compute mel filterbank center frequencies and triangular weights.
    float low_mel = hz_to_mel(0.0f);
    float high_mel = hz_to_mel((float)sample_rate / 2.0f);
    float mel_step = (high_mel - low_mel) / (float)(n_mels + 1);

    float *mel_points = (float *)malloc(sizeof(float) * (size_t)(n_mels + 2));
    int *bins = (int *)malloc(sizeof(int) * (size_t)(n_mels + 2));
    if (!mel_points || !bins) {
        free(mel_points);
        free(bins);
        free(bank->mel_filters);
        bank->mel_filters = NULL;
        return -1;
    }

    for (int i = 0; i < n_mels + 2; i++) {
        float mel = low_mel + mel_step * (float)i;
        float hz = mel_to_hz(mel);
        int bin = (int)floorf((float)(fft_size + 1) * hz / (float)sample_rate);
        if (bin < 0) bin = 0;
        if (bin >= bank->n_bins) bin = bank->n_bins - 1;
        mel_points[i] = mel;
        bins[i] = bin;
    }

    // Triangular filters across mel-spaced bins.
    for (int m = 1; m <= n_mels; m++) {
        int left = bins[m - 1];
        int center = bins[m];
        int right = bins[m + 1];
        for (int k = left; k < center; k++) {
            float weight = (center == left) ? 0.0f : (float)(k - left) / (float)(center - left);
            bank->mel_filters[(m - 1) * bank->n_bins + k] = weight;
        }
        for (int k = center; k < right; k++) {
            float weight = (right == center) ? 0.0f : (float)(right - k) / (float)(right - center);
            bank->mel_filters[(m - 1) * bank->n_bins + k] = weight;
        }
    }

    free(mel_points);
    free(bins);
    return 0;
}

void mfcc_free(MfccBank *bank) {
    if (!bank) {
        return;
    }
    free(bank->mel_filters);
    bank->mel_filters = NULL;
}

void mfcc_compute(const MfccBank *bank, const float *power_spectrum, float *mfcc_out) {
    const float eps = 1e-10f;
    float *log_energies;

    if (!bank || !power_spectrum || !mfcc_out) {
        return;
    }

    log_energies = (float *)malloc(sizeof(float) * (size_t)bank->n_mels);
    if (!log_energies) {
        return;
    }

    // Apply mel filters to get log energies.
    for (int m = 0; m < bank->n_mels; m++) {
        double sum = 0.0;
        const float *filter = &bank->mel_filters[m * bank->n_bins];
        for (int k = 0; k < bank->n_bins; k++) {
            sum += (double)power_spectrum[k] * (double)filter[k];
        }
        if (sum < eps) {
            sum = eps;
        }
        log_energies[m] = logf((float)sum);
    }

    // DCT-II of the log-mel energies.
    for (int k = 0; k < bank->n_mfcc; k++) {
        double acc = 0.0;
        for (int m = 0; m < bank->n_mels; m++) {
            acc += (double)log_energies[m] * cos((double)M_PI * (double)k * ((double)m + 0.5) / (double)bank->n_mels);
        }
        mfcc_out[k] = (float)acc;
    }

    free(log_energies);
}

// Lightweight DSP helpers (windowing + radix-2 FFT).
#include "dsp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

size_t next_pow2(size_t n) {
    size_t p = 1;
    // Grow by powers of two until p >= n.
    while (p < n) {
        p <<= 1;
    }
    return p;
}

void hamming_window(float *win, size_t n) {
    if (!win || n == 0) {
        return;
    }
    // Classic Hamming coefficients for spectral leakage reduction.
    for (size_t i = 0; i < n; i++) {
        win[i] = 0.54f - 0.46f * cosf((float)(2.0 * M_PI * i) / (float)(n - 1));
    }
}

static void bit_reverse(float *real, float *imag, size_t n) {
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < j) {
            float tmp = real[i];
            real[i] = real[j];
            real[j] = tmp;
            tmp = imag[i];
            imag[i] = imag[j];
            imag[j] = tmp;
        }
        // Flip bits to generate the bit-reversed index sequence.
        size_t bit = n >> 1;
        while (bit > 0 && (j & bit)) {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
    }
}

static void fft_inplace(float *real, float *imag, size_t n) {
    // Iterative Cooley-Tukey radix-2 FFT.
    for (size_t len = 2; len <= n; len <<= 1) {
        float ang = (float)(-2.0 * M_PI / (double)len);
        float wlen_real = cosf(ang);
        float wlen_imag = sinf(ang);
        for (size_t i = 0; i < n; i += len) {
            float w_real = 1.0f;
            float w_imag = 0.0f;
            for (size_t j = 0; j < len / 2; j++) {
                size_t u = i + j;
                size_t v = i + j + len / 2;
                float vr = real[v] * w_real - imag[v] * w_imag;
                float vi = real[v] * w_imag + imag[v] * w_real;
                float ur = real[u];
                float ui = imag[u];
                real[u] = ur + vr;
                imag[u] = ui + vi;
                real[v] = ur - vr;
                imag[v] = ui - vi;
                float next_w_real = w_real * wlen_real - w_imag * wlen_imag;
                float next_w_imag = w_real * wlen_imag + w_imag * wlen_real;
                w_real = next_w_real;
                w_imag = next_w_imag;
            }
        }
    }
}

void fft_real(const float *input, size_t n, float *real, float *imag) {
    if (!input || !real || !imag || n == 0) {
        return;
    }
    for (size_t i = 0; i < n; i++) {
        real[i] = input[i];
        imag[i] = 0.0f;
    }
    bit_reverse(real, imag, n);
    fft_inplace(real, imag, n);
}

void magnitude_spectrum(const float *real, const float *imag, size_t n, float *mag_out) {
    size_t half = n / 2 + 1;
    for (size_t i = 0; i < half; i++) {
        float re = real[i];
        float im = imag[i];
        mag_out[i] = sqrtf(re * re + im * im);
    }
}

// Simple autocorrelation-based pitch detection.
// Returns F0 in Hz, or 0.0 if unvoiced.
// Also returns the normalized ACF peak value as 'harmonicity'.
float compute_f0(const float *frame, size_t frame_len, int sample_rate, float *harmonicity) {
    const float min_f0 = 75.0f;  // Min typical human F0
    const float max_f0 = 500.0f; // Max typical human F0
    const float voicing_threshold = 0.85f; // ACF peak threshold for voicing

    if (!frame || frame_len == 0) {
        if (harmonicity) *harmonicity = 0.0f;
        return 0.0f;
    }

    size_t min_lag = (size_t)((float)sample_rate / max_f0);
    size_t max_lag = (size_t)((float)sample_rate / min_f0);
    if (max_lag >= frame_len) {
        max_lag = frame_len - 1;
    }

    float best_lag_val = -1.0f;
    size_t best_lag = 0;

    // Energy at lag 0 for normalization
    double energy = 0.0;
    for (size_t i = 0; i < frame_len; i++) {
        energy += frame[i] * frame[i];
    }
    if (energy < 1e-6) {
        if (harmonicity) *harmonicity = 0.0f;
        return 0.0f;
    }

    // Compute autocorrelation for the valid lag range
    for (size_t lag = min_lag; lag <= max_lag; lag++) {
        double sum = 0.0;
        for (size_t i = 0; i < frame_len - lag; i++) {
            sum += frame[i] * frame[i + lag];
        }
        float normalized_sum = (float)(sum / energy);
        if (normalized_sum > best_lag_val) {
            best_lag_val = normalized_sum;
            best_lag = lag;
        }
    }

    if (harmonicity) {
        *harmonicity = best_lag_val;
    }

    // Check if the frame is voiced based on the strength of the ACF peak
    if (best_lag_val > voicing_threshold && best_lag > 0) {
        // NOTE: This simple method can be prone to octave errors (e.g., finding half
        // or double the true F0). More advanced algorithms like YIN or pYIN provide
        // better accuracy but are more complex to implement.
        return (float)sample_rate / (float)best_lag;
    }

    // Unvoiced
    return 0.0f;
}

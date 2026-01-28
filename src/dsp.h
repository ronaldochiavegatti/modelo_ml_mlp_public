#ifndef DSP_H
#define DSP_H

#include <stddef.h>

void hamming_window(float *out, size_t len);
size_t next_pow2(size_t n);
void fft_real(const float *in, size_t n, float *real, float *imag);
void magnitude_spectrum(const float *real, const float *imag, size_t n, float *out);
float compute_f0(const float *frame, size_t frame_len, int sample_rate, float *harmonicity);

#endif // DSP_H

// Extract per-file audio features (MFCC + simple spectral stats).
#include "csv.h"
#include "dsp.h"
#include "frame_io.h"
#include "mfcc.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --input processed --metadata metadata.csv --output features.csv [--n-mfcc 13] [--n-mels 26] [--rolloff 0.85]\n",
            prog);
}

static int header_index(const CsvRow *row, const char *name) {
    for (size_t i = 0; i < row->count; i++) {
        if (strcasecmp(row->fields[i], name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static float compute_rms(const float *x, size_t len) {
    double sum = 0.0;
    if (!x || len == 0) {
        return 0.0f;
    }
    for (size_t i = 0; i < len; i++) {
        sum += x[i] * x[i];
    }
    return (float)sqrt(sum / (double)len);
}

static float compute_zcr(const float *x, size_t len) {
    size_t count = 0;
    if (!x || len < 2) {
        return 0.0f;
    }
    for (size_t i = 1; i < len; i++) {
        if ((x[i - 1] >= 0.0f && x[i] < 0.0f) || (x[i - 1] < 0.0f && x[i] >= 0.0f)) {
            count++;
        }
    }
    return (float)count / (float)(len - 1);
}

static float compute_crest_factor(const float *frame, size_t frame_len, float rms) {
    if (rms < 1e-9) {
        return 0.0f;
    }
    float peak = 0.0f;
    for (size_t i = 0; i < frame_len; i++) {
        float val = fabsf(frame[i]);
        if (val > peak) {
            peak = val;
        }
    }
    return peak / rms;
}

static float compute_spectral_flux(const float *mag, const float *prev_mag, size_t n_bins) {
    if (!mag || !prev_mag || n_bins == 0) {
        return 0.0f;
    }
    double sum_sq = 0.0;
    for (size_t k = 0; k < n_bins; k++) {
        double diff = (double)mag[k] - (double)prev_mag[k];
        sum_sq += diff * diff;
    }
    return (float)sqrt(sum_sq / (double)n_bins);
}

static void compute_spectral_features(const float *mag, size_t n_bins, int sample_rate, int fft_size,
                                      float rolloff_pct, float *centroid, float *rolloff,
                                      float *bandwidth, float *flatness) {
    double mag_sum = 0.0;
    double weighted_sum = 0.0;
    double log_sum = 0.0;

    if (n_bins == 0) {
        *centroid = 0; *rolloff = 0; *bandwidth = 0; *flatness = 0;
        return;
    }

    for (size_t k = 0; k < n_bins; k++) {
        float freq = (float)k * (float)sample_rate / (float)fft_size;
        mag_sum += mag[k];
        weighted_sum += (double)mag[k] * (double)freq;
        log_sum += log((double)mag[k] + 1e-9); // Epsilon for numerical stability
    }

    if (mag_sum > 1e-9) {
        *centroid = (float)(weighted_sum / mag_sum);
        // Flatness = Geometric Mean / Arithmetic Mean
        double arith_mean = mag_sum / (double)n_bins;
        double geo_mean = exp(log_sum / (double)n_bins);
        *flatness = (float)(geo_mean / arith_mean);

        // Bandwidth
        double weighted_sq_diff = 0.0;
        for (size_t k = 0; k < n_bins; k++) {
            float freq = (float)k * (float)sample_rate / (float)fft_size;
            weighted_sq_diff += ((double)freq - *centroid) * ((double)freq - *centroid) * mag[k];
        }
        *bandwidth = (float)sqrt(weighted_sq_diff / mag_sum);

    } else {
        *centroid = 0.0f;
        *flatness = 0.0f;
        *bandwidth = 0.0f;
    }

    // Rolloff
    double cumulative = 0.0;
    double target = mag_sum * (double)rolloff_pct;
    float rolloff_freq = 0.0f;
    for (size_t k = 0; k < n_bins; k++) {
        cumulative += mag[k];
        if (cumulative >= target) {
            rolloff_freq = (float)k * (float)sample_rate / (float)fft_size;
            break;
        }
    }
    *rolloff = rolloff_freq;
}

// A structure to hold computed statistics for a feature.
typedef struct {
    double mean;
    double std;
    double skewness;
    double kurtosis;
} FeatureStats;

// Computes mean, std, skewness, and kurtosis for a series of values.
static FeatureStats compute_stats(const float *values, size_t count) {
    FeatureStats stats = {0.0, 0.0, 0.0, 0.0};
    if (count < 2) { // Cannot compute std, skew, kurtosis for less than 2 values
        if (count == 1) stats.mean = values[0];
        return stats;
    }

    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
    }
    stats.mean = sum / (double)count;

    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = values[i] - stats.mean;
        sum_sq_diff += diff * diff;
    }
    // Use n-1 for sample standard deviation
    double variance = sum_sq_diff / (double)(count - 1);
    stats.std = sqrt(variance);

    if (stats.std > 1e-9) {
        double m3 = 0.0;
        double m4 = 0.0;
        for (size_t i = 0; i < count; i++) {
            double diff = values[i] - stats.mean;
            m3 += diff * diff * diff;
            m4 += diff * diff * diff * diff;
        }
        m3 /= (double)count;
        m4 /= (double)count;
        
        double std_pow3 = stats.std * stats.std * stats.std;
        double std_pow4 = std_pow3 * stats.std;

        stats.skewness = m3 / std_pow3;
        stats.kurtosis = (m4 / std_pow4) - 3.0; // Excess kurtosis
    }

    return stats;
}

// Computes delta coefficients for a set of features.
static void compute_deltas(const float *features, float *deltas, size_t num_frames, size_t num_features) {
    if (num_frames < 2) {
        memset(deltas, 0, sizeof(float) * num_frames * num_features);
        return;
    }
    for (size_t k = 0; k < num_features; k++) {
        deltas[0 * num_features + k] = features[1 * num_features + k] - features[0 * num_features + k];
        for (size_t f = 1; f < num_frames - 1; f++) {
            deltas[f * num_features + k] = features[(f + 1) * num_features + k] - features[(f - 1) * num_features + k];
        }
        deltas[(num_frames - 1) * num_features + k] = features[(num_frames - 1) * num_features + k] - features[(num_frames - 2) * num_features + k];
    }
}


int main(int argc, char **argv) {
    const char *input_dir = NULL;
    const char *metadata_path = NULL;
    const char *output_path = NULL;
    int n_mfcc = 13;
    int n_mels = 26;
    float rolloff_pct = 0.85f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) input_dir = argv[++i];
        else if (strcmp(argv[i], "--metadata") == 0 && i + 1 < argc) metadata_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "--n-mfcc") == 0 && i + 1 < argc) n_mfcc = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-mels") == 0 && i + 1 < argc) n_mels = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rolloff") == 0 && i + 1 < argc) rolloff_pct = (float)atof(argv[++i]);
        else { print_usage(argv[0]); return 1; }
    }

    if (!input_dir || !metadata_path || !output_path) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stdout, "Extract features: input=%s metadata=%s output=%s n_mfcc=%d n_mels=%d rolloff=%.2f\n",
            input_dir, metadata_path, output_path, n_mfcc, n_mels, rolloff_pct);

    FILE *meta = fopen(metadata_path, "r");
    if (!meta) { perror("fopen metadata"); return 1; }
    FILE *out = fopen(output_path, "w");
    if (!out) { perror("fopen output"); fclose(meta); return 1; }

    char line[8192];
    if (!fgets(line, sizeof(line), meta)) {
        fprintf(stderr, "Empty metadata file\n");
        fclose(meta); fclose(out); return 1;
    }

    CsvRow header = {0};
    csv_split_line(line, &header);
    int id_idx = header_index(&header, "id");
    int class_idx = header_index(&header, "classe");
    csv_free_row(&header);

    if (id_idx < 0 || class_idx < 0) {
        fprintf(stderr, "Missing required columns in metadata: id, classe\n");
        fclose(meta); fclose(out); return 1;
    }

    int n_static_other = 8; // rms, zcr, crest, centroid, rolloff, bandwidth, flatness, flux
    int n_static_feats = n_mfcc + n_static_other;
    
    char mfcc_names[n_mfcc][16];
    for (int i = 0; i < n_mfcc; i++) snprintf(mfcc_names[i], 16, "mfcc%02d", i + 1);
    
    const char *other_feat_names[] = {"rms", "zcr", "crest", "centroid", "rolloff", "bandwidth", "flatness", "flux"};
    const char *f0_feat_names[] = {"f0_mean", "f0_std", "voicing_rate", "harmonicity_mean", "harmonicity_std"};

    fprintf(out, "id,classe");
    const char *stats_names[] = {"mean", "std", "skew", "kurt"};
    for (int s = 0; s < 4; s++) {
        for (int i = 0; i < n_mfcc; i++) fprintf(out, ",%s_%s", mfcc_names[i], stats_names[s]);
        for (int i = 0; i < n_mfcc; i++) fprintf(out, ",%s_delta_%s", mfcc_names[i], stats_names[s]);
        for (int i = 0; i < n_mfcc; i++) fprintf(out, ",%s_delta2_%s", mfcc_names[i], stats_names[s]);
        for (int i = 0; i < n_static_other; i++) fprintf(out, ",%s_%s", other_feat_names[i], stats_names[s]);
    }
    for (int i = 0; i < 5; i++) fprintf(out, ",%s", f0_feat_names[i]);
    fprintf(out, "\n");

    size_t processed = 0, missing = 0, failed = 0;

    while (fgets(line, sizeof(line), meta)) {
        CsvRow row = {0};
        char frames_path[1024];
        const char *id, *class_name;

        if (csv_split_line(line, &row) != 0 || row.count == 0 || (size_t)id_idx >= row.count || (size_t)class_idx >= row.count) {
            csv_free_row(&row); continue;
        }
        id = row.fields[id_idx];
        class_name = row.fields[class_idx];
        if (!id || !*id || !class_name || !*class_name) {
            csv_free_row(&row); continue;
        }

        snprintf(frames_path, sizeof(frames_path), "%s/%s/%s.frames", input_dir, class_name, id);
        if (!file_exists(frames_path)) {
            missing++; csv_free_row(&row); continue;
        }

        FrameHeader fh;
        float *frames = NULL;
        if (frames_read_all(frames_path, &fh, &frames) != 0) {
            failed++; csv_free_row(&row); continue;
        }

        size_t frame_len = fh.frame_len;
        size_t num_frames = fh.num_frames;
        if (num_frames < 3) {
            free(frames); csv_free_row(&row); continue;
        }
        size_t fft_size = next_pow2(frame_len);
        size_t n_bins = fft_size / 2 + 1;

        float *static_features = (float *)malloc(num_frames * n_static_feats * sizeof(float));
        float *delta_mfccs = (float *)malloc(num_frames * n_mfcc * sizeof(float));
        float *delta2_mfccs = (float *)malloc(num_frames * n_mfcc * sizeof(float));
        float *f0_values = (float *)malloc(num_frames * sizeof(float));
        float *harmonicity_values = (float *)malloc(num_frames * sizeof(float));
        int *voiced_flags = (int *)malloc(num_frames * sizeof(int));
        float *window = (float *)malloc(sizeof(float) * frame_len);
        float *fft_in = (float *)calloc(fft_size, sizeof(float));
        float *real = (float *)malloc(sizeof(float) * fft_size);
        float *imag = (float *)malloc(sizeof(float) * fft_size);
        float *mag = (float *)malloc(sizeof(float) * n_bins);
        float *power = (float *)malloc(sizeof(float) * n_bins);
        float *mfcc_buffer = (float *)malloc(sizeof(float) * (size_t)n_mfcc);
        float *prev_mag = (float *)calloc(n_bins, sizeof(float));

        if (!static_features || !delta_mfccs || !delta2_mfccs || !f0_values || !harmonicity_values || !voiced_flags || !window || !fft_in || !real || !imag || !mag || !power || !mfcc_buffer || !prev_mag) {
            failed++; csv_free_row(&row); continue; // Simplified cleanup for brevity
        }

        hamming_window(window, frame_len);
        MfccBank bank;
        mfcc_init(&bank, (int)fh.sample_rate, (int)fft_size, n_mels, n_mfcc);

        for (size_t f = 0; f < num_frames; f++) {
            const float *frame = &frames[f * frame_len];
            
            float harmonicity = 0.0f;
            float f0 = compute_f0(frame, frame_len, (int)fh.sample_rate, &harmonicity);
            f0_values[f] = f0;
            harmonicity_values[f] = harmonicity;
            voiced_flags[f] = (f0 > 0.0f) ? 1 : 0;

            float rms = compute_rms(frame, frame_len);
            float zcr = compute_zcr(frame, frame_len);
            float crest = compute_crest_factor(frame, frame_len, rms);

            for (size_t i = 0; i < frame_len; i++) fft_in[i] = frame[i] * window[i];
            for (size_t i = frame_len; i < fft_size; i++) fft_in[i] = 0.0f;
            fft_real(fft_in, fft_size, real, imag);
            magnitude_spectrum(real, imag, fft_size, mag);
            
            float centroid=0, rolloff=0, bandwidth=0, flatness=0, flux=0;
            compute_spectral_features(mag, n_bins, (int)fh.sample_rate, (int)fft_size, rolloff_pct, &centroid, &rolloff, &bandwidth, &flatness);
            if (f > 0) {
                flux = compute_spectral_flux(mag, prev_mag, n_bins);
            }
            memcpy(prev_mag, mag, sizeof(float) * n_bins);
            
            mfcc_compute(&bank, power, mfcc_buffer);

            int offset = 0;
            for (int i = 0; i < n_mfcc; i++) static_features[f * n_static_feats + offset + i] = mfcc_buffer[i];
            offset += n_mfcc;
            static_features[f * n_static_feats + offset++] = rms;
            static_features[f * n_static_feats + offset++] = zcr;
            static_features[f * n_static_feats + offset++] = crest;
            static_features[f * n_static_feats + offset++] = centroid;
            static_features[f * n_static_feats + offset++] = rolloff;
            static_features[f * n_static_feats + offset++] = bandwidth;
            static_features[f * n_static_feats + offset++] = flatness;
            static_features[f * n_static_feats + offset]   = flux;
        }

        compute_deltas(static_features, delta_mfccs, num_frames, n_mfcc);
        compute_deltas(delta_mfccs, delta2_mfccs, num_frames, n_mfcc);

        float *temp_values = (float *)malloc(num_frames * sizeof(float));
        
        FeatureStats stats_static_mfcc[n_mfcc];
        for (int i = 0; i < n_mfcc; i++) {
            for (size_t f = 0; f < num_frames; f++) temp_values[f] = static_features[f * n_static_feats + i];
            stats_static_mfcc[i] = compute_stats(temp_values, num_frames);
        }
        FeatureStats stats_delta_mfcc[n_mfcc];
        for (int i = 0; i < n_mfcc; i++) {
            for (size_t f = 0; f < num_frames; f++) temp_values[f] = delta_mfccs[f * n_mfcc + i];
            stats_delta_mfcc[i] = compute_stats(temp_values, num_frames);
        }
        FeatureStats stats_delta2_mfcc[n_mfcc];
        for (int i = 0; i < n_mfcc; i++) {
            for (size_t f = 0; f < num_frames; f++) temp_values[f] = delta2_mfccs[f * n_mfcc + i];
            stats_delta2_mfcc[i] = compute_stats(temp_values, num_frames);
        }
        FeatureStats stats_other[n_static_other];
        for (int i = 0; i < n_static_other; i++) {
            for (size_t f = 0; f < num_frames; f++) temp_values[f] = static_features[f * n_static_feats + n_mfcc + i];
            stats_other[i] = compute_stats(temp_values, num_frames);
        }

        double voicing_rate = 0.0;
        size_t n_voiced = 0;
        for(size_t f = 0; f < num_frames; f++) if(voiced_flags[f]) n_voiced++;
        voicing_rate = (num_frames > 0) ? (double)n_voiced / (double)num_frames : 0.0;

        FeatureStats stats_f0 = {0};
        if (n_voiced > 0) {
            float *voiced_f0s = (float *)malloc(n_voiced * sizeof(float));
            size_t current_voiced_idx = 0;
            for(size_t f = 0; f < num_frames; f++) if(voiced_flags[f]) voiced_f0s[current_voiced_idx++] = f0_values[f];
            stats_f0 = compute_stats(voiced_f0s, n_voiced);
            free(voiced_f0s);
        }
        
        FeatureStats stats_harmonicity = compute_stats(harmonicity_values, num_frames);

        fprintf(out, "%s,%s", id, class_name);
        for (int s = 0; s < 4; s++) {
            double value = 0.0;
            for (int i = 0; i < n_mfcc; i++) {
                if (s == 0) value = stats_static_mfcc[i].mean; else if (s == 1) value = stats_static_mfcc[i].std;
                else if (s == 2) value = stats_static_mfcc[i].skewness; else value = stats_static_mfcc[i].kurtosis;
                fprintf(out, ",%.6f", value);
            }
            for (int i = 0; i < n_mfcc; i++) {
                if (s == 0) value = stats_delta_mfcc[i].mean; else if (s == 1) value = stats_delta_mfcc[i].std;
                else if (s == 2) value = stats_delta_mfcc[i].skewness; else value = stats_delta_mfcc[i].kurtosis;
                fprintf(out, ",%.6f", value);
            }
            for (int i = 0; i < n_mfcc; i++) {
                if (s == 0) value = stats_delta2_mfcc[i].mean; else if (s == 1) value = stats_delta2_mfcc[i].std;
                else if (s == 2) value = stats_delta2_mfcc[i].skewness; else value = stats_delta2_mfcc[i].kurtosis;
                fprintf(out, ",%.6f", value);
            }
            for (int i = 0; i < n_static_other; i++) {
                if (s == 0) value = stats_other[i].mean; else if (s == 1) value = stats_other[i].std;
                else if (s == 2) value = stats_other[i].skewness; else value = stats_other[i].kurtosis;
                fprintf(out, ",%.6f", value);
            }
        }
        fprintf(out, ",%.6f,%.6f,%.6f,%.6f,%.6f", stats_f0.mean, stats_f0.std, voicing_rate, stats_harmonicity.mean, stats_harmonicity.std);
        fprintf(out, "\n");
        processed++;

        mfcc_free(&bank);
        free(frames);
        free(static_features);
        free(delta_mfccs);
        free(delta2_mfccs);
        free(f0_values);
        free(harmonicity_values);
        free(voiced_flags);
        free(window);
        free(fft_in);
        free(real);
        free(imag);
        free(mag);
        free(prev_mag);
        free(power);
        free(mfcc_buffer);
        free(temp_values);
        csv_free_row(&row);
    }

    fclose(meta);
    fclose(out);

    fprintf(stdout, "Summary: processed=%zu missing=%zu failed=%zu\n", processed, missing, failed);
    return 0;
}

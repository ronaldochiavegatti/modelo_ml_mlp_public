// Preprocess WAV files: normalize, frame, and optionally remove silence.
#include "frame_io.h"
#include "util.h"

#include <dirent.h>
#include <math.h>
#include <sndfile.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>

typedef struct {
    size_t files;
    size_t failures;
    size_t frames;
} ProcessStats;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --input data --output processed [--frame-ms 30] [--hop-ms 10] [--remove-silence] [--silence-threshold 0.1]\n",
            prog);
}

static int has_wav_extension(const char *name) {
    const char *dot = strrchr(name, '.');
    if (!dot) {
        return 0;
    }
    return (strcasecmp(dot, ".wav") == 0);
}

static int compare_floats(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float *resample(const float *input, size_t input_len, int original_sr, int target_sr, size_t *output_len) {
    double ratio = (double)target_sr / (double)original_sr;
    *output_len = (size_t)((double)input_len * ratio);
    float *output = (float *)malloc(sizeof(float) * *output_len);
    if (!output) {
        return NULL;
    }

    for (size_t i = 0; i < *output_len; i++) {
        double src_index = (double)i / ratio;
        size_t index1 = (size_t)src_index;
        double frac = src_index - index1;

        if (index1 + 1 < input_len) {
            output[i] = (float)((1.0 - frac) * input[index1] + frac * input[index1 + 1]);
        } else {
            output[i] = input[index1];
        }
    }
    return output;
}


static int compute_rms(const float *x, size_t len, float *out_rms) {
    double sum = 0.0;
    if (!x || len == 0) {
        return -1;
    }
    for (size_t i = 0; i < len; i++) {
        sum += x[i] * x[i];
    }
    *out_rms = (float)sqrt(sum / (double)len);
    return 0;
}

static int extract_class_id(const char *root, const char *path, char *class_buf, size_t class_len, char *id_buf, size_t id_len) {
    size_t root_len = strlen(root);
    const char *rel = path;
    const char *slash;
    const char *file;

    if (strncmp(path, root, root_len) == 0) {
        rel = path + root_len;
        if (*rel == '/') {
            rel++;
        }
    }

    slash = strchr(rel, '/');
    if (!slash) {
        return -1;
    }

    if ((size_t)(slash - rel) >= class_len) {
        return -1;
    }
    memcpy(class_buf, rel, (size_t)(slash - rel));
    class_buf[slash - rel] = '\0';

    file = strrchr(path, '/');
    if (!file) {
        file = path;
    } else {
        file++;
    }

    if (!has_wav_extension(file)) {
        return -1;
    }

    size_t base_len = strlen(file) - 4;
    if (base_len >= id_len) {
        return -1;
    }
    memcpy(id_buf, file, base_len);
    id_buf[base_len] = '\0';
    return 0;
}

static int process_file(const char *root, const char *path, const char *output_dir,
                        int frame_ms, int hop_ms, int remove_silence, float silence_thresh,
                        size_t *frames_kept) {
    SF_INFO info;
    SNDFILE *snd = NULL;
    float *buffer = NULL;
    float *mono = NULL;
    long frames_read;
    float max_abs = 0.0f;
    size_t total_samples;
    int frame_len;
    int hop_len;
    size_t num_frames;
    float rms_thresh = 0.0f;
    size_t kept = 0;
    char class_name[256];
    char id[256];
    char out_dir[512];
    char out_path[1024];
    FILE *out = NULL;
    FrameHeader header;

    if (frames_kept) {
        *frames_kept = 0;
    }

    memset(&info, 0, sizeof(info));
    snd = sf_open(path, SFM_READ, &info);
    if (!snd) {
        fprintf(stderr, "Failed to open %s\n", path);
        return -1;
    }

    total_samples = (size_t)info.frames;
    buffer = (float *)malloc(sizeof(float) * total_samples * (size_t)info.channels);
    mono = (float *)malloc(sizeof(float) * total_samples);
    if (!buffer || !mono) {
        sf_close(snd);
        free(buffer);
        free(mono);
        return -1;
    }

    // Load audio and mix down to mono.
    frames_read = sf_readf_float(snd, buffer, info.frames);
    sf_close(snd);
    if (frames_read <= 0) {
        free(buffer);
        free(mono);
        return -1;
    }

    // Average channels to mono.
    for (size_t i = 0; i < (size_t)frames_read; i++) {
        double sum = 0.0;
        for (int ch = 0; ch < info.channels; ch++) {
            sum += buffer[i * (size_t)info.channels + ch];
        }
        mono[i] = (float)(sum / (double)info.channels);
    }

    // Resample to 16 kHz if necessary.
    if (info.samplerate != 16000) {
        size_t resampled_len = 0;
        float *resampled_mono = resample(mono, (size_t)frames_read, info.samplerate, 16000, &resampled_len);
        if (!resampled_mono) {
            free(buffer);
            free(mono);
            return -1;
        }
        free(mono);
        mono = resampled_mono;
        info.samplerate = 16000;
        info.frames = (sf_count_t)resampled_len;
        frames_read = (long)resampled_len;
        total_samples = resampled_len;
    }

    // Peak-normalize to [-1, 1] to reduce volume variability.
    for (size_t i = 0; i < (size_t)frames_read; i++) {
        float abs_val = fabsf(mono[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    if (max_abs > 0.0f) {
        for (size_t i = 0; i < (size_t)frames_read; i++) {
            mono[i] /= max_abs;
        }
    }

    // Convert frame/hop lengths from ms to samples.
    frame_len = (int)((info.samplerate * frame_ms) / 1000);
    hop_len = (int)((info.samplerate * hop_ms) / 1000);
    if (frame_len <= 0 || hop_len <= 0 || (size_t)frames_read < (size_t)frame_len) {
        free(buffer);
        free(mono);
        return -1;
    }

    num_frames = 1 + ((size_t)frames_read - (size_t)frame_len) / (size_t)hop_len;

    // If silence removal is enabled, compute RMS percentile to set relative threshold.
    if (remove_silence) {
        if (num_frames > 0) {
            float *all_rms = (float *)malloc(sizeof(float) * num_frames);
            if (!all_rms) {
                free(buffer);
                free(mono);
                return -1;
            }
            for (size_t i = 0; i < num_frames; i++) {
                size_t start = i * (size_t)hop_len;
                compute_rms(&mono[start], (size_t)frame_len, &all_rms[i]);
            }

            // Use 90th percentile of RMS as the reference for the threshold.
            qsort(all_rms, num_frames, sizeof(float), compare_floats);
            float percentile_rms = all_rms[(size_t)(0.9f * (float)num_frames)];
            rms_thresh = percentile_rms * silence_thresh;
            free(all_rms);
        }
    }

    if (extract_class_id(root, path, class_name, sizeof(class_name), id, sizeof(id)) != 0) {
        free(buffer);
        free(mono);
        return -1;
    }

    int out_dir_len = snprintf(out_dir, sizeof(out_dir), "%s/%s", output_dir, class_name);
    if (out_dir_len < 0 || (size_t)out_dir_len >= sizeof(out_dir)) {
        free(buffer);
        free(mono);
        return -1;
    }
    if (mkdir_p(out_dir) != 0) {
        free(buffer);
        free(mono);
        return -1;
    }

    int out_path_len = snprintf(out_path, sizeof(out_path), "%s/%s.frames", out_dir, id);
    if (out_path_len < 0 || (size_t)out_path_len >= sizeof(out_path)) {
        free(buffer);
        free(mono);
        return -1;
    }
    out = fopen(out_path, "wb");
    if (!out) {
        free(buffer);
        free(mono);
        return -1;
    }

    frames_fill_header(&header, (uint32_t)info.samplerate, (uint32_t)frame_len, (uint32_t)hop_len, 0);
    if (frames_write_header(out, &header) != 0) {
        fclose(out);
        free(buffer);
        free(mono);
        return -1;
    }

    // Write only frames above the silence threshold.
    for (size_t i = 0; i < num_frames; i++) {
        size_t start = i * (size_t)hop_len;
        float rms = 0.0f;
        if (remove_silence) {
            compute_rms(&mono[start], (size_t)frame_len, &rms);
            if (rms < rms_thresh) {
                continue;
            }
        }
        if (frames_write_frame(out, &mono[start], (size_t)frame_len) != 0) {
            fclose(out);
            free(buffer);
            free(mono);
            return -1;
        }
        kept++;
    }

    header.num_frames = (uint32_t)kept;
    if (frames_update_header(out, &header) != 0) {
        fclose(out);
        free(buffer);
        free(mono);
        return -1;
    }

    fclose(out);
    free(buffer);
    free(mono);

    if (frames_kept) {
        *frames_kept = kept;
    }
    fprintf(stdout, "Processed %s -> %s (%zu frames)\n", path, out_path, kept);
    return 0;
}

static int process_dir(const char *root, const char *dir_path, const char *output_dir,
                       int frame_ms, int hop_ms, int remove_silence, float silence_thresh,
                       ProcessStats *stats) {
    DIR *dir = opendir(dir_path);
    struct dirent *entry;
    if (!dir) {
        return -1;
    }

    while ((entry = readdir(dir)) != NULL) {
        char path[1024];
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
        struct stat st;
        if (stat(path, &st) != 0) {
            continue;
        }
        if (S_ISDIR(st.st_mode)) {
            process_dir(root, path, output_dir, frame_ms, hop_ms, remove_silence, silence_thresh, stats);
        } else if (S_ISREG(st.st_mode)) {
            if (has_wav_extension(entry->d_name)) {
                size_t kept = 0;
                if (process_file(root, path, output_dir, frame_ms, hop_ms, remove_silence, silence_thresh,
                                 &kept) != 0) {
                    if (stats) {
                        stats->failures++;
                    }
                } else if (stats) {
                    stats->files++;
                    stats->frames += kept;
                }
            }
        }
    }

    closedir(dir);
    return 0;
}

int main(int argc, char **argv) {
    const char *input_dir = NULL;
    const char *output_dir = NULL;
    int frame_ms = 30;
    int hop_ms = 10;
    int remove_silence = 0;
    float silence_thresh = 0.1f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_dir = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (strcmp(argv[i], "--frame-ms") == 0 && i + 1 < argc) {
            frame_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hop-ms") == 0 && i + 1 < argc) {
            hop_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--remove-silence") == 0) {
            remove_silence = 1;
        } else if (strcmp(argv[i], "--silence-threshold") == 0 && i + 1 < argc) {
            silence_thresh = (float)atof(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!input_dir || !output_dir) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stdout,
            "Preprocess: input=%s output=%s frame_ms=%d hop_ms=%d remove_silence=%d silence_threshold=%.3f\n",
            input_dir, output_dir, frame_ms, hop_ms, remove_silence, silence_thresh);

    ProcessStats stats = {0};
    if (process_dir(input_dir, input_dir, output_dir, frame_ms, hop_ms, remove_silence, silence_thresh, &stats) != 0) {
        fprintf(stderr, "Failed to process input directory\n");
        return 1;
    }

    fprintf(stdout, "Summary: files=%zu failures=%zu frames=%zu\n", stats.files, stats.failures, stats.frames);

    return 0;
}

// Split samples into train/test with speaker separation and z-score normalization.
#include "csv.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

typedef struct {
    char *id;
    char *class_name;
    char *speaker_id;
    float *features;
} Sample;

typedef struct {
    char *id;
    char *speaker_id;
} MetaEntry;

typedef struct {
    char **labels;
    size_t count;
    size_t cap;
} ClassList;

typedef struct {
    char *speaker_id;
    int *class_counts;
    int is_train;
} SpeakerInfo;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --features features.csv --metadata metadata.csv --train train.data --test test.data --classes classes.txt [--train-ratio 0.7] [--seed 42] [--scaler scaler.csv]\n",
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

static char *strdup_safe(const char *s) {
    if (!s) {
        return NULL;
    }
    size_t len = strlen(s);
    char *out = (char *)malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, s, len + 1);
    return out;
}

static int speaker_index(SpeakerInfo *speakers, size_t count, const char *speaker_id) {
    for (size_t i = 0; i < count; i++) {
        if (strcmp(speakers[i].speaker_id, speaker_id) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int speaker_add(SpeakerInfo **speakers, size_t *count, size_t *cap, size_t class_count, const char *speaker_id) {
    if (*count == *cap) {
        size_t new_cap = (*cap == 0) ? 64 : (*cap * 2);
        SpeakerInfo *tmp = (SpeakerInfo *)realloc(*speakers, new_cap * sizeof(SpeakerInfo));
        if (!tmp) {
            return -1;
        }
        *speakers = tmp;
        *cap = new_cap;
    }
    SpeakerInfo *sp = &(*speakers)[*count];
    sp->speaker_id = strdup_safe(speaker_id);
    sp->class_counts = (int *)calloc(class_count, sizeof(int));
    sp->is_train = 0;
    (*count)++;
    return (int)(*count - 1);
}

static const char *class_list_add(ClassList *list, const char *label) {
    for (size_t i = 0; i < list->count; i++) {
        if (strcmp(list->labels[i], label) == 0) {
            return list->labels[i];
        }
    }
    if (list->count == list->cap) {
        size_t new_cap = (list->cap == 0) ? 8 : list->cap * 2;
        char **tmp = (char **)realloc(list->labels, new_cap * sizeof(char *));
        if (!tmp) {
            return NULL;
        }
        list->labels = tmp;
        list->cap = new_cap;
    }
    list->labels[list->count] = strdup_safe(label);
    return list->labels[list->count++];
}

static int class_index(const ClassList *list, const char *label) {
    for (size_t i = 0; i < list->count; i++) {
        if (strcmp(list->labels[i], label) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static char *speaker_from_id(const char *id) {
    const char *sep = strchr(id, '_');
    size_t len = sep ? (size_t)(sep - id) : strlen(id);
    char *out = (char *)malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, id, len);
    out[len] = '\0';
    return out;
}

static const char *metadata_speaker(const MetaEntry *meta, size_t count, const char *id) {
    for (size_t i = 0; i < count; i++) {
        if (strcmp(meta[i].id, id) == 0) {
            return meta[i].speaker_id;
        }
    }
    return NULL;
}

static void shuffle_indices(size_t *arr, size_t n, unsigned int seed) {
    if (n < 2) {
        return;
    }
    srand(seed);
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rand() % (int)(i + 1));
        size_t tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

static int write_fann_data(const char *path, Sample *samples, size_t *indices, size_t count,
                           size_t feat_dim, const ClassList *classes, const float *mean, const float *std) {
    FILE *out = fopen(path, "w");
    if (!out) {
        perror("fopen fann");
        return -1;
    }

    // FANN training format: header then input/output pairs.
    fprintf(out, "%zu %zu %zu\n", count, feat_dim, classes->count);
    for (size_t i = 0; i < count; i++) {
        Sample *s = &samples[indices[i]];
        // Apply z-score normalization computed on the training set.
        for (size_t f = 0; f < feat_dim; f++) {
            float val = s->features[f];
            float norm = (std[f] > 0.0f) ? (val - mean[f]) / std[f] : 0.0f;
            fprintf(out, "%s%.6f", (f == 0) ? "" : " ", norm);
        }
        fprintf(out, "\n");

        int cls = class_index(classes, s->class_name);
        for (size_t c = 0; c < classes->count; c++) {
            float out_val = (c == (size_t)cls) ? 1.0f : 0.0f;
            fprintf(out, "%s%.1f", (c == 0) ? "" : " ", out_val);
        }
        fprintf(out, "\n");
    }

    fclose(out);
    return 0;
}

int main(int argc, char **argv) {
    const char *features_path = NULL;
    const char *metadata_path = NULL;
    const char *train_path = NULL;
    const char *test_path = NULL;
    const char *classes_path = NULL;
    const char *scaler_path = NULL;
    double train_ratio = 0.7;
    unsigned int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--features") == 0 && i + 1 < argc) {
            features_path = argv[++i];
        } else if (strcmp(argv[i], "--metadata") == 0 && i + 1 < argc) {
            metadata_path = argv[++i];
        } else if (strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            train_path = argv[++i];
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            test_path = argv[++i];
        } else if (strcmp(argv[i], "--classes") == 0 && i + 1 < argc) {
            classes_path = argv[++i];
        } else if (strcmp(argv[i], "--train-ratio") == 0 && i + 1 < argc) {
            train_ratio = atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--scaler") == 0 && i + 1 < argc) {
            scaler_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!features_path || !metadata_path || !train_path || !test_path || !classes_path) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stdout,
            "Split/normalize: features=%s metadata=%s train=%s test=%s classes=%s train_ratio=%.2f seed=%u scaler=%s\n",
            features_path, metadata_path, train_path, test_path, classes_path, train_ratio, seed,
            scaler_path ? scaler_path : "(none)");

    FILE *meta = fopen(metadata_path, "r");
    if (!meta) {
        perror("fopen metadata");
        return 1;
    }

    char line[8192];
    if (!fgets(line, sizeof(line), meta)) {
        fprintf(stderr, "Empty metadata file\n");
        fclose(meta);
        return 1;
    }

    CsvRow mheader = {0};
    csv_split_line(line, &mheader);
    int id_idx = header_index(&mheader, "id");
    int speaker_idx = header_index(&mheader, "speaker_id");
    csv_free_row(&mheader);

    if (id_idx < 0) {
        fprintf(stderr, "metadata.csv missing id column\n");
        fclose(meta);
        return 1;
    }

    // Load speaker mapping to avoid speaker leakage (when available).
    MetaEntry *meta_entries = NULL;
    size_t meta_count = 0;
    size_t meta_cap = 0;

    while (fgets(line, sizeof(line), meta)) {
        CsvRow row = {0};
        if (csv_split_line(line, &row) != 0 || row.count == 0) {
            csv_free_row(&row);
            continue;
        }
        if ((size_t)id_idx >= row.count) {
            csv_free_row(&row);
            continue;
        }
        const char *id = row.fields[id_idx];
        const char *speaker = NULL;
        if (speaker_idx >= 0 && (size_t)speaker_idx < row.count) {
            speaker = row.fields[speaker_idx];
        }

        if (meta_count == meta_cap) {
            size_t new_cap = (meta_cap == 0) ? 64 : meta_cap * 2;
            MetaEntry *tmp = (MetaEntry *)realloc(meta_entries, new_cap * sizeof(MetaEntry));
            if (!tmp) {
                csv_free_row(&row);
                break;
            }
            meta_entries = tmp;
            meta_cap = new_cap;
        }

        meta_entries[meta_count].id = strdup_safe(id);
        meta_entries[meta_count].speaker_id = speaker && *speaker ? strdup_safe(speaker) : NULL;
        meta_count++;
        csv_free_row(&row);
    }
    fclose(meta);

    FILE *feat = fopen(features_path, "r");
    if (!feat) {
        perror("fopen features");
        return 1;
    }

    if (!fgets(line, sizeof(line), feat)) {
        fprintf(stderr, "Empty features file\n");
        fclose(feat);
        return 1;
    }

    CsvRow fheader = {0};
    csv_split_line(line, &fheader);
    if (fheader.count < 3) {
        fprintf(stderr, "Invalid features header\n");
        csv_free_row(&fheader);
        fclose(feat);
        return 1;
    }

    size_t feat_dim = fheader.count - 2;
    csv_free_row(&fheader);

    // Load feature rows (id, class, features...).
    Sample *samples = NULL;
    size_t sample_count = 0;
    size_t sample_cap = 0;
    ClassList classes = {0};

    while (fgets(line, sizeof(line), feat)) {
        CsvRow row = {0};
        if (csv_split_line(line, &row) != 0 || row.count == 0) {
            csv_free_row(&row);
            continue;
        }
        if (row.count < feat_dim + 2) {
            csv_free_row(&row);
            continue;
        }

        const char *id = row.fields[0];
        const char *class_name = row.fields[1];

        if (sample_count == sample_cap) {
            size_t new_cap = (sample_cap == 0) ? 128 : sample_cap * 2;
            Sample *tmp = (Sample *)realloc(samples, new_cap * sizeof(Sample));
            if (!tmp) {
                csv_free_row(&row);
                break;
            }
            samples = tmp;
            sample_cap = new_cap;
        }

        samples[sample_count].id = strdup_safe(id);
        samples[sample_count].class_name = strdup_safe(class_name);

        const char *speaker = metadata_speaker(meta_entries, meta_count, id);
        if (speaker && *speaker) {
            samples[sample_count].speaker_id = strdup_safe(speaker);
        } else {
            samples[sample_count].speaker_id = speaker_from_id(id);
        }

        samples[sample_count].features = (float *)malloc(sizeof(float) * feat_dim);
        for (size_t f = 0; f < feat_dim; f++) {
            samples[sample_count].features[f] = (float)atof(row.fields[f + 2]);
        }

        class_list_add(&classes, class_name);
        sample_count++;
        csv_free_row(&row);
    }
    fclose(feat);

    if (sample_count == 0) {
        fprintf(stderr, "No samples loaded\n");
        return 1;
    }

    // Aggregate per-speaker class counts to build a balanced split.
    SpeakerInfo *speakers = NULL;
    size_t speaker_count = 0;
    size_t speaker_cap = 0;
    int *total_class = (int *)calloc(classes.count, sizeof(int));
    if (!total_class) {
        fprintf(stderr, "Allocation failed for class counts\n");
        return 1;
    }

    for (size_t i = 0; i < sample_count; i++) {
        int cls = class_index(&classes, samples[i].class_name);
        if (cls < 0) {
            continue;
        }
        total_class[cls] += 1;

        int sp_idx = speaker_index(speakers, speaker_count, samples[i].speaker_id);
        if (sp_idx < 0) {
            sp_idx = speaker_add(&speakers, &speaker_count, &speaker_cap, classes.count, samples[i].speaker_id);
            if (sp_idx < 0) {
                fprintf(stderr, "Failed to add speaker\n");
                return 1;
            }
        }
        speakers[sp_idx].class_counts[cls] += 1;
    }

    fprintf(stdout, "Loaded samples: %zu classes: %zu speakers: %zu feat_dim: %zu\n",
            sample_count, classes.count, speaker_count, feat_dim);

    double *target_train = (double *)malloc(sizeof(double) * classes.count);
    double *train_counts = (double *)calloc(classes.count, sizeof(double));
    size_t *order = (size_t *)malloc(sizeof(size_t) * speaker_count);
    if (!target_train || !train_counts || !order) {
        fprintf(stderr, "Allocation failed for speaker split\n");
        return 1;
    }

    for (size_t c = 0; c < classes.count; c++) {
        target_train[c] = (double)total_class[c] * train_ratio;
    }

    for (size_t i = 0; i < speaker_count; i++) {
        order[i] = i;
    }
    shuffle_indices(order, speaker_count, seed);

    size_t train_speakers = 0;
    // Greedy assignment of speakers to the train set to match target ratios.
    for (size_t i = 0; i < speaker_count; i++) {
        SpeakerInfo *sp = &speakers[order[i]];
        double err_train = 0.0;
        double err_test = 0.0;
        for (size_t c = 0; c < classes.count; c++) {
            double diff_train = (train_counts[c] + (double)sp->class_counts[c]) - target_train[c];
            double diff_test = train_counts[c] - target_train[c];
            err_train += diff_train * diff_train;
            err_test += diff_test * diff_test;
        }
        if (err_train <= err_test) {
            sp->is_train = 1;
            train_speakers++;
            for (size_t c = 0; c < classes.count; c++) {
                train_counts[c] += sp->class_counts[c];
            }
        } else {
            sp->is_train = 0;
        }
    }

    if (speaker_count > 1 && (train_speakers == 0 || train_speakers == speaker_count)) {
        size_t idx = order[0];
        speakers[idx].is_train = (train_speakers == 0) ? 1 : 0;
    }

    size_t *train_idx = (size_t *)malloc(sizeof(size_t) * sample_count);
    size_t *test_idx = (size_t *)malloc(sizeof(size_t) * sample_count);
    size_t train_count = 0;
    size_t test_count = 0;

    for (size_t i = 0; i < sample_count; i++) {
        int sp_idx = speaker_index(speakers, speaker_count, samples[i].speaker_id);
        if (sp_idx >= 0 && speakers[sp_idx].is_train) {
            train_idx[train_count++] = i;
        } else {
            test_idx[test_count++] = i;
        }
    }

    // Compute normalization stats on the training split only.
    float *mean = (float *)calloc(feat_dim, sizeof(float));
    float *std = (float *)calloc(feat_dim, sizeof(float));
    if (!mean || !std) {
        fprintf(stderr, "Allocation failed for normalization\n");
        return 1;
    }

    for (size_t i = 0; i < train_count; i++) {
        Sample *s = &samples[train_idx[i]];
        for (size_t f = 0; f < feat_dim; f++) {
            mean[f] += s->features[f];
        }
    }
    for (size_t f = 0; f < feat_dim; f++) {
        mean[f] = mean[f] / (float)train_count;
    }
    for (size_t i = 0; i < train_count; i++) {
        Sample *s = &samples[train_idx[i]];
        for (size_t f = 0; f < feat_dim; f++) {
            float diff = s->features[f] - mean[f];
            std[f] += diff * diff;
        }
    }
    for (size_t f = 0; f < feat_dim; f++) {
        std[f] = sqrtf(std[f] / (float)train_count);
        if (std[f] == 0.0f) {
            std[f] = 1.0f;
        }
    }

    if (write_fann_data(train_path, samples, train_idx, train_count, feat_dim, &classes, mean, std) != 0) {
        fprintf(stderr, "Failed to write train.data\n");
        return 1;
    }
    if (write_fann_data(test_path, samples, test_idx, test_count, feat_dim, &classes, mean, std) != 0) {
        fprintf(stderr, "Failed to write test.data\n");
        return 1;
    }

    FILE *class_out = fopen(classes_path, "w");
    if (class_out) {
        for (size_t i = 0; i < classes.count; i++) {
            fprintf(class_out, "%s\n", classes.labels[i]);
        }
        fclose(class_out);
    }

    if (scaler_path) {
        FILE *scaler = fopen(scaler_path, "w");
        if (scaler) {
            fprintf(scaler, "feature,mean,std\n");
            for (size_t f = 0; f < feat_dim; f++) {
                fprintf(scaler, "%zu,%.6f,%.6f\n", f, mean[f], std[f]);
            }
            fclose(scaler);
        }
    }

    fprintf(stdout, "Train samples: %zu, Test samples: %zu\n", train_count, test_count);
    fprintf(stdout, "Wrote train=%s test=%s classes=%s\n", train_path, test_path, classes_path);
    if (scaler_path) {
        fprintf(stdout, "Wrote scaler=%s\n", scaler_path);
    }

    return 0;
}

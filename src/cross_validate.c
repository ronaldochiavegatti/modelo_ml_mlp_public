// K-fold cross-validation with speaker-aware splits and hyperparameter grid.
#include "csv.h"

#include <fann.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

typedef struct {
    char *id;
    char *class_name;
    char *speaker_id;
    int class_index;
    int fold;
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
    int fold;
} SpeakerInfo;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --features features.csv --metadata metadata.csv --k 5 --hidden 32,64 --hidden2 0,32 --learning-rate 0.01,0.001 --max-epochs 300 --seed 42 --output cv_report.csv\n",
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
    sp->fold = -1;
    (*count)++;
    return (int)(*count - 1);
}

static int class_list_add(ClassList *list, const char *label) {
    for (size_t i = 0; i < list->count; i++) {
        if (strcmp(list->labels[i], label) == 0) {
            return (int)i;
        }
    }
    if (list->count == list->cap) {
        size_t new_cap = (list->cap == 0) ? 8 : list->cap * 2;
        char **tmp = (char **)realloc(list->labels, new_cap * sizeof(char *));
        if (!tmp) {
            return -1;
        }
        list->labels = tmp;
        list->cap = new_cap;
    }
    list->labels[list->count] = strdup_safe(label);
    list->count++;
    return (int)(list->count - 1);
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

static int parse_int_list(const char *str, int **out, size_t *count) {
    char *tmp = strdup_safe(str);
    char *token = strtok(tmp, ",");
    size_t cap = 8;
    size_t n = 0;
    int *vals = (int *)malloc(sizeof(int) * cap);
    if (!vals) {
        free(tmp);
        return -1;
    }
    while (token) {
        if (n == cap) {
            cap *= 2;
            int *nv = (int *)realloc(vals, sizeof(int) * cap);
            if (!nv) {
                free(vals);
                free(tmp);
                return -1;
            }
            vals = nv;
        }
        vals[n++] = atoi(token);
        token = strtok(NULL, ",");
    }
    free(tmp);
    *out = vals;
    *count = n;
    return 0;
}

static int parse_float_list(const char *str, float **out, size_t *count) {
    char *tmp = strdup_safe(str);
    char *token = strtok(tmp, ",");
    size_t cap = 8;
    size_t n = 0;
    float *vals = (float *)malloc(sizeof(float) * cap);
    if (!vals) {
        free(tmp);
        return -1;
    }
    while (token) {
        if (n == cap) {
            cap *= 2;
            float *nv = (float *)realloc(vals, sizeof(float) * cap);
            if (!nv) {
                free(vals);
                free(tmp);
                return -1;
            }
            vals = nv;
        }
        vals[n++] = (float)atof(token);
        token = strtok(NULL, ",");
    }
    free(tmp);
    *out = vals;
    *count = n;
    return 0;
}

static void compute_mean_std(Sample *samples, size_t *indices, size_t count, size_t feat_dim, float *mean, float *std) {
    for (size_t f = 0; f < feat_dim; f++) {
        mean[f] = 0.0f;
        std[f] = 0.0f;
    }

    for (size_t i = 0; i < count; i++) {
        Sample *s = &samples[indices[i]];
        for (size_t f = 0; f < feat_dim; f++) {
            mean[f] += s->features[f];
        }
    }
    for (size_t f = 0; f < feat_dim; f++) {
        mean[f] /= (float)count;
    }

    for (size_t i = 0; i < count; i++) {
        Sample *s = &samples[indices[i]];
        for (size_t f = 0; f < feat_dim; f++) {
            float diff = s->features[f] - mean[f];
            std[f] += diff * diff;
        }
    }
    for (size_t f = 0; f < feat_dim; f++) {
        std[f] = sqrtf(std[f] / (float)count);
        if (std[f] == 0.0f) {
            std[f] = 1.0f;
        }
    }
}

static struct fann_train_data *build_train_data(Sample *samples, size_t *indices, size_t count, size_t feat_dim,
                                                size_t num_classes, const float *mean, const float *std) {
    struct fann_train_data *data = fann_create_train((unsigned int)count, (unsigned int)feat_dim, (unsigned int)num_classes);
    for (size_t i = 0; i < count; i++) {
        Sample *s = &samples[indices[i]];
        for (size_t f = 0; f < feat_dim; f++) {
            float norm = (s->features[f] - mean[f]) / std[f];
            data->input[i][f] = norm;
        }
        for (size_t c = 0; c < num_classes; c++) {
            data->output[i][c] = (c == (size_t)s->class_index) ? 1.0f : 0.0f;
        }
    }
    return data;
}

static double eval_accuracy(struct fann *ann, Sample *samples, size_t *indices, size_t count, size_t feat_dim,
                            size_t num_classes, const float *mean, const float *std) {
    size_t correct = 0;
    float *input = (float *)malloc(sizeof(float) * feat_dim);
    for (size_t i = 0; i < count; i++) {
        Sample *s = &samples[indices[i]];
        for (size_t f = 0; f < feat_dim; f++) {
            input[f] = (s->features[f] - mean[f]) / std[f];
        }
        fann_type *out = fann_run(ann, input);
        int pred = 0;
        fann_type best = out[0];
        for (size_t c = 1; c < num_classes; c++) {
            if (out[c] > best) {
                best = out[c];
                pred = (int)c;
            }
        }
        if (pred == s->class_index) {
            correct++;
        }
    }
    free(input);
    return (count > 0) ? (double)correct / (double)count : 0.0;
}

int main(int argc, char **argv) {
    const char *features_path = NULL;
    const char *metadata_path = NULL;
    const char *hidden_list = NULL;
    const char *hidden2_list = NULL;
    const char *lr_list = NULL;
    const char *output_path = NULL;
    unsigned int max_epochs = 300;
    unsigned int k = 5;
    unsigned int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--features") == 0 && i + 1 < argc) {
            features_path = argv[++i];
        } else if (strcmp(argv[i], "--metadata") == 0 && i + 1 < argc) {
            metadata_path = argv[++i];
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            hidden_list = argv[++i];
        } else if (strcmp(argv[i], "--hidden2") == 0 && i + 1 < argc) {
            hidden2_list = argv[++i];
        } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
            lr_list = argv[++i];
        } else if (strcmp(argv[i], "--max-epochs") == 0 && i + 1 < argc) {
            max_epochs = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!features_path || !metadata_path || !hidden_list || !hidden2_list || !lr_list || !output_path) {
        print_usage(argv[0]);
        return 1;
    }

    int *hidden_vals = NULL;
    int *hidden2_vals = NULL;
    float *lr_vals = NULL;
    size_t hidden_count = 0;
    size_t hidden2_count = 0;
    size_t lr_count = 0;

    if (parse_int_list(hidden_list, &hidden_vals, &hidden_count) != 0 ||
        parse_int_list(hidden2_list, &hidden2_vals, &hidden2_count) != 0 ||
        parse_float_list(lr_list, &lr_vals, &lr_count) != 0) {
        fprintf(stderr, "Failed to parse hyperparameters\n");
        return 1;
    }

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

    // Load metadata to map id -> speaker_id when available.
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

    // Load features into memory and assign class indices.
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
        samples[sample_count].class_index = class_list_add(&classes, class_name);
        if (samples[sample_count].class_index < 0) {
            csv_free_row(&row);
            break;
        }

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
        sample_count++;
        csv_free_row(&row);
    }
    fclose(feat);

    if (sample_count == 0 || classes.count == 0) {
        fprintf(stderr, "No samples loaded\n");
        return 1;
    }

    // Build speaker pool so each speaker is assigned to one fold.
    SpeakerInfo *speakers = NULL;
    size_t speaker_count = 0;
    size_t speaker_cap = 0;
    int *total_class = (int *)calloc(classes.count, sizeof(int));
    if (!total_class) {
        fprintf(stderr, "Allocation failed for class counts\n");
        return 1;
    }

    for (size_t i = 0; i < sample_count; i++) {
        int cls = samples[i].class_index;
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

    double *target_per_fold = (double *)malloc(sizeof(double) * classes.count);
    double *fold_counts = (double *)calloc(k * classes.count, sizeof(double));
    size_t *order = (size_t *)malloc(sizeof(size_t) * speaker_count);
    if (!target_per_fold || !fold_counts || !order) {
        fprintf(stderr, "Allocation failed for folds\n");
        return 1;
    }

    for (size_t c = 0; c < classes.count; c++) {
        target_per_fold[c] = (double)total_class[c] / (double)k;
    }
    for (size_t i = 0; i < speaker_count; i++) {
        order[i] = i;
    }
    shuffle_indices(order, speaker_count, seed);

    // Greedy fold assignment to balance class counts across folds.
    for (size_t i = 0; i < speaker_count; i++) {
        SpeakerInfo *sp = &speakers[order[i]];
        int best_fold = 0;
        double best_error = -1.0;
        for (unsigned int fold = 0; fold < k; fold++) {
            double error = 0.0;
            for (size_t c = 0; c < classes.count; c++) {
                double current = fold_counts[fold * classes.count + c];
                double diff = (current + (double)sp->class_counts[c]) - target_per_fold[c];
                error += diff * diff;
            }
            if (best_error < 0.0 || error < best_error) {
                best_error = error;
                best_fold = (int)fold;
            }
        }
        sp->fold = best_fold;
        for (size_t c = 0; c < classes.count; c++) {
            fold_counts[best_fold * classes.count + c] += sp->class_counts[c];
        }
    }

    // Propagate fold assignment to each sample.
    for (size_t i = 0; i < sample_count; i++) {
        int sp_idx = speaker_index(speakers, speaker_count, samples[i].speaker_id);
        samples[i].fold = (sp_idx >= 0) ? speakers[sp_idx].fold : 0;
    }

    FILE *report = fopen(output_path, "w");
    if (!report) {
        perror("fopen report");
        return 1;
    }
    fprintf(report, "hidden1,hidden2,learning_rate,mean_accuracy,std_accuracy\n");

    float *mean = (float *)malloc(sizeof(float) * feat_dim);
    float *std = (float *)malloc(sizeof(float) * feat_dim);

    size_t *train_idx = (size_t *)malloc(sizeof(size_t) * sample_count);
    size_t *test_idx = (size_t *)malloc(sizeof(size_t) * sample_count);

    // Grid search across hidden sizes and learning rates.
    for (size_t h = 0; h < hidden_count; h++) {
        for (size_t h2 = 0; h2 < hidden2_count; h2++) {
            for (size_t lr = 0; lr < lr_count; lr++) {
                double sum_acc = 0.0;
                double sumsq_acc = 0.0;

                for (unsigned int fold = 0; fold < k; fold++) {
                    size_t train_count = 0;
                    size_t test_count = 0;

                    for (size_t i = 0; i < sample_count; i++) {
                        if (samples[i].fold == (int)fold) {
                            test_idx[test_count++] = i;
                        } else {
                            train_idx[train_count++] = i;
                        }
                    }

                    compute_mean_std(samples, train_idx, train_count, feat_dim, mean, std);

                    struct fann_train_data *train_data = build_train_data(samples, train_idx, train_count, feat_dim,
                                                                         classes.count, mean, std);

                    unsigned int layers = (hidden2_vals[h2] > 0) ? 4 : 3;
                    struct fann *ann;
                    if (layers == 4) {
                        ann = fann_create_standard(layers, train_data->num_input, hidden_vals[h], (unsigned int)hidden2_vals[h2], train_data->num_output);
                    } else {
                        ann = fann_create_standard(layers, train_data->num_input, hidden_vals[h], train_data->num_output);
                    }
                    fann_set_learning_rate(ann, lr_vals[lr]);
                    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
                    fann_set_activation_function_hidden(ann, FANN_SIGMOID);
                    fann_set_activation_function_output(ann, FANN_SIGMOID);

                    for (unsigned int epoch = 0; epoch < max_epochs; epoch++) {
                        fann_train_epoch(ann, train_data);
                    }

                    double acc = eval_accuracy(ann, samples, test_idx, test_count, feat_dim, classes.count, mean, std);
                    sum_acc += acc;
                    sumsq_acc += acc * acc;

                    fann_destroy_train(train_data);
                    fann_destroy(ann);
                }

                double mean_acc = sum_acc / (double)k;
                double var = (sumsq_acc / (double)k) - mean_acc * mean_acc;
                if (var < 0.0) {
                    var = 0.0;
                }
                double std_acc = sqrt(var);

                fprintf(report, "%d,%d,%.6f,%.6f,%.6f\n", hidden_vals[h], hidden2_vals[h2], lr_vals[lr], mean_acc, std_acc);
            }
        }
    }

    fclose(report);
    return 0;
}

// Evaluate a trained FANN model and emit metrics + confusion matrix.
#include <fann.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char **labels;
    size_t count;
    size_t cap;
} ClassList;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --model model.net --test test.data --classes classes.txt --output results.csv\n",
            prog);
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

static int class_list_add(ClassList *list, const char *label) {
    if (list->count == list->cap) {
        size_t new_cap = (list->cap == 0) ? 8 : list->cap * 2;
        char **tmp = (char **)realloc(list->labels, new_cap * sizeof(char *));
        if (!tmp) {
            return -1;
        }
        list->labels = tmp;
        list->cap = new_cap;
    }
    list->labels[list->count++] = strdup_safe(label);
    return 0;
}

static int argmax(const fann_type *arr, unsigned int len) {
    unsigned int best = 0;
    fann_type best_val = arr[0];
    for (unsigned int i = 1; i < len; i++) {
        if (arr[i] > best_val) {
            best_val = arr[i];
            best = i;
        }
    }
    return (int)best;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *test_path = NULL;
    const char *classes_path = NULL;
    const char *output_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            test_path = argv[++i];
        } else if (strcmp(argv[i], "--classes") == 0 && i + 1 < argc) {
            classes_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !test_path || !classes_path || !output_path) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stdout, "Evaluate: model=%s test=%s classes=%s output=%s\n",
            model_path, test_path, classes_path, output_path);

    FILE *cls = fopen(classes_path, "r");
    if (!cls) {
        perror("fopen classes");
        return 1;
    }

    // Load class labels to map output indices -> names.
    ClassList classes = {0};
    char line[512];
    while (fgets(line, sizeof(line), cls)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) {
            continue;
        }
        class_list_add(&classes, line);
    }
    fclose(cls);
    fprintf(stdout, "Classes loaded: %zu\n", classes.count);

    struct fann *ann = fann_create_from_file(model_path);
    if (!ann) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    struct fann_train_data *test_data = fann_read_train_from_file(test_path);
    if (!test_data) {
        fprintf(stderr, "Failed to read test data\n");
        fann_destroy(ann);
        return 1;
    }

    if (classes.count != test_data->num_output) {
        fprintf(stderr, "Class count mismatch: classes=%zu outputs=%u\n", classes.count, test_data->num_output);
        fann_destroy_train(test_data);
        fann_destroy(ann);
        return 1;
    }

    size_t num_classes = classes.count;
    size_t total = test_data->num_data;
    size_t correct = 0;

    unsigned int **conf = (unsigned int **)calloc(num_classes, sizeof(unsigned int *));
    for (size_t i = 0; i < num_classes; i++) {
        conf[i] = (unsigned int *)calloc(num_classes, sizeof(unsigned int));
    }

    // Build confusion matrix from argmax predictions.
    for (unsigned int i = 0; i < test_data->num_data; i++) {
        fann_type *out = fann_run(ann, test_data->input[i]);
        int pred = argmax(out, test_data->num_output);
        int truth = argmax(test_data->output[i], test_data->num_output);
        if (pred == truth) {
            correct++;
        }
        conf[truth][pred]++;
    }

    double accuracy = (total > 0) ? (double)correct / (double)total : 0.0;

    double precision_macro = 0.0;
    double recall_macro = 0.0;

    double *precision = (double *)calloc(num_classes, sizeof(double));
    double *recall = (double *)calloc(num_classes, sizeof(double));

    for (size_t c = 0; c < num_classes; c++) {
        unsigned int tp = conf[c][c];
        unsigned int fp = 0;
        unsigned int fn = 0;
        for (size_t j = 0; j < num_classes; j++) {
            if (j != c) {
                fp += conf[j][c];
                fn += conf[c][j];
            }
        }
        precision[c] = (tp + fp) > 0 ? (double)tp / (double)(tp + fp) : 0.0;
        recall[c] = (tp + fn) > 0 ? (double)tp / (double)(tp + fn) : 0.0;
        precision_macro += precision[c];
        recall_macro += recall[c];
    }

    if (num_classes > 0) {
        precision_macro /= (double)num_classes;
        recall_macro /= (double)num_classes;
    }

    fprintf(stdout, "Accuracy=%.6f precision_macro=%.6f recall_macro=%.6f\n",
            accuracy, precision_macro, recall_macro);

    FILE *out = fopen(output_path, "w");
    if (!out) {
        perror("fopen output");
        fann_destroy_train(test_data);
        fann_destroy(ann);
        return 1;
    }

    fprintf(out, "metric,accuracy,%.6f\n", accuracy);
    fprintf(out, "metric,precision_macro,%.6f\n", precision_macro);
    fprintf(out, "metric,recall_macro,%.6f\n", recall_macro);
    for (size_t c = 0; c < num_classes; c++) {
        fprintf(out, "metric,precision_%s,%.6f\n", classes.labels[c], precision[c]);
        fprintf(out, "metric,recall_%s,%.6f\n", classes.labels[c], recall[c]);
    }

    fprintf(out, "confusion,actual/pred");
    for (size_t c = 0; c < num_classes; c++) {
        fprintf(out, ",%s", classes.labels[c]);
    }
    fprintf(out, "\n");

    for (size_t i = 0; i < num_classes; i++) {
        fprintf(out, "confusion,%s", classes.labels[i]);
        for (size_t j = 0; j < num_classes; j++) {
            fprintf(out, ",%u", conf[i][j]);
        }
        fprintf(out, "\n");
    }

    fclose(out);
    fprintf(stdout, "Saved results to %s\n", output_path);

    fann_destroy_train(test_data);
    fann_destroy(ann);

    for (size_t i = 0; i < num_classes; i++) {
        free(conf[i]);
    }
    free(conf);
    free(precision);
    free(recall);

    return 0;
}

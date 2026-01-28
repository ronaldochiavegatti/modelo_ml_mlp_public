// Train/evaluate an SVM baseline from FANN-formatted data.
#include <svm.h>

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
            "Usage: %s --train train.data --test test.data --classes classes.txt --output svm_results.csv [--c 1.0] [--gamma 0.0] [--model svm_model.svm]\n",
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

static int argmax(const float *arr, size_t len) {
    size_t best = 0;
    float best_val = arr[0];
    for (size_t i = 1; i < len; i++) {
        if (arr[i] > best_val) {
            best_val = arr[i];
            best = i;
        }
    }
    return (int)best;
}

static int read_fann_data(const char *path, float **inputs_out, int **labels_out,
                          size_t *num_samples, size_t *num_inputs, size_t *num_outputs) {
    FILE *in = fopen(path, "r");
    if (!in) {
        return -1;
    }

    unsigned int n_samples = 0;
    unsigned int n_inputs = 0;
    unsigned int n_outputs = 0;
    if (fscanf(in, "%u %u %u", &n_samples, &n_inputs, &n_outputs) != 3) {
        fclose(in);
        return -1;
    }

    float *inputs = (float *)malloc(sizeof(float) * (size_t)n_samples * n_inputs);
    int *labels = (int *)malloc(sizeof(int) * n_samples);
    float *outputs = (float *)malloc(sizeof(float) * n_outputs);
    if (!inputs || !labels || !outputs) {
        fclose(in);
        free(inputs);
        free(labels);
        free(outputs);
        return -1;
    }

    // Each sample has one input line and one one-hot output line.
    for (unsigned int i = 0; i < n_samples; i++) {
        for (unsigned int j = 0; j < n_inputs; j++) {
            if (fscanf(in, "%f", &inputs[i * n_inputs + j]) != 1) {
                fclose(in);
                free(inputs);
                free(labels);
                free(outputs);
                return -1;
            }
        }
        for (unsigned int j = 0; j < n_outputs; j++) {
            if (fscanf(in, "%f", &outputs[j]) != 1) {
                fclose(in);
                free(inputs);
                free(labels);
                free(outputs);
                return -1;
            }
        }
        labels[i] = argmax(outputs, n_outputs);
    }

    free(outputs);
    fclose(in);

    *inputs_out = inputs;
    *labels_out = labels;
    *num_samples = n_samples;
    *num_inputs = n_inputs;
    *num_outputs = n_outputs;
    return 0;
}

int main(int argc, char **argv) {
    const char *train_path = NULL;
    const char *test_path = NULL;
    const char *classes_path = NULL;
    const char *output_path = NULL;
    const char *model_path = "svm_model.svm";
    double c_param = 1.0;
    double gamma_param = 0.0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            train_path = argv[++i];
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            test_path = argv[++i];
        } else if (strcmp(argv[i], "--classes") == 0 && i + 1 < argc) {
            classes_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--c") == 0 && i + 1 < argc) {
            c_param = atof(argv[++i]);
        } else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) {
            gamma_param = atof(argv[++i]);
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!train_path || !test_path || !classes_path || !output_path) {
        print_usage(argv[0]);
        return 1;
    }

    ClassList classes = { NULL, 0, 0 };
    FILE *cls = fopen(classes_path, "r");
    if (!cls) {
        perror("fopen classes");
        return 1;
    }
    char line[256];
    while (fgets(line, sizeof(line), cls)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len > 0) {
            class_list_add(&classes, line);
        }
    }
    fclose(cls);

    float *train_inputs = NULL;
    int *train_labels = NULL;
    size_t train_samples = 0;
    size_t num_inputs = 0;
    size_t num_outputs = 0;

    if (read_fann_data(train_path, &train_inputs, &train_labels, &train_samples, &num_inputs, &num_outputs) != 0) {
        fprintf(stderr, "Failed to read train.data\n");
        return 1;
    }

    float *test_inputs = NULL;
    int *test_labels = NULL;
    size_t test_samples = 0;
    size_t test_inputs_dim = 0;
    size_t test_outputs = 0;

    if (read_fann_data(test_path, &test_inputs, &test_labels, &test_samples, &test_inputs_dim, &test_outputs) != 0) {
        fprintf(stderr, "Failed to read test.data\n");
        return 1;
    }

    if (num_inputs != test_inputs_dim || num_outputs != test_outputs) {
        fprintf(stderr, "Train/test dimension mismatch\n");
        return 1;
    }
    if (classes.count != num_outputs) {
        fprintf(stderr, "Class count mismatch: classes=%zu outputs=%zu\n", classes.count, num_outputs);
        return 1;
    }

    // Build libsvm problem in sparse format.
    struct svm_problem prob;
    memset(&prob, 0, sizeof(prob));
    prob.l = (int)train_samples;
    prob.y = (double *)malloc(sizeof(double) * train_samples);
    prob.x = (struct svm_node **)malloc(sizeof(struct svm_node *) * train_samples);

    for (size_t i = 0; i < train_samples; i++) {
        prob.y[i] = (double)train_labels[i];
        prob.x[i] = (struct svm_node *)malloc(sizeof(struct svm_node) * (num_inputs + 1));
        for (size_t j = 0; j < num_inputs; j++) {
            prob.x[i][j].index = (int)j + 1;
            prob.x[i][j].value = train_inputs[i * num_inputs + j];
        }
        prob.x[i][num_inputs].index = -1;
        prob.x[i][num_inputs].value = 0.0;
    }

    // Default RBF settings with optional C/gamma override.
    struct svm_parameter param;
    memset(&param, 0, sizeof(param));
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = (gamma_param > 0.0) ? gamma_param : (1.0 / (double)num_inputs);
    param.coef0 = 0;
    param.cache_size = 100;
    param.eps = 1e-3;
    param.C = c_param;
    param.nr_weight = 0;
    param.nu = 0.5;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;

    const char *err = svm_check_parameter(&prob, &param);
    if (err) {
        fprintf(stderr, "SVM parameter error: %s\n", err);
        return 1;
    }

    struct svm_model *model = svm_train(&prob, &param);
    if (!model) {
        fprintf(stderr, "Failed to train SVM\n");
        return 1;
    }

    svm_save_model(model_path, model);

    size_t num_classes = classes.count;
    unsigned int **conf = (unsigned int **)calloc(num_classes, sizeof(unsigned int *));
    for (size_t i = 0; i < num_classes; i++) {
        conf[i] = (unsigned int *)calloc(num_classes, sizeof(unsigned int));
    }

    size_t correct = 0;
    // Evaluate on test set to build confusion matrix.
    for (size_t i = 0; i < test_samples; i++) {
        struct svm_node *x = (struct svm_node *)malloc(sizeof(struct svm_node) * (num_inputs + 1));
        for (size_t j = 0; j < num_inputs; j++) {
            x[j].index = (int)j + 1;
            x[j].value = test_inputs[i * num_inputs + j];
        }
        x[num_inputs].index = -1;
        x[num_inputs].value = 0.0;

        int pred = (int)svm_predict(model, x);
        int truth = test_labels[i];
        if (pred == truth) {
            correct++;
        }
        if (truth >= 0 && (size_t)truth < num_classes && pred >= 0 && (size_t)pred < num_classes) {
            conf[truth][pred]++;
        }
        free(x);
    }

    double accuracy = (test_samples > 0) ? (double)correct / (double)test_samples : 0.0;

    double *precision = (double *)calloc(num_classes, sizeof(double));
    double *recall = (double *)calloc(num_classes, sizeof(double));
    double precision_macro = 0.0;
    double recall_macro = 0.0;

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

    FILE *out = fopen(output_path, "w");
    if (!out) {
        perror("fopen output");
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
    svm_free_and_destroy_model(&model);
    return 0;
}

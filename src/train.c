// Train a simple MLP with FANN and save the model.
#include <fann.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --train train.data --model model.net [--hidden 32] [--hidden2 0] [--learning-rate 0.01] [--max-epochs 500] [--desired-error 0.001] [--log train.log] [--seed 42]\n",
            prog);
}

static int file_exists(const char *path) {
    FILE *fh = fopen(path, "r");
    if (fh) {
        fclose(fh);
        return 1;
    }
    return 0;
}

static char *make_svg_path(const char *model_path) {
    if (!model_path) {
        return NULL;
    }
    const char *slash = strrchr(model_path, '/');
    const char *base = slash ? slash + 1 : model_path;
    const char *dot = strrchr(base, '.');
    size_t len = strlen(model_path);
    size_t base_offset = (size_t)(base - model_path);
    size_t new_len = dot ? (base_offset + (size_t)(dot - base) + 4) : (len + 4);
    char *out = (char *)malloc(new_len + 1);
    if (!out) {
        return NULL;
    }
    if (dot) {
        size_t prefix = base_offset + (size_t)(dot - base);
        memcpy(out, model_path, prefix);
        memcpy(out + prefix, ".svg", 4);
        out[prefix + 4] = '\0';
    } else {
        memcpy(out, model_path, len);
        memcpy(out + len, ".svg", 4);
        out[len + 4] = '\0';
    }
    return out;
}

static int find_script_path(const char *argv0, char *out, size_t out_len) {
    const char *default_path = "src/plot_network_svg.py";
    if (file_exists(default_path)) {
        if (strlen(default_path) + 1 > out_len) {
            return -1;
        }
        memcpy(out, default_path, strlen(default_path) + 1);
        return 0;
    }
    if (argv0 && strchr(argv0, '/')) {
        char buf[1024];
        strncpy(buf, argv0, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        char *slash = strrchr(buf, '/');
        if (slash) {
            *slash = '\0';
            const char *suffix = "/../src/plot_network_svg.py";
            size_t base_len = strlen(buf);
            size_t suffix_len = strlen(suffix);
            if (base_len + suffix_len + 1 <= out_len) {
                memcpy(out, buf, base_len);
                memcpy(out + base_len, suffix, suffix_len + 1);
                if (file_exists(out)) {
                    return 0;
                }
            }
        }
    }
    return -1;
}

static void maybe_generate_svg(const char *model_path, const char *argv0) {
    char script_path[1024];
    if (find_script_path(argv0, script_path, sizeof(script_path)) != 0) {
        fprintf(stdout, "SVG: plot_network_svg.py not found; skipping diagram generation\n");
        return;
    }

    char *svg_path = make_svg_path(model_path);
    if (!svg_path) {
        fprintf(stderr, "SVG: failed to allocate output path\n");
        return;
    }

    const char *title = "MLP Architecture";
    char *args[] = {
        "python3",
        script_path,
        "--input",
        (char *)model_path,
        "--output",
        svg_path,
        "--title",
        (char *)title,
        NULL
    };

    pid_t pid = fork();
    if (pid == 0) {
        execvp(args[0], args);
        _exit(127);
    }
    if (pid < 0) {
        fprintf(stderr, "SVG: failed to fork process\n");
        free(svg_path);
        return;
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        fprintf(stderr, "SVG: failed to wait for generator\n");
        free(svg_path);
        return;
    }

    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        fprintf(stdout, "Saved network SVG to %s\n", svg_path);
    } else if (WIFEXITED(status)) {
        fprintf(stderr, "SVG: generator failed (exit=%d)\n", WEXITSTATUS(status));
    } else {
        fprintf(stderr, "SVG: generator failed\n");
    }

    free(svg_path);
}

int main(int argc, char **argv) {
    const char *train_path = NULL;
    const char *model_path = NULL;
    const char *log_path = NULL;
    unsigned int hidden = 32;
    unsigned int hidden2 = 0;
    float learning_rate = 0.01f;
    unsigned int max_epochs = 500;
    float desired_error = 0.001f;
    unsigned int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            train_path = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            hidden = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden2") == 0 && i + 1 < argc) {
            hidden2 = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
            learning_rate = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-epochs") == 0 && i + 1 < argc) {
            max_epochs = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--desired-error") == 0 && i + 1 < argc) {
            desired_error = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
            log_path = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!train_path || !model_path) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stdout,
            "Train: train=%s model=%s hidden=%u hidden2=%u learning_rate=%.6f max_epochs=%u desired_error=%.6f seed=%u\n",
            train_path, model_path, hidden, hidden2, learning_rate, max_epochs, desired_error, seed);

    // Load normalized training data produced by split_normalize.
    struct fann_train_data *train_data = fann_read_train_from_file(train_path);
    if (!train_data) {
        fprintf(stderr, "Failed to read training data\n");
        return 1;
    }

    fprintf(stdout, "Training data: samples=%u inputs=%u outputs=%u\n",
            train_data->num_data, train_data->num_input, train_data->num_output);

    unsigned int layers = (hidden2 > 0) ? 4 : 3;
    struct fann *ann;

    srand(seed);

    if (layers == 4) {
        ann = fann_create_standard(layers, train_data->num_input, hidden, hidden2, train_data->num_output);
    } else {
        ann = fann_create_standard(layers, train_data->num_input, hidden, train_data->num_output);
    }

    // Configure learning settings (RPROP + sigmoid activations).
    fann_set_learning_rate(ann, learning_rate);
    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    fann_set_activation_function_hidden(ann, FANN_SIGMOID);
    fann_set_activation_function_output(ann, FANN_SIGMOID);

    FILE *log = NULL;
    if (log_path) {
        log = fopen(log_path, "w");
    }

    // Train until desired error or max epochs; optionally log MSE.
    unsigned int report_every = 50;
    for (unsigned int epoch = 1; epoch <= max_epochs; epoch++) {
        float mse = fann_train_epoch(ann, train_data);
        if (log) {
            fprintf(log, "%u,%.6f\n", epoch, mse);
        }
        if (epoch == 1 || epoch % report_every == 0 || epoch == max_epochs || mse <= desired_error) {
            fprintf(stdout, "Epoch %u/%u mse=%.6f\n", epoch, max_epochs, mse);
        }
        if (mse <= desired_error) {
            fprintf(stdout, "Early stop at epoch %u (mse=%.6f)\n", epoch, mse);
            break;
        }
    }

    if (log) {
        fclose(log);
    }

    fann_save(ann, model_path);
    fprintf(stdout, "Saved model to %s\n", model_path);
    maybe_generate_svg(model_path, argv[0]);

    fann_destroy(ann);
    fann_destroy_train(train_data);

    return 0;
}

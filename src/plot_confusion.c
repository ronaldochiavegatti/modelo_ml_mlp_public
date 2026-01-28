// Plot a confusion matrix from results.csv into an SVG file.
#include "csv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char **items;
    size_t count;
} LabelList;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --input results.csv [--output confusion.svg] [--title \"Confusion Matrix\"]\n",
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

static void free_labels(LabelList *labels) {
    if (!labels || !labels->items) {
        return;
    }
    for (size_t i = 0; i < labels->count; i++) {
        free(labels->items[i]);
    }
    free(labels->items);
    labels->items = NULL;
    labels->count = 0;
}

static int read_confusion(const char *path, LabelList *labels, unsigned int **matrix_out) {
    FILE *in = fopen(path, "r");
    if (!in) {
        perror("fopen input");
        return -1;
    }

    char line[8192];
    unsigned int *matrix = NULL;
    size_t row_idx = 0;
    int header_seen = 0;

    while (fgets(line, sizeof(line), in)) {
        CsvRow row = {0};
        if (csv_split_line(line, &row) != 0) {
            csv_free_row(&row);
            continue;
        }
        if (row.count < 2) {
            csv_free_row(&row);
            continue;
        }
        if (strcmp(row.fields[0], "confusion") != 0) {
            csv_free_row(&row);
            continue;
        }

        if (strcmp(row.fields[1], "actual/pred") == 0) {
            if (row.count < 3) {
                csv_free_row(&row);
                continue;
            }

            free_labels(labels);
            free(matrix);
            matrix = NULL;

            labels->count = row.count - 2;
            labels->items = (char **)calloc(labels->count, sizeof(char *));
            if (!labels->items) {
                csv_free_row(&row);
                fclose(in);
                return -1;
            }
            for (size_t i = 0; i < labels->count; i++) {
                labels->items[i] = strdup_safe(row.fields[i + 2]);
                if (!labels->items[i]) {
                    csv_free_row(&row);
                    fclose(in);
                    free_labels(labels);
                    return -1;
                }
            }

            matrix = (unsigned int *)calloc(labels->count * labels->count, sizeof(unsigned int));
            if (!matrix) {
                csv_free_row(&row);
                fclose(in);
                free_labels(labels);
                return -1;
            }
            header_seen = 1;
            row_idx = 0;
            csv_free_row(&row);
            continue;
        }

        if (!header_seen || !matrix || labels->count == 0) {
            csv_free_row(&row);
            continue;
        }
        if (row.count < labels->count + 2) {
            csv_free_row(&row);
            continue;
        }
        if (row_idx >= labels->count) {
            csv_free_row(&row);
            continue;
        }

        for (size_t i = 0; i < labels->count; i++) {
            char *end = NULL;
            unsigned long val = strtoul(row.fields[i + 2], &end, 10);
            (void)end;
            matrix[row_idx * labels->count + i] = (unsigned int)val;
        }
        row_idx++;
        csv_free_row(&row);
    }

    fclose(in);

    if (!header_seen || !matrix || row_idx == 0) {
        fprintf(stderr, "No confusion matrix found in %s\n", path);
        free(matrix);
        free_labels(labels);
        return -1;
    }
    if (row_idx != labels->count) {
        fprintf(stderr, "Incomplete confusion matrix: expected %zu rows, got %zu\n", labels->count, row_idx);
        free(matrix);
        free_labels(labels);
        return -1;
    }

    *matrix_out = matrix;
    return 0;
}

static void svg_escape(FILE *out, const char *text) {
    if (!text) {
        return;
    }
    for (const char *p = text; *p; p++) {
        switch (*p) {
            case '&':
                fputs("&amp;", out);
                break;
            case '<':
                fputs("&lt;", out);
                break;
            case '>':
                fputs("&gt;", out);
                break;
            case '"':
                fputs("&quot;", out);
                break;
            case '\'':
                fputs("&apos;", out);
                break;
            default:
                fputc(*p, out);
        }
    }
}

static void color_from_value(double t, int *r, int *g, int *b) {
    int r0 = 247;
    int g0 = 251;
    int b0 = 255;
    int r1 = 8;
    int g1 = 48;
    int b1 = 107;

    if (t < 0.0) {
        t = 0.0;
    } else if (t > 1.0) {
        t = 1.0;
    }

    *r = (int)(r0 + t * (r1 - r0));
    *g = (int)(g0 + t * (g1 - g0));
    *b = (int)(b0 + t * (b1 - b0));
}

static int write_svg(const LabelList *labels, const unsigned int *matrix,
                     const char *output_path, const char *title) {
    FILE *out = fopen(output_path, "w");
    if (!out) {
        perror("fopen output");
        return -1;
    }

    size_t n = labels->count;
    unsigned int max_val = 0;
    size_t max_label = 0;
    for (size_t i = 0; i < n * n; i++) {
        if (matrix[i] > max_val) {
            max_val = matrix[i];
        }
    }
    for (size_t i = 0; i < n; i++) {
        size_t len = strlen(labels->items[i]);
        if (len > max_label) {
            max_label = len;
        }
    }

    int cell = 40;
    if (n > 12) {
        cell = 32;
    }
    if (n > 20) {
        cell = 26;
    }
    if (n > 28) {
        cell = 22;
    }

    int left_margin = 20 + (int)(max_label * 7);
    if (left_margin < 120) {
        left_margin = 120;
    }
    int bottom_margin = 60 + (int)(max_label * 6);
    if (bottom_margin < 100) {
        bottom_margin = 100;
    }

    int top_margin = 70;
    int right_margin = 20;
    int grid = cell * (int)n;
    int width = left_margin + grid + right_margin;
    int height = top_margin + grid + bottom_margin;

    int font_label = (cell >= 32) ? 12 : 10;
    int font_value = (cell >= 32) ? 12 : 9;
    int title_font = 18;

    fprintf(out, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(out,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n",
            width, height, width, height);
    fprintf(out, "<rect width=\"100%%\" height=\"100%%\" fill=\"white\" />\n");
    fprintf(out,
            "<text x=\"%d\" y=\"%d\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"%d\" text-anchor=\"middle\">",
            width / 2, 30, title_font);
    svg_escape(out, title);
    fprintf(out, "</text>\n");

    int grid_x = left_margin;
    int grid_y = top_margin;

    fprintf(out,
            "<text x=\"%d\" y=\"%d\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"%d\" text-anchor=\"middle\">Predicted</text>\n",
            grid_x + grid / 2, grid_y + grid + bottom_margin - 30, font_label);
    fprintf(out,
            "<text x=\"%d\" y=\"%d\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"%d\" text-anchor=\"middle\" "
            "transform=\"rotate(-90 %d %d)\">Actual</text>\n",
            20, grid_y + grid / 2, font_label, 20, grid_y + grid / 2);

    fprintf(out, "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"none\" stroke=\"#999999\" />\n",
            grid_x, grid_y, grid, grid);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            unsigned int val = matrix[i * n + j];
            double t = (max_val > 0) ? (double)val / (double)max_val : 0.0;
            int r, g, b;
            color_from_value(t, &r, &g, &b);
            int x = grid_x + (int)j * cell;
            int y = grid_y + (int)i * cell;

            fprintf(out,
                    "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"#%02x%02x%02x\" stroke=\"#ffffff\" />\n",
                    x, y, cell, cell, r, g, b);

            const char *text_color = (t > 0.55) ? "#ffffff" : "#000000";
            fprintf(out,
                    "<text x=\"%d\" y=\"%d\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"%d\" "
                    "text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"%s\">%u</text>\n",
                    x + cell / 2, y + cell / 2, font_value, text_color, val);
        }
    }

    for (size_t i = 0; i < n; i++) {
        int x = grid_x - 10;
        int y = grid_y + (int)i * cell + cell / 2;
        fprintf(out,
                "<text x=\"%d\" y=\"%d\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"%d\" "
                "text-anchor=\"end\" dominant-baseline=\"central\">",
                x, y, font_label);
        svg_escape(out, labels->items[i]);
        fprintf(out, "</text>\n");
    }

    for (size_t i = 0; i < n; i++) {
        int x = grid_x + (int)i * cell + cell / 2;
        int y = grid_y + grid + 10;
        fprintf(out,
                "<text x=\"%d\" y=\"%d\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"%d\" "
                "text-anchor=\"start\" transform=\"rotate(-45 %d %d)\">",
                x, y, font_label, x, y);
        svg_escape(out, labels->items[i]);
        fprintf(out, "</text>\n");
    }

    fprintf(out, "</svg>\n");
    fclose(out);
    return 0;
}

int main(int argc, char **argv) {
    const char *input_path = NULL;
    const char *output_path = "confusion.svg";
    const char *title = "Confusion Matrix";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--title") == 0 && i + 1 < argc) {
            title = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!input_path) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stdout, "Plot confusion: input=%s output=%s title=%s\n", input_path, output_path, title);

    LabelList labels = {0};
    unsigned int *matrix = NULL;
    if (read_confusion(input_path, &labels, &matrix) != 0) {
        return 1;
    }

    fprintf(stdout, "Classes: %zu\n", labels.count);

    if (write_svg(&labels, matrix, output_path, title) != 0) {
        free(matrix);
        free_labels(&labels);
        return 1;
    }

    fprintf(stdout, "Saved confusion plot to %s\n", output_path);

    free(matrix);
    free_labels(&labels);
    return 0;
}

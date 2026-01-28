// Organize dataset by class: copy/link audio files into data/<classe>/.
#define _POSIX_C_SOURCE 200809L

#include "csv.h"
#include "util.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    int id_idx;
    int class_idx;
    int path_idx;
} HeaderIndex;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --metadata metadata.csv --output data [--log logs/verify.log] [--mode copy|link] [--overwrite] [--dry-run]\n",
            prog);
}

static int header_index(const CsvRow *row, const char *name) {
    size_t i;
    for (i = 0; i < row->count; i++) {
        if (strcasecmp(row->fields[i], name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static void log_line(FILE *log, const char *level, const char *msg) {
    if (log) {
        fprintf(log, "[%s] %s\n", level, msg);
    }
}

int main(int argc, char **argv) {
    const char *metadata_path = NULL;
    const char *output_dir = "data";
    const char *log_path = NULL;
    const char *mode = "copy";
    int overwrite = 0;
    int dry_run = 0;
    FILE *meta = NULL;
    FILE *log = NULL;
    char line[8192];
    HeaderIndex idx = {-1, -1, -1};
    size_t total = 0;
    size_t missing = 0;
    size_t copied = 0;
    size_t skipped = 0;
    size_t linked = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--metadata") == 0 && i + 1 < argc) {
            metadata_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
            log_path = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--overwrite") == 0) {
            overwrite = 1;
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            dry_run = 1;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!metadata_path) {
        print_usage(argv[0]);
        return 1;
    }

    meta = fopen(metadata_path, "r");
    if (!meta) {
        perror("fopen metadata");
        return 1;
    }

    // Create log directory if needed.
    if (log_path) {
        char log_dir[1024];
        strncpy(log_dir, log_path, sizeof(log_dir) - 1);
        log_dir[sizeof(log_dir) - 1] = '\0';
        char *slash = strrchr(log_dir, '/');
        if (slash) {
            *slash = '\0';
            if (strlen(log_dir) > 0 && mkdir_p(log_dir) != 0) {
                fprintf(stderr, "Failed to create log directory\n");
            }
        }
        log = fopen(log_path, "w");
        if (!log) {
            perror("fopen log");
            fclose(meta);
            return 1;
        }
    } else {
        log = stdout;
    }

    fprintf(log, "Organizing dataset: metadata=%s output=%s mode=%s overwrite=%d dry_run=%d\n",
            metadata_path, output_dir, mode, overwrite, dry_run);

    if (!fgets(line, sizeof(line), meta)) {
        fprintf(stderr, "Empty metadata file\n");
        fclose(meta);
        if (log_path) fclose(log);
        return 1;
    }

    CsvRow header = {0};
    if (csv_split_line(line, &header) != 0) {
        fprintf(stderr, "Failed to parse metadata header\n");
        fclose(meta);
        if (log_path) fclose(log);
        return 1;
    }

    idx.id_idx = header_index(&header, "id");
    idx.class_idx = header_index(&header, "classe");
    idx.path_idx = header_index(&header, "filepath");
    csv_free_row(&header);

    if (idx.id_idx < 0 || idx.class_idx < 0 || idx.path_idx < 0) {
        fprintf(stderr, "Missing required columns: id, classe, filepath\n");
        fclose(meta);
        if (log_path) fclose(log);
        return 1;
    }

    if (mkdir_p(output_dir) != 0) {
        fprintf(stderr, "Failed to create output directory: %s\n", output_dir);
        fclose(meta);
        if (log_path) fclose(log);
        return 1;
    }

    // Iterate each metadata row and materialize it on disk.
    while (fgets(line, sizeof(line), meta)) {
        CsvRow row = {0};
        char dest_dir[1024];
        char dest_path[1200];
        const char *id;
        const char *class_name;
        const char *src_path;
        char msg[1500];

        if (csv_split_line(line, &row) != 0 || row.count == 0) {
            csv_free_row(&row);
            continue;
        }

        if ((size_t)idx.id_idx >= row.count || (size_t)idx.class_idx >= row.count || (size_t)idx.path_idx >= row.count) {
            csv_free_row(&row);
            continue;
        }

        id = row.fields[idx.id_idx];
        class_name = row.fields[idx.class_idx];
        src_path = row.fields[idx.path_idx];

        if (!id || !*id || !class_name || !*class_name || !src_path || !*src_path) {
            csv_free_row(&row);
            continue;
        }

        total++;

        if (!file_exists(src_path)) {
            missing++;
            snprintf(msg, sizeof(msg), "Missing source: %s", src_path);
            log_line(log, "WARN", msg);
            csv_free_row(&row);
            continue;
        }

        snprintf(dest_dir, sizeof(dest_dir), "%s/%s", output_dir, class_name);
        if (mkdir_p(dest_dir) != 0) {
            snprintf(msg, sizeof(msg), "Failed to create dir: %s", dest_dir);
            log_line(log, "ERROR", msg);
            csv_free_row(&row);
            continue;
        }

        snprintf(dest_path, sizeof(dest_path), "%s/%s.wav", dest_dir, id);

        if (file_exists(dest_path) && !overwrite) {
            skipped++;
            snprintf(msg, sizeof(msg), "Skip existing: %s", dest_path);
            log_line(log, "INFO", msg);
            csv_free_row(&row);
            continue;
        }

        if (dry_run) {
            snprintf(msg, sizeof(msg), "Dry-run: %s -> %s", src_path, dest_path);
            log_line(log, "INFO", msg);
            csv_free_row(&row);
            continue;
        }

        if (strcmp(mode, "link") == 0) {
            if (unlink(dest_path) != 0 && errno != ENOENT) {
                snprintf(msg, sizeof(msg), "Failed to remove existing: %s", dest_path);
                log_line(log, "ERROR", msg);
            }
            if (symlink(src_path, dest_path) != 0) {
                snprintf(msg, sizeof(msg), "Failed to link: %s", dest_path);
                log_line(log, "ERROR", msg);
                csv_free_row(&row);
                continue;
            }
            linked++;
        } else {
            if (copy_file(src_path, dest_path) != 0) {
                snprintf(msg, sizeof(msg), "Failed to copy: %s", dest_path);
                log_line(log, "ERROR", msg);
                csv_free_row(&row);
                continue;
            }
            copied++;
        }

        snprintf(msg, sizeof(msg), "OK: %s", dest_path);
        log_line(log, "INFO", msg);
        csv_free_row(&row);
    }

    fprintf(log, "Summary: total=%zu copied=%zu linked=%zu missing=%zu skipped=%zu\n",
            total, copied, linked, missing, skipped);

    fclose(meta);
    if (log_path) {
        fclose(log);
    }

    return 0;
}

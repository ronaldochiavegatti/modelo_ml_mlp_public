// Minimal CSV parser for metadata/features with quoted-field support.
#include "csv.h"

#include <stdlib.h>
#include <string.h>

static char *csv_strdup(const char *s) {
    size_t len;
    char *out;
    if (!s) {
        return NULL;
    }
    len = strlen(s);
    out = (char *)malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, s, len + 1);
    return out;
}

static void csv_append_char(char **buf, size_t *len, size_t *cap, char c) {
    if (*len + 1 >= *cap) {
        size_t new_cap = (*cap == 0) ? 64 : (*cap * 2);
        char *tmp = (char *)realloc(*buf, new_cap);
        if (!tmp) {
            return;
        }
        *buf = tmp;
        *cap = new_cap;
    }
    (*buf)[(*len)++] = c;
}

int csv_split_line(const char *line, CsvRow *row) {
    size_t cap = 8;
    size_t count = 0;
    char **fields = (char **)malloc(cap * sizeof(char *));
    const char *p = line;

    if (!fields) {
        return -1;
    }

    if (!line || !*line) {
        row->fields = fields;
        row->count = 0;
        return 0;
    }

    // Parse one field at a time, honoring quotes and escaped quotes ("").
    while (*p) {
        int in_quotes = 0;
        char *field = NULL;
        size_t len = 0;
        size_t fcap = 0;

        if (*p == '"') {
            in_quotes = 1;
            p++;
        }

        while (*p) {
            if (in_quotes) {
                if (*p == '"') {
                    if (*(p + 1) == '"') {
                        // Escaped quote within a quoted field.
                        csv_append_char(&field, &len, &fcap, '"');
                        p += 2;
                        continue;
                    }
                    in_quotes = 0;
                    p++;
                    continue;
                }
                csv_append_char(&field, &len, &fcap, *p);
                p++;
            } else {
                if (*p == ',') {
                    p++;
                    break;
                }
                if (*p == '\r' || *p == '\n') {
                    p++;
                    break;
                }
                csv_append_char(&field, &len, &fcap, *p);
                p++;
            }
        }

        if (field) {
            csv_append_char(&field, &len, &fcap, '\0');
        } else {
            // Empty field.
            field = (char *)malloc(1);
            if (field) {
                field[0] = '\0';
            }
        }

        if (count >= cap) {
            size_t new_cap = cap * 2;
            char **tmp = (char **)realloc(fields, new_cap * sizeof(char *));
            if (!tmp) {
                break;
            }
            fields = tmp;
            cap = new_cap;
        }
        fields[count++] = field ? field : csv_strdup("");

        if (*p == '\0') {
            break;
        }
    }

    row->fields = fields;
    row->count = count;
    return 0;
}

void csv_free_row(CsvRow *row) {
    size_t i;

    if (!row || !row->fields) {
        return;
    }

    for (i = 0; i < row->count; i++) {
        free(row->fields[i]);
    }
    free(row->fields);
    row->fields = NULL;
    row->count = 0;
}

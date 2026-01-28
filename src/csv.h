#ifndef CSV_H
#define CSV_H

#include <stddef.h>

typedef struct {
    char **fields;
    size_t count;
} CsvRow;

// Split a CSV line into fields (supports quoted fields and escaped quotes).
int csv_split_line(const char *line, CsvRow *row);
// Free memory allocated by csv_split_line.
void csv_free_row(CsvRow *row);

#endif

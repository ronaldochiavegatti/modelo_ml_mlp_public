#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>

// Ensure a directory path exists, creating intermediate directories as needed.
int mkdir_p(const char *path);
// Return 1 if the file exists, 0 otherwise.
int file_exists(const char *path);
// Copy a file from src_path to dst_path (overwrites destination).
int copy_file(const char *src_path, const char *dst_path);

#endif

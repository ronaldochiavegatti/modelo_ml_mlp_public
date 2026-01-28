// Small filesystem helpers shared across the pipeline tools.
#include "util.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int file_exists(const char *path) {
    struct stat st;
    // stat() returns 0 if the path exists and can be accessed.
    return (stat(path, &st) == 0);
}

int mkdir_p(const char *path) {
    char *tmp;
    size_t len;
    size_t i;

    if (!path) {
        return -1;
    }

    len = strlen(path);
    tmp = (char *)malloc(len + 1);
    if (!tmp) {
        return -1;
    }
    memcpy(tmp, path, len + 1);

    // Walk the path and create each directory segment in order.
    for (i = 1; i < len; i++) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            if (strlen(tmp) > 0) {
                if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                    free(tmp);
                    return -1;
                }
            }
            tmp[i] = '/';
        }
    }

    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        free(tmp);
        return -1;
    }

    free(tmp);
    return 0;
}

int copy_file(const char *src_path, const char *dst_path) {
    int src_fd;
    int dst_fd;
    char buf[8192];
    ssize_t nread;

    src_fd = open(src_path, O_RDONLY);
    if (src_fd < 0) {
        return -1;
    }

    dst_fd = open(dst_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (dst_fd < 0) {
        close(src_fd);
        return -1;
    }

    // Stream-copy to avoid loading the entire file in memory.
    while ((nread = read(src_fd, buf, sizeof(buf))) > 0) {
        char *out = buf;
        ssize_t remaining = nread;
        while (remaining > 0) {
            ssize_t nwritten = write(dst_fd, out, (size_t)remaining);
            if (nwritten < 0) {
                close(src_fd);
                close(dst_fd);
                return -1;
            }
            remaining -= nwritten;
            out += nwritten;
        }
    }

    close(src_fd);
    close(dst_fd);
    return (nread < 0) ? -1 : 0;
}

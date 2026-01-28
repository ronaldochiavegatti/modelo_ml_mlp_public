#ifndef FRAME_IO_H
#define FRAME_IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    char magic[8];
    uint32_t sample_rate;
    uint32_t frame_len;
    uint32_t hop_len;
    uint32_t num_frames;
} FrameHeader;

// Write a new header at the start of a .frames file.
int frames_write_header(FILE *out, const FrameHeader *header);
// Seek to start and overwrite the header (to update frame count).
int frames_update_header(FILE *out, const FrameHeader *header);
// Append a single frame (frame_len floats) to the file.
int frames_write_frame(FILE *out, const float *frame, size_t frame_len);
// Read an entire .frames file into memory.
int frames_read_all(const char *path, FrameHeader *header, float **frames_out);
// Fill header fields with a fixed magic and metadata.
void frames_fill_header(FrameHeader *header, uint32_t sample_rate, uint32_t frame_len, uint32_t hop_len, uint32_t num_frames);

#endif

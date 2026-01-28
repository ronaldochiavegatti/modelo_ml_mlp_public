// Binary .frames format: fixed header + contiguous float frames.
#include "frame_io.h"

#include <stdlib.h>
#include <string.h>

static const char kFrameMagic[8] = { 'F', 'R', 'A', 'M', 'E', 'S', '1', '\0' };

int frames_write_header(FILE *out, const FrameHeader *header) {
    if (!out || !header) {
        return -1;
    }
    if (fwrite(header, sizeof(FrameHeader), 1, out) != 1) {
        return -1;
    }
    return 0;
}

int frames_update_header(FILE *out, const FrameHeader *header) {
    if (!out || !header) {
        return -1;
    }
    if (fseek(out, 0, SEEK_SET) != 0) {
        return -1;
    }
    if (fwrite(header, sizeof(FrameHeader), 1, out) != 1) {
        return -1;
    }
    return 0;
}

int frames_write_frame(FILE *out, const float *frame, size_t frame_len) {
    if (!out || !frame || frame_len == 0) {
        return -1;
    }
    if (fwrite(frame, sizeof(float), frame_len, out) != frame_len) {
        return -1;
    }
    return 0;
}

int frames_read_all(const char *path, FrameHeader *header, float **frames_out) {
    FILE *in;
    FrameHeader local;
    size_t total;
    float *frames;

    if (!path || !header || !frames_out) {
        return -1;
    }

    in = fopen(path, "rb");
    if (!in) {
        return -1;
    }

    if (fread(&local, sizeof(FrameHeader), 1, in) != 1) {
        fclose(in);
        return -1;
    }

    // Validate magic to avoid reading wrong binary format.
    if (memcmp(local.magic, kFrameMagic, sizeof(kFrameMagic)) != 0) {
        fclose(in);
        return -1;
    }

    total = (size_t)local.frame_len * (size_t)local.num_frames;
    frames = (float *)malloc(sizeof(float) * total);
    if (!frames) {
        fclose(in);
        return -1;
    }

    if (fread(frames, sizeof(float), total, in) != total) {
        free(frames);
        fclose(in);
        return -1;
    }

    fclose(in);
    *header = local;
    *frames_out = frames;
    return 0;
}

void frames_fill_header(FrameHeader *header, uint32_t sample_rate, uint32_t frame_len, uint32_t hop_len, uint32_t num_frames) {
    if (!header) {
        return;
    }
    memset(header, 0, sizeof(*header));
    memcpy(header->magic, kFrameMagic, sizeof(kFrameMagic));
    header->sample_rate = sample_rate;
    header->frame_len = frame_len;
    header->hop_len = hop_len;
    header->num_frames = num_frames;
}

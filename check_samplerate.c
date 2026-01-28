#include <stdio.h>
#include <sndfile.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <file.wav>\n", argv[0]);
        return 1;
    }

    SF_INFO info;
    SNDFILE *snd = sf_open(argv[1], SFM_READ, &info);
    if (!snd) {
        fprintf(stderr, "Failed to open %s\n", argv[1]);
        return 1;
    }

    printf("%d\n", info.samplerate);
    sf_close(snd);
    return 0;
}

#include "common.h"

float get_pixel(Sample sample, int index, int i, int j) {

    assert(index < sample.length && i < sample.width && j < sample.height);
    return sample.data[index * sample.size + i * sample.width + j];
}

float get_usetime(struct timeval start, struct timeval end) {

    long timeuse = 0;
    timeuse =
        1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

    float time = 0;
    time = (float)timeuse / (float)1000000;

    return time;
}
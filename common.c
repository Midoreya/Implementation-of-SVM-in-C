#include "common.h"

float get_pixel(int i, int j, Sample data) {
    return data.data[data.size * i + j];
}

float get_usetime(struct timeval start, struct timeval end) {

    long timeuse = 0;
    timeuse =
        1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

    float time = 0;
    time = (float)timeuse / (float)1000000;

    return time;
}
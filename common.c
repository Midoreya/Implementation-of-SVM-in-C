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

int write_weight(Weight weight, int class, int size, char *file) {
    int i = 0, j = 0;
    FILE *fp;

    fp = fopen(file, "wb");

    for (i = 0; i < class * class; i++) {
        for (j = 0; j < size + 1; j++) {
            if (j == size)
                fprintf(fp, "%lf \n", weight.w[i][j] * pow(10, 3));
            else
                fprintf(fp, "%lf ", weight.w[i][j] * pow(10, 3));
        }
    }

    fclose(fp);
    return 0;
}
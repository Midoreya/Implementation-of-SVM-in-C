#include "common.h"

#include "val.h"

int hy_func(int in, Weight weight, Sample data) {
    int p = 0, n = 0;
    int d = 0;
    float *result;
    int *score;
    int index = 0;
    int r = 0, final = 0;

    result = (float *)calloc(data.classes * data.classes, sizeof(float));
    score = (int *)calloc(data.classes, sizeof(int));

    for (p = 0; p < data.classes; p++) {
        if (score[p] != 0) {
            printf("score != 0\n");
            exit(1);
        }
        for (n = 0; n < data.classes; n++) {
            if (result[p * data.classes + n] != 0) {
                printf("result != 0\n");
                exit(1);
            }
        }
    }

    for (p = 0; p < data.classes; p++) {
        for (n = 0; n < data.classes; n++) {
            if (p == n)
                continue;
            index = p * data.classes + n;
            for (d = 0; d < data.size; d++) {
                result[index] =
                    result[index] + weight.w[index][d] * get_pixel(in, d, data);
            }
            result[index] = result[index] + weight.w[index][data.size];
            if (result[index] > 0)
                score[p]++;
        }
    }

    for (p = 0; p < data.classes; p++) {
        if (score[p] > r) {
            r = score[p];
            final = p;
        }
    }

    return final;
}

Acc validation(Sample data, Weight weight) {
    int i = 0;
    Acc acc = {0};

    for (i = 0; i < data.length; i++) {
        if (hy_func(i, weight, data) == data.label[i])
            acc.true ++;
        else
            acc.false ++;
    }

    acc.acc = ((float)acc.true / (float)data.length);

    printf("Start validation...\n");
    return acc;
}
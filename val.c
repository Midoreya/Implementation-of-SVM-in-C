#include "common.h"

#include "inference.h"
#include "val.h"

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
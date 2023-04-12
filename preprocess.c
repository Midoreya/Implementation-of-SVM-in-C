#include <cblas.h>

#include "common.h"
#include "preprocess.h"

Parameter create_parameter(int max_epoch, float c, float ero) {

    Parameter parameter = {0};
    parameter.max_epoch = max_epoch;
    parameter.c = c;
    parameter.ero = ero;

    return parameter;
}

Sample prepare_sample(char *path, int size, int height, int width, int length,
                      int classes) {

    int i = 0, j = 0;

    FILE *fp;
    fp = fopen(path, "r");
    assert(fp != NULL);

    Sample sample = {0};
    sample.length = length;
    sample.size = size;
    sample.height = height;
    sample.width = width;
    sample.classes = classes;

    sample.data = (float *)calloc(length * size, sizeof(float));
    sample.label = (int *)calloc(length, sizeof(int));

    for (i = 0; i < length; i++) {
        for (j = 0; j < size + 1; j++) {
            if (j == 0) {
                fscanf(fp, "%d", &sample.label[i]);
                continue;
            } else
                fscanf(fp, "%f", &sample.data[i * size + j]);
        }
    }

    printf("-------------------------\n");
    printf("Create sample complete:\n");
    printf("length:  %d\nsize:    %d\nwidth:   %d\nheight:  %d\nclasses: %d\n",
           sample.length, sample.size, sample.width, sample.height,
           sample.classes);
    printf("-------------------------\n\n");

    return sample;
}

K_matrix compute_k_matrix(Sample sample) {
    
    K_matrix matrix = {0};
    clock_t start = 0, finish = 0;
    float use_time = 0;

    matrix.dim = sample.length;
    matrix.data = (float *)calloc(matrix.dim * matrix.dim, sizeof(float));

    start = clock();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix.dim, matrix.dim,
                sample.size, 1, sample.data, sample.size, sample.data,
                sample.size, 0, matrix.data, matrix.dim);

    finish = clock();

    use_time = (float)(finish - start) / 1000000;

    printf("----------------------------------\n");
    printf("Compute kernel matrix complete:\nUsing %.2fs\n", use_time);
    printf("----------------------------------\n\n");

    return matrix;
}
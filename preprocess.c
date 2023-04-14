#include <cblas.h>

#include "common.h"
#include "preprocess.h"

Parameter create_parameter(int max_epoch, int smo_epoch, float c, float ero) {

    Parameter parameter = {0};
    parameter.smo_epoch = smo_epoch;
    parameter.max_epoch = max_epoch;
    parameter.c = c;
    parameter.ero = ero;

    return parameter;
}

Sample create_sample(char *path, int size, int height, int width, int length,
                     int classes) {

    int i = 0, j = 0;

    float use_time = 0;
    struct timeval start, end;

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

    gettimeofday(&start, NULL);

    for (i = 0; i < length; i++) {
        for (j = 0; j < size + 1; j++) {
            if (j == 0) {
                fscanf(fp, "%d", &sample.label[i]);
                continue;
            } else
                fscanf(fp, "%f", &sample.data[i * size + j]);
        }
    }

    gettimeofday(&end, NULL);
    use_time = get_usetime(start, end);

    printf("-------------------------\n");
    printf("Create sample complete:\n");
    printf("Using:   %.2fs\n", use_time);
    printf("length:  %d\nsize:    %d\nwidth:   %d\nheight:  %d\nclasses: %d\n",
           sample.length, sample.size, sample.width, sample.height,
           sample.classes);
    printf("-------------------------\n\n");

    return sample;
}

Weight create_weight(int classes, int size) {

    Weight weight = {0};
    weight.w = (float **)calloc(classes * classes, sizeof(float *));

    int i = 0;
    for (i = 0; i < classes * classes; i++) {
        weight.w[i] = (float *)calloc(size + 1, sizeof(float));
    }

    return weight;
}

int copy_message(Sample *dest, Sample *src, int length) {

    assert(dest->size == 0);
    assert(dest->width == 0);
    assert(dest->height == 0);
    assert(dest->classes == 0);
    assert(dest->length == 0);

    dest->size = src->size;
    dest->width = src->width;
    dest->height = src->height;
    dest->classes = src->classes;

    dest->length = length;

    return 0;
}

Split_length create_length(int training, int val, int test) {

    Split_length length = {0};

    length.training_length = training;
    length.val_length = val;
    length.test_length = test;

    return length;
}

Dataset copy_data(Dataset data, Sample sample) {

    data.training.data = (float *)calloc(
        data.training.size * data.training.length, sizeof(float));
    data.val.data =
        (float *)calloc(data.val.size * data.val.length, sizeof(float));
    data.test.data =
        (float *)calloc(data.test.size * data.test.length, sizeof(float));

    int i = 0;
    for (i = 0; i < sample.length * sample.size; i++) {
        if (i < data.training.length * data.training.size)
            data.training.data[i] = sample.data[i];
        else if (i < data.training.length * data.training.size +
                         data.val.length * data.val.size)
            data.val.data[i - data.training.length * data.training.size] =
                sample.data[i];
        else
            data.test.data[i - data.training.length * data.training.size -
                           data.val.length * data.val.size] = sample.data[i];
    }

    return data;
}

Dataset copy_label(Dataset data, Sample sample) {

    data.training.label = (int *)calloc(data.training.length, sizeof(int));
    data.val.label = (int *)calloc(data.val.length, sizeof(int));
    data.test.label = (int *)calloc(data.test.length, sizeof(int));

    int i = 0;
    for (i = 0; i < sample.length; i++) {
        if (i < data.training.length)
            data.training.label[i] = sample.label[i];
        else if (i < data.training.length + data.val.length)
            data.val.label[i - data.training.length] = sample.label[i];
        else
            data.test.label[i - data.training.length - data.val.length] =
                sample.label[i];
    }

    return data;
}

Dataset dataloader(Sample sample, Split_length s_length) {

    Dataset data = {0};
    assert(s_length.training_length + s_length.val_length +
               s_length.test_length ==
           sample.length);

    float use_time = 0;
    struct timeval start, end;

    gettimeofday(&start, NULL);

    int ret = 0;
    ret = copy_message(&data.training, &sample, s_length.training_length);
    assert(ret == 0);

    ret = copy_message(&data.val, &sample, s_length.val_length);
    assert(ret == 0);

    ret = copy_message(&data.test, &sample, s_length.test_length);
    assert(ret == 0);

    data = copy_data(data, sample);
    data = copy_label(data, sample);

    gettimeofday(&end, NULL);
    use_time = get_usetime(start, end);

    printf("-------------------------\n");
    printf("Data loader complete:\n");
    printf("Using:           %.2fs\n", use_time);
    printf("training length: %d\n", data.training.length);
    printf("val length:      %d\n", data.val.length);
    printf("test length:     %d\n", data.test.length);

    printf("-------------------------\n\n");

    return data;
}

K_matrix compute_k_matrix(Sample sample) {

    K_matrix matrix = {0};

    float use_time = 0;
    struct timeval start, end;

    matrix.dim = sample.length;
    matrix.data = (float *)calloc(matrix.dim * matrix.dim, sizeof(float));

    gettimeofday(&start, NULL);

    printf("-------------------------\n");
    printf("Start compute k_matrix...\n");

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix.dim, matrix.dim,
                sample.size, 1, sample.data, sample.size, sample.data,
                sample.size, 0, matrix.data, matrix.dim);

    gettimeofday(&end, NULL);

    use_time = get_usetime(start, end);

    printf("Complete compute.\nUsing %.2fs.\n", use_time);
    printf("-------------------------\n\n");

    return matrix;
}
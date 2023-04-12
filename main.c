#include "common.h"
#include "preprocess.h"

Weight create_weight(int classes, int size) {

    Weight weight = {0};
    weight.w = (float **)calloc(classes * classes, sizeof(float *));

    int i = 0;
    for (i = 0; i < classes * classes; i++) {
        weight.w[i] = (float *)calloc(size + 1, sizeof(float));
    }

    return weight;
}

int print_starting(int class_p, int class_n, int count_0, int count_n,
                   int count_p) {
    int total = count_n + count_p + count_0;
    printf("---------------------------------------\n");
    printf("Start class[%d], class[%d] training.\n", class_p, class_n);
    printf("Positive sample count: %d\n", count_p);
    printf("Negative sample count: %d\n", count_n);
    printf("Other sample count:    %d\n", count_0);
    printf("Total sample count:    %d\n", total);

    return 0;
}

int print_complete(int class_p, int class_n, int count, float use_time) {

    printf("Complete class[%d], class[%d] training.\n", class_p, class_n);
    printf("Support vecter count = %d, use %.2fs.\n", count, use_time);
    printf("---------------------------------------\n\n");

    return 0;
}

Weight training_SMO(Weight weight, int class) { return weight; }

Weight training(Sample data, K_matrix matrix, Parameter parameter) {

    Weight weight = {0};
    weight = create_weight(data.classes, data.size);

    int ret = 0;
    int i = 0, j = 0;
    int i2 = 0;
    int epoch = 0;
    int class_p = 0, class_n = 0;
    clock_t start = 0, finish = 0;
    float use_time = 0;

    float *yk, *alpha;

    for (epoch = 0; epoch < parameter.max_epoch; epoch++) {
        for (i = 0; i < data.classes; i++) {
            class_p = i;
            for (i2 = 0; i2 < data.classes; i2++) {
                class_n = i2;
                if (class_n == class_p)
                    continue;
                yk = (float *)calloc(data.length, sizeof(float));

                int count_n = 0;
                int count_p = 0;
                int count_0 = 0;
                int total = 0;

                for (j = 0; j < data.length; j++) {
                    if (data.label[j] == class_p) {
                        yk[j] = 1;
                        count_p++;
                    } else if (data.label[j] == class_n) {
                        yk[j] = -1;
                        count_n++;
                    } else {
                        yk[j] = 0;
                        count_0++;
                    }
                }

                ret =
                    print_starting(class_p, class_n, count_0, count_n, count_p);

                start = clock();

                weight = training_SMO(weight, class_p * data.classes + class_n);

                finish = clock();

                use_time = (float)(finish - start) / 1000000;

                ret = print_complete(class_p, class_n, weight.num_svecter,
                                     use_time);

                free(yk);
            }
        }
    }

    return weight;
}

int main(int argc, char **argv) {

    int size = 784;
    int width = 28;
    int height = 28;
    int length = 42000;
    int classes = 10;

    assert(argc != 1);

    Sample sample = {0};
    sample = prepare_sample(argv[1], size, height, width, length, classes);

    K_matrix matrix = {0};
    matrix = compute_k_matrix(sample);

    Parameter parameter = {0};
    int epoch = 100;
    float c = 1;
    float ero = 1e-4;
    parameter = create_parameter(epoch, c, ero);

    Weight weight = {0};
    weight = training(sample, matrix, parameter);

    return 0;
}
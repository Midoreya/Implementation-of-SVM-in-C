#include "common.h"

#include "inference.h"
#include "preprocess.h"
#include "training.h"

int main(int argc, char **argv) {

    int size = 784;
    int width = 28;
    int height = 28;
    int classes = 10;
    int length = 42000;

    int training_length = 35000;
    int val_length = 5000;
    int test_length = 2000;

    int max_epoch = 2;
    int smo_epoch = 200;
    float c = 1;
    float ero = 1e-4;

    max_epoch = 2;
    smo_epoch = 10;

    // length = 10000;
    // training_length = 5000;
    // val_length = 3000;
    // test_length = 2000;

    int mode = -1;
    if (strcmp(argv[1], "training") == 0)
        mode = 0;
    else if (strcmp(argv[1], "inf") == 0)
        mode = 1;
    else
        mode = -1;
    assert(mode != -1);

    if (mode == 0) {

        Sample sample = {0};
        sample = create_sample(argv[2], size, height, width, length, classes);

        Split_length s_length = {0};
        s_length = create_length(training_length, val_length, test_length);

        Dataset data = {0};
        data = dataloader(sample, s_length);

        K_matrix matrix = {0};
        matrix = compute_k_matrix(sample);

        Parameter parameter = {0};
        parameter = create_parameter(max_epoch, smo_epoch, c, ero);

        Weight weight = {0};
        if (argc > 3)
            weight = load_weight(argv[3], sample.size, sample.classes);
        else
            weight = create_weight(sample.classes, sample.size);

        weight = training(data.training, data.val, matrix, parameter, weight);

        int ret = 0;
        ret = write_weight(weight, sample.classes, sample.size,
                           "./weight/weight.txt");
        assert(ret == 0);
    }

    if (mode == 1) {

        Sample sample = {0};
        sample = create_sample(argv[3], size, height, width, length, classes);

        Weight weight = {0};
        weight = load_weight(argv[4], sample.size, sample.classes);

        int re = -1;
        re = inference(atoi(argv[2]), sample, weight);
        assert(re != -1);
    }

    return 0;
}
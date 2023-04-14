#include "common.h"

#include "preprocess.h"
#include "training.h"

int main(int argc, char **argv) {

    int size = 784;
    int width = 28;
    int height = 28;
    int length = 5000;
    int classes = 10;

    int training_length = 3000;
    int val_length = 1000;
    int test_length = 1000;

    int max_epoch = 1;
    int smo_epoch = 100;
    float c = 1;
    float ero = 1e-4;

    assert(argc != 1);

    Sample sample = {0};
    sample = create_sample(argv[1], size, height, width, length, classes);

    Split_length s_length = {0};
    s_length = create_length(training_length, val_length, test_length);

    Dataset data = {0};
    data = dataloader(sample, s_length);

    K_matrix matrix = {0};
    matrix = compute_k_matrix(sample);

    Parameter parameter = {0};
    parameter = create_parameter(max_epoch, smo_epoch, c, ero);

    Weight weight = {0};
    weight = training(data.training, data.val, matrix, parameter);

    int ret = 0;
    ret = write_weight(weight, sample.classes, sample.size, "weight.txt");
    assert(ret == 0);

    return 0;
}
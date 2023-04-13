#include "common.h"

#include "preprocess.h"
#include "training.h"



int main(int argc, char **argv) {

    int size = 784;
    int width = 28;
    int height = 28;
    int length = 42000;
    int classes = 10;

    int training_length = 35000;
    int val_length = 5000;
    int test_length = 2000;

    assert(argc != 1);

    Sample sample = {0};
    sample = create_sample(argv[1], size, height, width, length, classes);

    Split_length s_length = {0};
    s_length = create_length(training_length, val_length, test_length);

    Dataset data = {0};
    data = dataloader(sample, s_length);

    // exit(0);

    K_matrix matrix = {0};
    matrix = compute_k_matrix(sample);

    exit(0);

    Parameter parameter = {0};
    int epoch = 100;
    float c = 1;
    float ero = 1e-4;
    parameter = create_parameter(epoch, c, ero);

    Weight weight = {0};
    weight = training(data.training, data.val, matrix, parameter);

    return 0;
}
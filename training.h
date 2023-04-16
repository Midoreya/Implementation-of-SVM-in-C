#ifndef _TRAINING_H
#define _TRAINING_H

int print_starting(int class_p, int class_n, int count_0, int count_n,
                   int count_p, int max_epoch, int epoch);

int print_complete(Acc acc, int class_p, int class_n, int count,
                   float use_time);

Weight training(Sample data, Sample val, K_matrix matrix, Parameter parameter,
                Weight weight);

#endif
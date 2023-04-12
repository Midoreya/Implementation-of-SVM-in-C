#ifndef _PREPROCESS_H
#define _PREPROCESS_H

Sample prepare_sample(char *path, int size, int height, int width, int length,
                      int classes);

K_matrix compute_k_matrix(Sample sample);

Parameter create_parameter(int epoch, float c, float ero);

#endif
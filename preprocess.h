#ifndef _PREPROCESS_H
#define _PREPROCESS_H

Sample create_sample(char *path, int size, int height, int width, int length,
                     int classes);

K_matrix compute_k_matrix(Sample sample);

Parameter create_parameter(int max_epoch, int smo_epoch, float c, float ero);

Weight create_weight(int classes, int size);

Dataset copy_label(Dataset data, Sample sample);

Dataset copy_data(Dataset data, Sample sample);

Split_length create_length(int training, int val, int test);

int copy_message(Sample *dest, Sample *src, int length);

Dataset dataloader(Sample sample, Split_length s_length);

#endif
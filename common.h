#ifndef _COMMON_H
#define _COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <assert.h>
#include <sys/time.h>

typedef struct {
    int size;
    int width;
    int height;
    int length;
    int classes;
    int *label;
    float *data;
} Sample;

typedef struct {
    Sample training;
    Sample val;
    Sample test;
} Dataset;

typedef struct {
    int training_length;
    int val_length;
    int test_length;
} Split_length;

typedef struct {
    float *data;
    long dim;
} K_matrix;

typedef struct {
    float **w;
    float b;
    int num_svecter;
} Weight;

typedef struct {
    int max_epoch;
    float c;
    float ero;
} Parameter;

float get_pixel(Sample data, int index, int i, int j);

float get_usetime(struct timeval start, struct timeval end);

#endif
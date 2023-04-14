#ifndef _COMMON_H
#define _COMMON_H

#include <math.h>
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
    int smo_epoch;
    float c;
    float ero;
} Parameter;

typedef struct {
    int true;
    int false;
    float acc;
} Acc;

float get_pixel(int i, int j, Sample data);
float get_matrix(int i, int j, K_matrix matrix);
float get_usetime(struct timeval start, struct timeval end);

float max_float(float x, float y);
float min_float(float x, float y);

int write_weight(Weight weight, int class, int size, char *file);

#endif
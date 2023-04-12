#ifndef _COMMON_H
#define _COMMON_H

#include <stdio.h>
#include <stdlib.h>

#include <assert.h>
#include <time.h>

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
    float *data;
    long dim;
} K_matrix;

typedef struct {
    float **w;
    float b;
} Weight;

typedef struct {
    int max_epoch;
    float c;
    float ero;
} Parameter;

float get_pixel(Sample data, int index, int i, int j);

#endif
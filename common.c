#include "common.h"

float get_pixel(Sample sample, int index, int i, int j) {
    
    assert(index < sample.length && i < sample.width && j < sample.height);
    return sample.data[index * sample.size + i * sample.width + j];
}
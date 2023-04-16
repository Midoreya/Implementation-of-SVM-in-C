# Implementation-of-SVM-in-C

- [Implementation-of-SVM-in-C](#implementation-of-svm-in-c)
  - [Quick Start](#quick-start)
    - [1. Training](#1-training)
    - [2. Inference](#2-inference)


## Quick Start

### 1. Training

    make
    ./test training [Dataset path] [Pretrain weight path]

If not use pretrain weight, also:

    ./test training [Dataset path]

    

### 2. Inference

    make
    ./test inf [index] [Dataset path] [Pretrain weight path]

If only load one image(tensor), index = 0, like:

    ./test inf 0 [Dataset path] [Pretrain weight path]

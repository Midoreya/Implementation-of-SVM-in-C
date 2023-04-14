#include "common.h"

#include "preprocess.h"
#include "training.h"
#include "val.h"

float e(int i, int count, float *alpha, float *yk, K_matrix matrix, float b) {
    int j = 0;
    float result = 0;
    for (j = 0; j < count; j++) {
        result = result + alpha[j] * yk[j] * get_matrix(i, j, matrix);
    }
    result = result + b - yk[i];
    return result;
}

float l_operation(int u, int v, float *yk, float *alpha, float c) {
    if (yk[u] != yk[v])
        return max_float(0, alpha[u] - alpha[v]);
    else
        return max_float(0, alpha[u] + alpha[v] - c);
}

float h_operation(int u, int v, float *yk, float *alpha, float c) {
    if (yk[u] != yk[v])
        return min_float(c, alpha[u] - alpha[v] + c);
    else
        return min_float(c, alpha[u] + alpha[v]);
}

Weight compute_wb(Sample data, Weight weight, int class, float *alpha,
                  float *yk) {
    int i = 0, j = 0, d = 0;
    int *s_vector;

    s_vector = (int *)calloc(data.length, sizeof(int));

    for (i = 0; i < data.length; i++) {
        if (alpha[i] > 0) {
            s_vector[j] = i;
            j = j + 1;
        }
    }

    weight.num_svecter = j;

    for (d = 0; d < data.size; d++) {
        for (i = 0; i < j; i++) {
            weight.w[class][d] =
                weight.w[class][d] + alpha[s_vector[i]] * yk[s_vector[i]] *
                                         get_pixel(s_vector[i], d, data);
        }
    }

    free(s_vector);

    return weight;
}

int print_starting(int class_p, int class_n, int count_0, int count_n,
                   int count_p) {
    int total = count_n + count_p + count_0;
    printf("---------------------------------------\n");
    printf("Start class[%d], class[%d] training...\n", class_p, class_n);
    printf("Positive sample count: %d\n", count_p);
    printf("Negative sample count: %d\n", count_n);
    printf("Other sample count:    %d\n", count_0);
    printf("Total sample count:    %d\n\n", total);

    return 0;
}

int print_complete(Acc acc, int class_p, int class_n, int count,
                   float use_time) {

    printf("Support vecter count:  %d\n", count);
    printf("True:                  %d\n", acc.true);
    printf("False:                 %d\n", acc.false);
    printf("Accuracy:              %.2f%%\n", acc.acc * 100);
    printf("Using:                 %.2fs\n", use_time);
    printf("Complete class[%d], class[%d] training.\n", class_p, class_n);
    printf("---------------------------------------\n\n");

    return 0;
}

Weight training_SMO(Weight weight, K_matrix matrix, Sample data,
                    Parameter parameter, float *yk, int class) {
    int i = 0, j = 0;
    int passes = 0, num_changed_alphas = 0;
    float alpha_i_old = 0, alpha_j_old = 0;
    float l = 0, h = 0;
    float eta = 0;
    float bi = 0, bj = 0;
    int epoch = 0;

    float *alpha;
    alpha = (float *)calloc(data.length, sizeof(float));
    weight.b = 0;

    srand(100);

    while (passes < parameter.smo_epoch) {
        num_changed_alphas = 0;
        for (i = 0; i < data.length; i++) {
            if (yk[i] == 0)
                continue;

            if ((yk[i] * e(i, data.length, alpha, yk, matrix, weight.b) <
                     -parameter.ero &&
                 alpha[i] < parameter.c) ||
                (yk[i] * e(i, data.length, alpha, yk, matrix, weight.b) >
                     parameter.ero &&
                 alpha[i] > 0)) {
                do {
                    j = rand() % data.length;
                } while (j == i);

                alpha_i_old = alpha[i];
                alpha_j_old = alpha[j];

                epoch++;

                l = l_operation(j, i, yk, alpha, parameter.c);
                h = h_operation(j, i, yk, alpha, parameter.c);

                if (fabs(l - h) < parameter.ero)
                    continue;

                eta = 2 * get_matrix(i, j, matrix) - get_matrix(i, i, matrix) -
                      get_matrix(j, j, matrix);

                if (eta >= 0)
                    continue;

                alpha[j] =
                    alpha[j] -
                    yk[j] *
                        (e(i, data.length, alpha, yk, matrix, weight.b) -
                         e(j, data.length, alpha, yk, matrix, weight.b)) /
                        eta;

                if (alpha[j] > h)
                    alpha[j] = h;
                if (alpha[j] < l)
                    alpha[j] = l;

                if (fabs(alpha[j] - alpha_j_old) <= fabs(1e-4 * alpha[j]))
                    continue;

                alpha[i] = alpha[i] + yk[i] * yk[j] * (alpha_j_old - alpha[j]);

                bi =
                    weight.b - e(i, data.length, alpha, yk, matrix, weight.b) -
                    yk[i] * (alpha[i] - alpha_i_old) *
                        get_matrix(i, i, matrix) -
                    yk[j] * (alpha[j] - alpha_j_old) * get_matrix(i, j, matrix);
                bj =
                    weight.b - e(j, data.length, alpha, yk, matrix, weight.b) -
                    yk[i] * (alpha[i] - alpha_i_old) *
                        get_matrix(i, j, matrix) -
                    yk[j] * (alpha[j] - alpha_j_old) * get_matrix(j, j, matrix);

                if (alpha[i] < parameter.c && alpha[i] > 0)
                    weight.b = bi;
                else if (alpha[j] < parameter.c && alpha[j] > 0)
                    weight.b = bj;
                else
                    weight.b = (bi + bj) * 0.5;
                num_changed_alphas = num_changed_alphas + 1;
            }

            if (epoch > parameter.smo_epoch) {
                epoch = 0;
                passes = passes + 1;
            }

            if (num_changed_alphas == 0)
                passes = passes + 1;
            else
                passes = 0;
        }
    }

    weight = compute_wb(data, weight, class, alpha, yk);
    weight.w[class][data.size] = weight.b;

    return weight;
}

Weight training(Sample data, Sample val, K_matrix matrix, Parameter parameter) {

    Weight weight = {0};
    weight = create_weight(data.classes, data.size);

    int ret = 0;
    int i = 0, j = 0;
    int i2 = 0;
    int epoch = 0;
    int class_p = 0, class_n = 0;

    float use_time = 0;
    struct timeval start, end;

    float *yk;
    for (epoch = 0; epoch < parameter.max_epoch; epoch++) {
        for (i = 0; i < data.classes; i++) {
            class_p = i;
            for (i2 = 0; i2 < data.classes; i2++) {
                class_n = i2;
                if (class_n == class_p)
                    continue;
                yk = (float *)calloc(data.length, sizeof(float));

                int count_n = 0;
                int count_p = 0;
                int count_0 = 0;

                for (j = 0; j < data.length; j++) {
                    if (data.label[j] == class_p) {
                        yk[j] = 1;
                        count_p++;
                    } else if (data.label[j] == class_n) {
                        yk[j] = -1;
                        count_n++;
                    } else {
                        yk[j] = 0;
                        count_0++;
                    }
                }

                ret =
                    print_starting(class_p, class_n, count_0, count_n, count_p);
                assert(ret == 0);

                gettimeofday(&start, NULL);

                weight = training_SMO(weight, matrix, data, parameter, yk,
                                      class_p * data.classes + class_n);

                Acc acc = {0};
                acc = validation(val, weight);

                gettimeofday(&end, NULL);
                use_time = get_usetime(start, end);

                ret = print_complete(acc, class_p, class_n, weight.num_svecter,
                                     use_time);
                assert(ret == 0);
            }
        }
    }

    return weight;
}
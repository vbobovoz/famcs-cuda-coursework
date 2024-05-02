#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <stdio.h>
#include <stdlib.h> // Для функции rand()
#include <time.h>  // Для инициализации генератора случайных чисел

#define TILE_DIM 16

// Функция-ядро
__global__ void mulKernel(int *A, int *B, int *C, int N) {
    // Создание 2 тайлов для матрицы A и B в shared memory
    __shared__ int ATile[TILE_DIM][TILE_DIM];
    __shared__ int BTile[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int thrX = threadIdx.x;
    int thrY = threadIdx.y;

    int elementC = 0;

    for(int t = 0; t < (N - 1) / TILE_DIM + 1; ++t) {
        // Потоки для загрузки матрицы A в shared memory
        if(row < N && t * TILE_DIM + thrX < N)
            ATile[thrY][thrX] = A[row * N + t * TILE_DIM + thrX];
        else
            ATile[thrY][thrX] = 0.0f;

        // Потоки для загрузки матрицы B в shared memory
        if(t * TILE_DIM + thrY < N && col < N)
            BTile[thrY][thrX] = B[(t * TILE_DIM + thrY) * N + col];
        else
            BTile[thrY][thrX] = 0.0f;

        __syncthreads();

        // Вычисление частичного значения для матрицы C
        for(int i = 0; i < TILE_DIM; ++i)
            elementC += ATile[thrY][i] * BTile[i][thrX];

        __syncthreads();

    }
    // Копирование конечного значения в матрицу C
    if(row < N && col < N)
        C[row * N + col] = elementC;

}

// Получение результатов на CPU + tiling
void matrix_mul_cpu_tiling(int *a, int *b, int *c, int N) {
    // Обнуление результирующей матрицы
    for(int i = 0; i < N * N; i++) {
        c[i] = 0;
    }

    // Перемножение матриц с использованием разбиения на плитки
    for(int i = 0; i < N; i += TILE_DIM) { // строки
        for(int j = 0; j < N; j += TILE_DIM) { // столбцы
            for(int k = 0; k < N; k += TILE_DIM) { // внутренний размер плитки

                // Подплиточные границы
                const int minI = std::min(i + TILE_DIM, N);
                const int minJ = std::min(j + TILE_DIM, N);
                const int minK = std::min(k + TILE_DIM, N);

                // Перемножение в пределах плитки
                for(int ii = i; ii < minI; ii++) { // строки плитки
                    for(int jj = j; jj < minJ; jj++) { // столбцы плитки
                        int sum = 0; // промежуточное значение
                        for(int kk = k; kk < minK; kk++) { // перемножение в пределах плитки
                            sum += a[ii * N + kk] * b[kk * N + jj];
                        }
                        c[ii * N + jj] += sum; // накопление результата
                    }
                }
            }
        }
    }
}

// Получение результатов на CPU
void matrix_mul_cpu(int *a, int *b, int *c, int N) {
    int tmp;
    for(int i = 0; i < N; ++i) { // Для каждой строки
        for(int j = 0; j < N; ++j) { // Для каждого столбца
            tmp = 0;
            for(int k = 0; k < N; ++k) { // Для каждого элемента в этой строке и столбце
                tmp += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = tmp;
        }
    }
}

// void verify_results(int* arr1, int* arr2, int* arr3, int size) {
//     for(int i = 0; i < size; i++) {
//         if(arr1[i] != arr2[i] || arr1[i] != arr3[i] || arr2[i] != arr3[i]) {
//             printf("Arrays are NOT the same!\n");
//             return;
//         }
//     }
//     printf("Arrays are the same!\n");
// }

void print_all_matrix(int *hostA, int *hostB, int *hostC, int *hostCCPU, int *hostCCPUTiling, int N) {
    // Вывод матрицы A
    printf("Matrix A:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%d ", hostA[i * N + j]);
        }
        printf("\n");
    }

    // Вывод матрицы C
    printf("Matrix B:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%d ", hostB[i * N + j]);
        }
        printf("\n");
    }

    // Вывод матрицы C
    printf("Matrix C:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%d ", hostC[i * N + j]);
        }
        printf("\n");
    }

    // Вывод матрицы CCPU
    printf("Matrix C_CPU:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%d ", hostCCPU[i * N + j]);
        }
        printf("\n");
    }

    // Вывод матрицы CCPUTILING
    printf("Matrix C_CPUTILING:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%d ", hostCCPUTiling[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    int *hostA, *hostB, *hostC, *hostCCPUTiling, *hostCCPU;
    int *deviceA, *deviceB, *deviceC;
    int N;

    // Создание переменных-событий
    float timerValueGPU, timerValueCPU, timerValueCPUTiling;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    N = 2048; // Размерность матриц

    // Инициализация генератора случайных чисел
    srand(time(NULL));
    
    // Выделение памяти на хосте
    hostA = (int *)malloc(N * N * sizeof(int));
    hostB = (int *)malloc(N * N * sizeof(int));
    hostC = (int *)malloc(N * N * sizeof(int));
    hostCCPU = (int *)malloc(N * N * sizeof(int));
    hostCCPUTiling = (int *)malloc(N * N * sizeof(int));

    // Выделение памяти на устройстве (GPU)
    cudaMalloc((void **) &deviceA, N * N * sizeof(int));
    cudaMalloc((void **) &deviceB, N * N * sizeof(int));
    cudaMalloc((void **) &deviceC, N * N * sizeof(int));
    
    // Конфигурация сетки и блока
    dim3 DimGrid((N - 1) / TILE_DIM + 1, (N - 1) / TILE_DIM + 1, 1);
    dim3 DimBlock(TILE_DIM, TILE_DIM, 1);

    // Генерация случайных матриц
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            hostA[i * N + j] = ((int)rand() % 101) - 50; // значения от -50 до 50
        }
    }

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            hostB[i * N + j] = ((int)rand() % 101) - 50; // значения от -50 до 50
        }
    }

    // ------------------------ GPU-tiling-вариант ------------------------
    // Запуск таймера
    cudaEventRecord(start, 0);

    // Копирование данных на GPU
    cudaMemcpy(deviceA, hostA, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Запуск ядра
    mulKernel<<<DimGrid,DimBlock>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();

    // Копирование результата обратно на хост
    cudaMemcpy(hostC, deviceC, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Оценка времени вычисления GPU-варианта
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);
    printf("\n GPU tiling calculation time %f msec\n", timerValueGPU);
    // --------------------------------------------------------------------

    // Освобождение памяти на устройстве
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    // ------------------------ CPU_tiling-вариант ------------------------
    // Запуск таймера
    cudaEventRecord(start, 0);

    // Запуск функции
    matrix_mul_cpu_tiling(hostA, hostB, hostCCPUTiling, N);

    // Оценка времени вычисления GPU-варианта
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueCPUTiling, start, stop);
    printf("\n CPU tiling calculation time %f msec\n", timerValueCPUTiling);
    // -----------------------------------------------------------------------

    // ------------------------ CPU_no_tiling-вариант ------------------------
    // Запуск таймера
    cudaEventRecord(start, 0);

    // Запуск функции
    matrix_mul_cpu(hostA, hostB, hostCCPU, N);

    // Оценка времени вычисления GPU-варианта
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueCPU, start, stop);
    printf("\n CPU calculation time %f msec\n", timerValueCPU);
    // -----------------------------------------------------------------------
    
    // Вывод ускорения
    printf("\n Acceleration between NoTilingCPU and TilingGPU %fx\n", timerValueCPU / timerValueGPU);
    printf("\n Acceleration between TilingCPU and TilingGPU %fx\n", timerValueCPUTiling / timerValueGPU);
    printf("\n Acceleration between NoTilingCPU and TilingCPU %fx\n", timerValueCPU / timerValueCPUTiling);

    // Проверка решения
    // verify_results(hostC, hostCCPU, hostCCPUTiling, N);

    // Вывод всех матриц
    // print_all_matrix(hostA, hostB, hostC, hostCCPU, hostCCPUTiling, N);

    // Освобождение памяти на хосте
    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

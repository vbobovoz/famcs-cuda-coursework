#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <stdio.h>
#include <stdlib.h> // Для функции rand()
#include <time.h>  // Для инициализации генератора случайных чисел

#define TILE_DIM 16

// Функция-ядро без тайлинга
__global__ void mulKernelGlobal(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N) {
        int sum = 0;
        for(int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Функция-ядро с tiling
__global__ void mulKernelTiling(int *A, int *B, int *C, int N) {
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

int main() {
    int *hostA, *hostB, *hostCGPUTiling, *hostCGPU, *hostCCPUTiling, *hostCCPU;
    int *deviceA, *deviceB, *deviceC;
    int N;

    // Создание переменных-событий
    float timerValueGPU, timerValueGPUTiling, timerValueCPU, timerValueCPUTiling;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    N = 2048; // Размерность матриц

    // Инициализация генератора случайных чисел
    srand(time(NULL));
    
    // Выделение памяти на хосте
    hostA = (int *)malloc(N * N * sizeof(int));
    hostB = (int *)malloc(N * N * sizeof(int));
    hostCGPU = (int *)malloc(N * N * sizeof(int));
    hostCCPU = (int *)malloc(N * N * sizeof(int));
    hostCGPUTiling = (int *)malloc(N * N * sizeof(int));
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

    // Вывод времени
    printf("\nTIME:");

    // ------------------------ GPU-tiling-вариант ------------------------
    // Запуск таймера
    cudaEventRecord(start, 0);

    // Копирование данных на GPU
    cudaMemcpy(deviceA, hostA, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Запуск ядра
    mulKernelTiling<<<DimGrid,DimBlock>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();

    // Копирование результата обратно на хост
    cudaMemcpy(hostCGPUTiling, deviceC, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Оценка времени вычисления GPU-варианта
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPUTiling, start, stop);
    printf("\n GPU-tiling    %f msec    ", timerValueGPUTiling);
    // --------------------------------------------------------------------

    // ------------------------ GPU-no-tiling-вариант ------------------------
    // Запуск таймера
    cudaEventRecord(start, 0);

    // Копирование данных на GPU
    cudaMemcpy(deviceA, hostA, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Запуск ядра
    mulKernelGlobal<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();

    // Копирование результата обратно на хост
    cudaMemcpy(hostCGPU, deviceC, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Оценка времени вычисления GPU-варианта без tiling
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);
    printf("\n GPU-No-Tiling %f msec ", timerValueGPU);
    // -----------------------------------------------------------------------

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
    printf("\n CPU-tiling    %f msec    ", timerValueCPUTiling);
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
    printf("\n CPU-No-Tiling %f msec \n", timerValueCPU);
    // -----------------------------------------------------------------------
    
    // Вывод ускорения
    printf("\nACCELERATION:");
    printf("\n NoTilingCPU / TilingGPU   %fx", timerValueCPU / timerValueGPUTiling);
    printf("\n NoTilingCPU / NoTilingGPU %fx", timerValueCPU / timerValueGPU);
    printf("\n NoTilingCPU / TilingCPU   %fx", timerValueCPU / timerValueCPUTiling);
    printf("\n TilingCPU   / TilingGPU   %fx", timerValueCPUTiling / timerValueGPUTiling);
    printf("\n TilingCPU   / NoTilingGPU %fx", timerValueCPUTiling / timerValueGPU);
    printf("\n NoTilingGPU / TilingGPU   %fx", timerValueGPU / timerValueGPUTiling);

    // Освобождение памяти на хосте
    free(hostA);
    free(hostB);
    free(hostCCPU);
    free(hostCGPU);
    free(hostCCPUTiling);
    free(hostCGPUTiling);

    return 0;
}

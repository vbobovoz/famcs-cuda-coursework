#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <omp.h>

using namespace std;

#define TILE_DIM 16


// --------------------------------------- PRINT ---------------------------------------
void printMatrix(double* A, int N, int precision) {
    cout << fixed << setprecision(precision); // устанавливаем точность вывода
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            cout << A[i * N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void printVector(double* v, int N, int precision) {
    cout << fixed << setprecision(precision); // устанавливаем точность вывода
    for(int i = 0; i < N; ++i) {
        cout << v[i] << " ";
    }
    cout << endl << endl;
}
// -------------------------------------------------------------------------------------

// --------------------------------- NORM-CALCULATION ---------------------------------
void multiplyMatrixVector(double* A, double* x, double* result, int N) {
    for(int i = 0; i < N; ++i) {
        result[i] = 0;
        for(int j = 0; j < N; ++j) {
            result[i] += A[i * N + j] * x[j];
        }
    }
}

double computeNorm(double* Ax, double* b, int N) {
    double norm = 0;
    for(int i = 0; i < N; ++i) {
        norm += (Ax[i] - b[i]) * (Ax[i] - b[i]);
    }
    return sqrt(norm);
}

void multiplyMatrixVectorOMP(double* A, double* x, double* result, int N) {
    #pragma omp parallel for
    for(int i = 0; i < N; ++i) {
        result[i] = 0;
        for(int j = 0; j < N; ++j) {
            result[i] += A[i * N + j] * x[j];
        }
    }
}

double computeNormOMP(double* Ax, double* b, int N) {
    double norm = 0;
    #pragma omp parallel for reduction(+:norm)
    for(int i = 0; i < N; ++i) {
        norm += (Ax[i] - b[i]) * (Ax[i] - b[i]);
    }
    return sqrt(norm);
}
// ------------------------------------------------------------------------------------

// ----------------------------------- CPU-NO-TILING -----------------------------------
void CPU_NO_TILING_Seidel_Method(double* A, double* f, double* x, int N, double eps) {
    int iterations = 0;
    double* x_old = new double[N];
    double* Ax = new double[N]();

    while(true) {
        for(int i = 0; i < N; ++i) {
            x_old[i] = x[i];
        }

        for(int i = 0; i < N; ++i) {
            double sigma = 0.0;
            for(int j = 0; j < N; ++j) {
                if(j != i) {
                    sigma += A[i * N + j] * x[j];
                }
            }
            x[i] = (f[i] - sigma) / A[i * N + i];
        }
        
        iterations++;

        // Вычисляем Ax^(k)
        multiplyMatrixVector(A, x, Ax, N);

        // Проверка условия остановки
        if(computeNorm(Ax, f, N) <= eps) {
            break;
        }
    }

    cout << "\nNumber of iterations: " << iterations;
    delete[] x_old;
}
// -------------------------------------------------------------------------------------

// ------------------------------------- CPU-OMP -------------------------------------
void CPU_Parallel_Seidel_Method(double* A, double* f, double* x, int N, double eps) {
    int iterations = 0;
    double* x_old = new double[N];
    double* Ax = new double[N]();

    while(true) {
        // Копируем текущее решение в x_old
        #pragma omp parallel for
        for(int i = 0; i < N; ++i) {
            x_old[i] = x[i];
        }

        // Основной цикл метода Зейделя
        #pragma omp parallel for
        for(int i = 0; i < N; ++i) {
            double sigma = 0.0;
            for(int j = 0; j < N; ++j) {
                if(j != i) {
                    sigma += A[i * N + j] * x[j];
                }
            }
            x[i] = (f[i] - sigma) / A[i * N + i];
        }

        iterations++;

        // Вычисляем Ax^(k)
        multiplyMatrixVectorOMP(A, x, Ax, N);

        // Проверка условия остановки
        if(computeNormOMP(Ax, f, N) <= eps) {
            break;
        }
    }

    std::cout << "\nNumber of iterations: " << iterations;
    delete[] x_old;
    delete[] Ax;
}
// -----------------------------------------------------------------------------------

// ---------------------------------- GPU-No-Tiling ----------------------------------
__global__ void seidel_no_tiling_Kernel(double* A, double* f, double* x, double* x_old, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N) {
        double sigma = 0.0;
        for(int j = 0; j < N; ++j) {
            if(j != i) {
                sigma += A[i * N + j] * x[j];
            }
        }
        x[i] = (f[i] - sigma) / A[i * N + i];
    }
}

void GPU_NO_TILING_Seidel_Method(double* A, double* f, double* x, int N, double eps) {
    int iterations = 0;
    double* x_old = new double[N];
    double* Ax = new double[N]();

    int threads_per_block = 16;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    double *d_A, *d_f, *d_x, *d_x_old;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_f, N * sizeof(double));
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_x_old, N * sizeof(double));
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

    while(true) {
        cudaMemcpy(d_x_old, d_x, N * sizeof(double), cudaMemcpyDeviceToDevice);
        seidel_no_tiling_Kernel<<<threads_per_block, blocks_per_grid>>>(d_A, d_f, d_x, d_x_old, N);
        cudaDeviceSynchronize();

        iterations++;

        cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
        multiplyMatrixVector(A, x, Ax, N);
        if(computeNorm(Ax, f, N) <= eps) {
            break;
        }
    }

    std::cout << "\nNumber of iterations: " << iterations;

    cudaFree(d_A);
    cudaFree(d_f);
    cudaFree(d_x);
    cudaFree(d_x_old);
    delete[] x_old;
}
// -----------------------------------------------------------------------------------

// ------------------------------------- GENERATION -------------------------------------
// Генерация матрицы с диагональным доминированием
void generateMatrix(double* A, int N) {
    double min = 0;
    double max = 50;
    for(int i = 0; i < N * N; ++i) {
        double tmp = (double)rand() / RAND_MAX; // значение в пределах [0; 1]
        A[i] = min + tmp * (max - min); // присваивание значения в пределах [0; 50]
    }

    for(int i = 0; i < N; ++i) {
        double sum = 0;
        for(int j = 0; j < N; ++j) {
            if(i != j) {
                sum += A[i * N + j];
            }
        }
        // A[i * N + i] += sum + 1; // диагональное доминирование
        A[i * N + i] = sum + 1; // диагональное доминирование
    }
}

// Генерация вектора
void generateVector(double* f, int N) {
    double min = 0;
    double max = 50;
    for(int i = 0; i < N; ++i) {
        double tmp = (double)rand() / RAND_MAX; // значение в пределах [0; 1]
        f[i] = min + tmp * (max - min); // присваивание значения в пределах [0; 50]
    }
}
// --------------------------------------------------------------------------------------

int main() {
    srand(time(NULL)); // инициализация генератора случайных чисел
 
    int N = 512; // размер матрицы и вектора
    double eps = 1e-5; // погрешность
    int precision = 10; // количество знаков после запятой в выводе

    double* A = new double[N * N]; // матрица A
    double* f = new double[N]; // вектор f
    double* x_cpu = new double[N](); // начальное приближение CPU_NO_TILING (нулевой вектор)
    double* x_cpu_omp = new double[N](); // начальное приближение CPU_OMP (нулевой вектор)
    double* x_gpu = new double[N](); // начальное приближение GPU_NO_TILING (нулевой вектор)

    // События для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Генерация матрицы и вектора
    generateMatrix(A, N);
    generateVector(f, N);

    // Вывод времени
    printf("\nTIME:");

    // -------------------------------- CPU --------------------------------    
    cudaEventRecord(start, 0); // старт таймера
    CPU_NO_TILING_Seidel_Method(A, f, x_cpu, N, eps); // вызов метода Якоби

    float timerValueCPU;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueCPU, start, stop);
    printf("\nCPU         %f msec", timerValueCPU);
    // ---------------------------------------------------------------------    

    // -------------------------------- CPU-OMP --------------------------------
    cudaEventRecord(start, 0); // старт таймера
    CPU_Parallel_Seidel_Method(A, f, x_cpu_omp, N, eps); // вызов метода Якоби

    float timerValueCPUOpenMP;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueCPUOpenMP, start, stop);
    printf("\nCPU(OpenMP) %f msec\n", timerValueCPUOpenMP);
    // -------------------------------------------------------------------------

    // -------------------------------- GPU --------------------------------
    cudaEventRecord(start, 0); // старт таймера
    GPU_NO_TILING_Seidel_Method(A, f, x_gpu, N, eps); // вызов метода Якоби

    float timerValueGPU;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);
    printf("\nGPU         %f msec\n", timerValueGPU);
    // ---------------------------------------------------------------------

    // Освобождение памяти
    delete[] A;
    delete[] f;
    delete[] x_cpu;
    delete[] x_cpu_omp;
    delete[] x_gpu;

    return 0;
}
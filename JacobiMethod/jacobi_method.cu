#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <omp.h>

#define DIM 16

using namespace std;


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
// Метод Якоби для решения СЛАУ Ax = f
void CPU_NO_TILING_Jacobi_Method(double* A, double* f, double* x, int N, double eps) {
    double* x_prev = new double[N]();
    double* Ax = new double[N]();
    int iterations = 0;

    while(true) { // цикл продолжается до достижения критерия остановки
        // Копируем текущее значение x в x_prev перед каждой итерацией
        for(int i = 0; i < N; ++i) {
            x_prev[i] = x[i];
        }

        for(int i = 0; i < N; ++i) {
            double sum = 0;

            for(int j = 0; j < N; ++j) {
                if(j != i) {
                    sum += A[i * N + j] * x_prev[j];
                }
            }
            
            // Получаем новое приближение x
            x[i] = (f[i] - sum) / A[i * N + i];
        }

        iterations++;

        // Вычисляем Ax^(k)
        multiplyMatrixVector(A, x, Ax, N);

        // Проверка условия остановки
        if(computeNorm(Ax, f, N) <= eps) {
            break;
        }
    }

    delete[] x_prev; // освобождаем память
    delete[] Ax; // освобождаем память
    cout << "Iteration count: " << iterations;
}
// -------------------------------------------------------------------------------------

// ------------------------------------- CPU-OMP -------------------------------------
// Метод Якоби + OpenMP
void CPU_Parallel_Jacobi_Method(double* A, double* f, double* x, int N, double eps) {
    double* x_prev = new double[N]();
    double* Ax = new double[N]();
    int iterations = 0;

    while(true) {
        // Копируем текущее значение x в x_prev параллельно
        #pragma omp parallel for
        for(int i = 0; i < N; ++i) {
            x_prev[i] = x[i];
        }

        // Новое приближение x параллельно
        #pragma omp parallel for
        for(int i = 0; i < N; ++i) {
            double sum = 0;

            // Суммируем элементы в строке, исключая диагональ
            #pragma omp simd
            for(int j = 0; j < N; ++j) {
                if(j != i) {
                    sum += A[i * N + j] * x_prev[j];
                }
            }

            x[i] = (f[i] - sum) / A[i * N + i];
        }

        iterations++;

        // Вычисляем Ax^(k)
        multiplyMatrixVectorOMP(A, x, Ax, N);

        // Проверка условия остановки
        if(computeNormOMP(Ax, f, N) <= eps) {
            break;
        }
    }

    delete[] x_prev; // освобождаем память
    delete[] Ax; // освобождаем память
    cout << "Iteration count: " << iterations;
}
// -----------------------------------------------------------------------------------

// ---------------------------------- GPU-No-Tiling ----------------------------------
__global__ void jacobi_no_tiling_Kernel(double* x_next, const double* A, const double* x_now, const double* b_h, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N) {
        double sigma = 0.0;
        for(int j = 0; j < N; j++) {
            if(i != j) {
                sigma += A[i * N + j] * x_now[j];
            }
        }
        x_next[i] = (b_h[i] - sigma) / A[i * N + i];
    }
}

void GPU_NO_TILING_Jacobi_Method(double* A, double* f, double* x, int N, double eps) {
    // Размеры блока и сетки
    dim3 DimGrid((N + DIM - 1) / DIM, (N + DIM - 1) / DIM, 1);
    dim3 DimBlock(DIM, DIM, 1);

    // Выделяем память на устройстве
    double* d_A;
    double* d_x_now;
    double* d_x_next;
    double* d_f;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_x_now, N * sizeof(double));
    cudaMalloc(&d_x_next, N * sizeof(double));
    cudaMalloc(&d_f, N * sizeof(double));

    // Копируем данные Host->Device
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_now, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, N * sizeof(double), cudaMemcpyHostToDevice);

    double* x_prev = new double[N];
    double* Ax = new double[N]();
    int iteration = 0;

    while(true) {
        // Копируем текущее приближение в x_prev
        cudaMemcpy(x_prev, d_x_now, N * sizeof(double), cudaMemcpyDeviceToHost);

        // Запускаем ядро Якоби
        jacobi_no_tiling_Kernel<<<DimGrid, DimBlock>>>(d_x_next, d_A, d_x_now, d_f, N);

        // Копируем результат Device->Host для проверки условия остановки
        cudaMemcpy(x, d_x_next, N * sizeof(double), cudaMemcpyDeviceToHost);
        
        iteration++;

        // Вычисляем Ax^(k)
        multiplyMatrixVector(A, x, Ax, N);

        // Проверка условия остановки
        if(computeNorm(Ax, f, N) <= eps) {
            break;
        }

        // Обновляем x_now для следующей итерации
        cudaMemcpy(d_x_now, d_x_next, N * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Вывод числа итераций
    cout << "Iteration count: " << iteration;

    // Освобождаем память
    cudaFree(d_A);
    cudaFree(d_x_now);
    cudaFree(d_x_next);
    cudaFree(d_f);
    delete[] x_prev;
    delete[] Ax;
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

int main() {
    srand(time(NULL)); // инициализация генератора случайных чисел
 
    int N = 256; // размер матрицы и вектора
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

    // cout << "Матрица A:" << endl;
    // printMatrix(A, N, precision);

    // cout << "Вектор f:" << endl;
    // printVector(f, N, precision);

    // Вывод времени
    printf("\nTIME:");

    // -------------------------------- CPU --------------------------------    
    cudaEventRecord(start, 0); // старт таймера
    CPU_NO_TILING_Jacobi_Method(A, f, x_cpu, N, eps); // вызов метода Якоби

    float timerValueCPU;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueCPU, start, stop);
    printf("\n СPU         %f msec", timerValueCPU);

    // Вывод результата
    // cout << "Решение x_cpu:" << endl;
    // printVector(x_cpu, N, precision);
    // ---------------------------------------------------------------------    

    // -------------------------------- CPU-OMP --------------------------------
    cudaEventRecord(start, 0); // старт таймера
    CPU_Parallel_Jacobi_Method(A, f, x_cpu_omp, N, eps); // вызов метода Якоби

    float timerValueCPUOpenMP;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueCPUOpenMP, start, stop);
    printf("\n СPU(OpenMP) %f msec", timerValueCPUOpenMP);

    // Вывод результата
    // cout << "Решение x_cpu_omp:" << endl;
    // printVector(x_cpu_omp, N, precision);
    // -------------------------------------------------------------------------

    // -------------------------------- GPU --------------------------------
    cudaEventRecord(start, 0); // старт таймера
    GPU_NO_TILING_Jacobi_Method(A, f, x_gpu, N, eps); // вызов метода Якоби

    float timerValueGPU;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);
    printf("\n GPU         %f msec", timerValueGPU);

    // Вывод результата
    // cout << "Решение x_gpu:" << endl;
    // printVector(x_gpu, N, precision);
    // ---------------------------------------------------------------------

    // Вывод ускорения
    printf("\nACCELERATION:");
    printf("\n CPU / CPU(OpenMP)  %fx", timerValueCPU / timerValueCPUOpenMP);
    printf("\n CPU / GPU          %fx", timerValueCPU / timerValueGPU);
    printf("\n CPU(OpenMP) / GPU  %fx", timerValueCPUOpenMP / timerValueGPU);

    // Освобождение памяти
    delete[] A;
    delete[] f;
    delete[] x_cpu;
    delete[] x_cpu_omp;
    delete[] x_gpu;

    return 0;
}

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <omp.h>

// #define TILE_SIZE 16

using namespace std;

// ----------------------------------- CPU-NO-TILING -----------------------------------
// Вычисления нормы
double norm(double* x, double* x_prev, int N) {
    double sum = 0;
    for(int i = 0; i < N; ++i) {
        sum += (x[i] - x_prev[i]) * (x[i] - x_prev[i]);
    }
    return sqrt(sum);
}

// Метод Якоби для решения СЛАУ Ax = f
void CPU_NO_TILING_Jacobi_Method(double* A, double* f, double* x, int N, double eps) {
    double* x_prev = new double[N]();

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

        // Проверка условия остановки
        if(norm(x, x_prev, N) <= eps) {
            break;
        }
    }

    delete[] x_prev; // освобождаем память
    cout << "Iteration count: " << iterations;
}
// -------------------------------------------------------------------------------------

// ------------------------------------- CPU-OMP -------------------------------------
// Вычисления нормы
double norm_omp(double* x, double* x_prev, int N) {
    double sum = 0;
    #pragma omp parallel for reduction(+:sum) // параллельное сложение
    for(int i = 0; i < N; ++i) {
        sum += (x[i] - x_prev[i]) * (x[i] - x_prev[i]);
    }
    return sqrt(sum);
}

// Метод Якоби + OpenMP
void CPU_Parallel_Jacobi_Method(double* A, double* f, double* x, int N, double eps) {
    double* x_prev = new double[N]();

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

        // Проверка условия остановки
        if(norm_omp(x, x_prev, N) <= eps) {
            break;
        }
    }

    delete[] x_prev; // освобождаем память
    cout << "Iteration count: " << iterations;
}
// -----------------------------------------------------------------------------------

// ---------------------------------- GPU-No-Tiling ----------------------------------
__global__ void jacobiKernel(double* x_next, const double* A, const double* x_now, const double* b_h, int N) {
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
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

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
    int iteration = 0;

    while(true) {
        // Копируем текущее приближение в x_prev
        cudaMemcpy(x_prev, d_x_now, N * sizeof(double), cudaMemcpyDeviceToHost);

        // Запускаем ядро Якоби
        jacobiKernel<<<blocks_per_grid, threads_per_block>>>(d_x_next, d_A, d_x_now, d_f, N);

        // Копируем результат Device->Host для проверки условия остановки
        cudaMemcpy(x, d_x_next, N * sizeof(double), cudaMemcpyDeviceToHost);
        
        iteration++;

        // Проверка условия остановки
        if(norm(x, x_prev, N) <= eps) {
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
}
// -----------------------------------------------------------------------------------

// ------------------------------------ GPU-Tiling ------------------------------------
//
// ------------------------------------------------------------------------------------

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

// --------------------------------- CONDITION NUMBER ---------------------------------
// Вычисление нормы по максимуму строк
double rowMaxNorm(double* A, int N) {
    double max_norm = 0;
    for(int i = 0; i < N; ++i) {
        double row_sum = 0;
        for(int j = 0; j < N; ++j) {
            row_sum += fabs(A[i * N + j]);
        }
        if(row_sum > max_norm) {
            max_norm = row_sum;
        }
    }
    return max_norm;
}

// Вычисление обратной матрицы
void invertMatrix(double* A, int N, double* A_inv) {
    // Создаем расширенную матрицу
    double* augmented = new double[2 * N * N]; // расширенная матрица с единичной матрицей справа

    // Инициализируем расширенную матрицу
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            augmented[2 * i * N + j] = A[i * N + j]; // копируем A
        }
        augmented[2 * i * N + N + i] = 1; // добавляем единичную матрицу
    }

    // Прямой ход Гаусса
    for(int i = 0; i < N; ++i) {
        // Если на диагонали 0, пробуем переставить строки
        if(augmented[2 * i * N + i] == 0) {
            int swap_row = -1;
            for(int k = i + 1; k < N; ++k) {
                if(augmented[2 * k * N + i] != 0) {
                    swap_row = k;
                    break;
                }
            }
            if(swap_row == -1) {
                delete[] augmented;
                throw runtime_error("Матрица вырожденная и не может быть инвертирована.");
            }
            // Меняем строки
            for(int j = 0; j < 2 * N; ++j) {
                swap(augmented[2 * i * N + j], augmented[swap_row * 2 * N + j]);
            }
        }

        // Нормализуем строку
        double pivot = augmented[i * 2 * N + i];
        for(int j = 0; j < 2 * N; ++j) {
            augmented[2 * i * N + j] /= pivot;
        }

        // Зануляем элементы в других строках
        for(int k = 0; k < N; ++k) {
            if(k == i) {
                continue;
            }
            double factor = augmented[2 * k * N + i];
            for(int j = 0; j < 2 * N; ++j) {
                augmented[2 * k * N + j] -= factor * augmented[2 * i * N + j];
            }
        }
    }

    // Копируем обратную матрицу из правой половины
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            A_inv[i * N + j] = augmented[i * 2 * N + N + j];
        }
    }

    delete[] augmented; // освобождаем память
}

// Вычисление числа обусловленности
double conditionNumber(double* A, int N) {
    double* A_inv = new double[N * N]; // массив для обратной матрицы

    // Вычисляем обратную матрицу
    invertMatrix(A, N, A_inv);

    // Вычисляем нормы
    double norm_A = rowMaxNorm(A, N);
    double norm_A_inv = rowMaxNorm(A_inv, N);

    delete[] A_inv; // освобождаем память
    
    // Число обусловленности
    return norm_A * norm_A_inv;
}
// ------------------------------------------------------------------------------------

int main() {
    srand(time(NULL)); // инициализация генератора случайных чисел
 
    int N = 128; // размер матрицы и вектора
    double eps = 1e-5; // погрешность
    int precision = 10; // количество знаков после запятой в выводе

    double* A = new double[N * N]; // матрица A
    double* f = new double[N]; // вектор f
    double* x_cpu_no_tiling = new double[N](); // начальное приближение CPU_NO_TILING (нулевой вектор)
    double* x_cpu_omp = new double[N](); // начальное приближение CPU_OMP (нулевой вектор)
    double* x_gpu_no_tiling = new double[N](); // начальное приближение GPU_NO_TILING (нулевой вектор)
    double* x_gpu_tiling = new double[N](); // начальное приближение GPU_TILING (нулевой вектор)

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

    cout << "Condition number: " << conditionNumber(A, N) << endl << endl;

    // -------------------------------- CPU-NO-TILING --------------------------------
    cudaEventRecord(start, 0); // старт таймера
    CPU_NO_TILING_Jacobi_Method(A, f, x_cpu_no_tiling, N, eps); // вызов метода Якоби

    float timer_value_СPU_NO_TILING;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer_value_СPU_NO_TILING, start, stop);
    printf("\n TIME: СPU-No-Tiling calculation - %f msec\n", timer_value_СPU_NO_TILING);

    // Вывод результата
    // cout << "Решение x_cpu_no_tiling:" << endl;
    // printVector(x_cpu_no_tiling, N, precision);
    // -------------------------------------------------------------------------------

    // -------------------------------- CPU-OMP --------------------------------
    cudaEventRecord(start, 0); // старт таймера
    CPU_Parallel_Jacobi_Method(A, f, x_cpu_omp, N, eps); // вызов метода Якоби

    float timer_value_СPU_OMP;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer_value_СPU_OMP, start, stop);
    printf("\n TIME: СPU-OMP calculation - %f msec\n", timer_value_СPU_OMP);

    // Вывод результата
    // cout << "Решение x_cpu_omp:" << endl;
    // printVector(x_cpu_omp, N, precision);
    // -------------------------------------------------------------------------

    // -------------------------------- GPU-NO-TILING --------------------------------
    cudaEventRecord(start, 0); // старт таймера
    GPU_NO_TILING_Jacobi_Method(A, f, x_gpu_no_tiling, N, eps); // вызов метода Якоби

    float timer_value_GPU_NO_TILING;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer_value_GPU_NO_TILING, start, stop);
    printf("\n TIME: GPU-No-Tiling calculation - %f msec\n", timer_value_GPU_NO_TILING);

    // Вывод результата
    // cout << "Решение x_gpu_no_tiling:" << endl;
    // printVector(x_gpu_no_tiling, N, precision);
    // -------------------------------------------------------------------------------

    // -------------------------------- GPU-TILING --------------------------------
    // cudaEventRecord(start, 0); // старт таймера
    // // GPU_TILING_Jacobi_Method(A, f, x_gpu_tiling, N, eps); // вызов метода Якоби

    // float timer_value_GPU_TILING;
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&timer_value_GPU_TILING, start, stop);
    // printf("\n TIME: GPU-Tiling calculation - %f msec\n", timer_value_GPU_TILING);

    // Вывод результата
    // cout << "Решение x_gpu_tiling:" << endl;
    // printVector(x_gpu_tiling, N, precision);
    // ----------------------------------------------------------------------------

    // Вывод ускорения
    printf("\nAcceleration:");
    // printf("\n Between NoTilingCPU and TilingGPU %fx", timer_value_СPU_NO_TILING / timer_value_GPU_TILING);
    printf("\n Between NoTilingCPU and NoTilingGPU %fx", timer_value_СPU_NO_TILING / timer_value_GPU_NO_TILING);
    // printf("\n Between OMPCPU and TilingGPU %fx", timer_value_СPU_OMP / timer_value_GPU_TILING);
    printf("\n Between OMPCPU and NoTilingGPU %fx", timer_value_СPU_OMP / timer_value_GPU_NO_TILING);

    // освобождение памяти
    delete[] A;
    delete[] f;
    delete[] x_cpu_no_tiling;
    delete[] x_cpu_omp;
    delete[] x_gpu_no_tiling;
    delete[] x_gpu_tiling;

    return 0;
}

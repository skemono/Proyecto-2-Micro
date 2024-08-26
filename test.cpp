#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <omp.h>
#include <cstdlib>
#include <ctime>

namespace fs = std::filesystem;

int main() {
    clock_t start_time = clock();
    std::string test_dir = "./testData";  // Ruta a tu conjunto de datos de prueba
    std::string output_file = "output.csv";
    int num_threads = 2;  // Número de hilos para el procesamiento en paralelo

    // Obtener una lista de todos los archivos en el directorio de prueba
    std::vector<std::string> test_files;
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (fs::is_regular_file(entry.path())) {
            test_files.push_back(entry.path().string());
        }
    }

    // Determinar el tamaño de cada lote
    int batch_size = test_files.size() / num_threads;
    if (batch_size == 0) batch_size = 1; // Asegurarse de que haya al menos un archivo por lote

    std::vector<std::string> batch_dirs;

    // Crear directorios para los lotes y distribuir los archivos
    for (int i = 0; i < num_threads; ++i) {
        std::string batch_dir = test_dir + "/batch" + std::to_string(i + 1);
        fs::create_directory(batch_dir);
        batch_dirs.push_back(batch_dir);

        for (int j = 0; j < batch_size && !test_files.empty(); ++j) {
            std::string file_path = test_files.back();
            test_files.pop_back();
            fs::rename(file_path, batch_dir + "/" + fs::path(file_path).filename().string());
        }
    }

    // Si quedan archivos, agregarlos al último lote
    while (!test_files.empty()) {
        std::string file_path = test_files.back();
        test_files.pop_back();
        fs::rename(file_path, batch_dirs.back() + "/" + fs::path(file_path).filename().string());
    }

    // Paralelizar la ejecución de test_model_paralel.py
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < batch_dirs.size(); ++i) {
        std::string command = "python test_model_paralel.py " + batch_dirs[i] + " " + output_file;
        int result = system(command.c_str());
        if (result != 0) {
            std::cerr << "Error al ejecutar test_model_paralel.py en el lote: " << batch_dirs[i] << std::endl;
        }
    }
    clock_t end_time = clock();

    // Opcionalmente mover los archivos de nuevo al directorio de prueba
    for (const auto& batch_dir : batch_dirs) {
        for (const auto& entry : fs::directory_iterator(batch_dir)) {
            if (fs::is_regular_file(entry.path())) {
                std::string file_path = entry.path().string();
                fs::rename(file_path, test_dir + "/" + fs::path(file_path).filename().string());
            }
        }
        fs::remove_all(batch_dir);
    }
    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Time taken: " << elapsed_time << " seconds" << std::endl;

    return 0;
}

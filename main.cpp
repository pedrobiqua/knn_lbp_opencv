#include "knn.hpp"
#include "lbp.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>

using namespace knn_ml;

namespace fs = std::filesystem;

void save_to_csv(const std::vector<knn_ml::data_point>& points,
    const std::string& filename = "points.csv")
{
    std::ofstream file(filename);

    // Escreve cabeçalho do CSV (assumimos no máximo duas features para plotar no
    // 2D)
    file << "x,y,label\n";

    for (const auto& point : points) {
        if (point.features.n_elem >= 2) {
            file << point.features[0] << "," << point.features[1] << ","
                 << point.label << "\n";
        } else {
            std::cerr << "Ponto com menos de 2 dimensões não será salvo: "
                      << point.features << std::endl;
        }
    }

    file.close();
    std::cout << "Pontos salvos no arquivo: " << filename << std::endl;
}

void plot(const std::vector<knn_ml::data_point>& data, std::string name)
{
    // Save image results
    save_to_csv(data);
    std::string param = "python3 ../plot_points.py points.csv " + name;
    int result = system(param.c_str());
    if (result == 0) {
        std::cout << "Script Python executado com sucesso!" << std::endl;
    } else {
        std::cerr << "Erro ao executar o script Python." << std::endl;
    }
}
// "/home/pedro/projects/knn_lbp_opencv/build/output_lbp.jpg"
cv::Mat load_image(std::string path_image)
{
    cv::Mat image = cv::imread(path_image, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Erro ao carregar a imagem. Certifique-se de que o caminho está correto!" << std::endl;
    }
    return image;
}

cv::Mat calculate_lbp(cv::Mat& image)
{
    ImageProcessing::LocalBinaryPattern lbp;

    // Calcular o LBP com P=8, R=1.0, método="default"
    int P = 8;
    double R = 1.0;
    std::string method = "default";
    cv::Mat lbp_image = lbp.calculate(image, P, R, method);

    // Normalizar a saída para visualização
    cv::normalize(lbp_image, lbp_image, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Salvar e exibir a imagem LBP
    // cv::imwrite("output_lbp.jpg", lbp_image);
    // std::cout << "LBP calculado com sucesso. Resultado salvo em 'output_lbp.jpg'." << std::endl;

    // Exibir a imagem original e a imagem LBP
    // cv::imshow("Imagem Original", image);
    // cv::imshow("LBP", lbp_image);
    // cv::waitKey(0);

    return lbp_image;
}

cv::Mat calculate_hist_image(cv::Mat& image, int histSize, float range[], const float* histRange)
{
    cv::Mat hist;

    // Calcular o histograma
    // cv::Mat image_test = cv::imread("/home/pedro/projects/knn_lbp_opencv/build/output_lbp.jpg", cv::IMREAD_GRAYSCALE);
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // Normalizar o histograma para caber na altura da imagem
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX); // Normalizando os dados

    return hist;
}

void save_hist_csv(cv::Mat& hist, int hist_size, int class_target, std::ofstream& csvFile)
{
    int count = 0;
    // Escrever os valores do histograma no formato especificado
    for (int i = 0; i < hist_size; i++) {
        csvFile << hist.at<float>(i); // Valor do bin
        count++;
        if (i < hist_size - 1) {
            csvFile << ";"; // Adicionar separador entre os valores
        }
    }

    // Adicionar a classe da imagem no final
    csvFile << ";" << class_target << "\n";
}

std::vector<knn_ml::data_point> load_csv(const std::string& filename)
{
    std::vector<knn_ml::data_point> data;

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Erro ao abrir o arquivo CSV!");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row_values;
        std::string value;

        // Ler cada valor na linha (separado por vírgula)
        while (std::getline(ss, value, ',')) {
            row_values.push_back(std::stod(value)); // Converter string para double
        }

        // Separar a última coluna como o rótulo (classe)
        int label = static_cast<int>(row_values.back()); // Último valor como classe
        row_values.pop_back(); // Remover o rótulo das features

        // Adicionar ao vetor de pontos de dados
        arma::rowvec features(row_values); // Criar um rowvec para as features
        data.push_back(knn_ml::data_point(features, label));
    }

    file.close();
    return data;
}

void split_data(const std::vector<data_point>& data,
    std::vector<data_point>& train_data,
    std::vector<data_point>& test_data,
    double test_ratio = 0.1)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<data_point> shuffled_data = data;

    // Embaralhar os dados
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), gen);

    // Dividir os dados
    size_t train_size = static_cast<size_t>(data.size() * (1 - test_ratio));
    train_data = std::vector<data_point>(shuffled_data.begin(), shuffled_data.begin() + train_size);
    test_data = std::vector<data_point>(shuffled_data.begin() + train_size, shuffled_data.end());
}

// void split_data(const std::vector<data_point>& data,
//     std::vector<data_point>& train_data,
//     std::vector<data_point>& test_data,
//     size_t test_size = 10) // Tamanho fixo para o conjunto de teste
// {
//     // Verificar se o conjunto de dados é suficiente
//     if (data.size() < test_size) {
//         throw std::invalid_argument("O conjunto de dados não contém elementos suficientes para o tamanho do teste.");
//     }

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::vector<data_point> shuffled_data = data;

//     // Embaralhar os dados
//     std::shuffle(shuffled_data.begin(), shuffled_data.end(), gen);

//     // Separar os dados de treino e teste
//     test_data = std::vector<data_point>(shuffled_data.begin(), shuffled_data.begin() + test_size);
//     train_data = std::vector<data_point>(shuffled_data.begin() + test_size, shuffled_data.end());
// }

void testar_app(std::string& filename)
{
    try {
        // Carregar dados do CSV
        // std::string filename = "/home/pedro/projects/knn_lbp_opencv/build/output_histograms.csv";
        std::vector<data_point> data = load_csv(filename);

        // Separar em treino e teste
        std::vector<data_point> train_data, test_data;
        split_data(data, train_data, test_data, 0.5);

        std::cout << "Passei do split" << std::endl;
        // Criar o objeto KNN com a métrica desejada (por exemplo, "euclidean")
        knn_ml::knn knn("euclidean");

        // Treinar o modelo com os dados de treino
        knn.fit(train_data);

        std::cout << "Passei do fit" << std::endl;

        // Fazer previsões nos dados de teste
        int k = 3;
        std::vector<int> predictions = knn.predict(test_data, k);

        std::cout << "Passei do predict" << std::endl;

        // Mostrar os resultados
        std::cout << "Previsões:" << std::endl;
        for (size_t i = 0; i < predictions.size(); ++i) {
            std::cout << "Ponto " << i + 1 << ": Classe prevista = "
                      << predictions[i] << ", Classe real = "
                      << test_data[i].label << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
    }
}

int main()
{

    std::string filename = "/home/pedro/projects/knn_lbp_opencv/build/output_histograms.csv";

    if (fs::exists(filename)) {
        auto start = std::chrono::high_resolution_clock::now();
        testar_app(filename);
        auto end = std::chrono::high_resolution_clock::now();

        // Calcular a duração em milissegundos
        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "Tempo: " << duration.count() << " ms" << std::endl;
        return 0;
    }

    try {

        // Parâmetros do histograma
        int histSize = 256; // Número de bins
        float range[] = { 0, 256 }; // Intervalo [0, 256)
        const float* histRange = { range };

        // Caminho da pasta principal
        std::string root_path = "/home/pedro/Downloads/PKLot/PKLotSegmented/PUC";

        // Nome do arquivo CSV de saída
        std::string output_csv = "output_histograms.csv";

        // Abrir arquivo para salvar todos os histogramas
        std::ofstream csvFile(output_csv, std::ios::app);
        if (!csvFile.is_open()) {
            std::cerr << "Erro ao abrir o arquivo para escrita: " << output_csv << std::endl;
            return -1;
        }

        // Iterar sobre todas as imagens na pasta (e subpastas)
        for (const auto& entry : fs::recursive_directory_iterator(root_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
                std::string image_path = entry.path().string();
                std::cout << "Processando imagem: " << image_path << std::endl;

                // Carregar a imagem
                cv::Mat image = load_image(image_path);
                if (image.empty()) {
                    std::cerr << "Erro ao carregar a imagem: " << image_path << std::endl;
                    continue;
                }

                // Calcular o LBP
                cv::Mat lbp_image = calculate_lbp(image);

                // Calcular o histograma
                cv::Mat hist = calculate_hist_image(lbp_image, histSize, range, histRange);

                // Inferir a classe da imagem a partir do caminho (exemplo: "Occupied" -> 1, "Empty" -> 0)
                int class_target = (image_path.find("Occupied") != std::string::npos) ? 1 : 0;

                // Salvar o histograma no CSV
                save_hist_csv(hist, 256, class_target, csvFile);
            }
        }

        // Fechar o arquivo CSV
        csvFile.close();
        std::cout << "Processamento concluído! Resultados salvos em: " << output_csv << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
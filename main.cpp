#include "knn.hpp"
#include "lbp.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

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

int main()
{
    // Testando o knn já implementado
    // using namespace knn_ml;

    // // Dados de treino e teste
    // std::vector<data_point> train_data = {
    //     data_point({ 1.0, 2.0 }, 0), data_point({ 2.0, 3.0 }, 0),
    //     data_point({ 3.0, 3.0 }, 0), data_point({ 6.0, 6.0 }, 1),
    //     data_point({ 7.0, 7.0 }, 1), data_point({ 8.0, 8.0 }, 1)
    // };

    // std::vector<data_point> test_data = {
    //     data_point({ 5.0, 5.0 }, -1), // Classe desconhecida
    //     data_point({ 2.5, 2.5 }, -1) // Classe desconhecida
    // };

    // // Criando o objeto KNN com a métrica de distância desejada (Ex: "manhattan",
    // // "minkowski", "euclidean")
    // knn knn("minkowski");

    // // Treinando o modelo com os dados de treino
    // knn.fit(train_data);

    // // Realizando previsões com k = 3
    // std::vector<int> predictions = knn.predict(test_data, 3);

    // // Exibindo as previsões
    // for (int i = 0; i < predictions.size(); ++i) {
    //     std::cout << "Classe prevista para o ponto de teste " << i + 1 << ": "
    //               << predictions[i] << std::endl;
    //     test_data[i].label = predictions[i];
    // }

    // plot(train_data, "train.png");
    // plot(test_data, "test.png");

    // return 0;

    // Testando o lbp
    try {
        // Carregar uma imagem em tons de cinza
        cv::Mat image = cv::imread("/home/pedro/Downloads/PKLot/PKLotSegmented/PUC/Sunny/2012-09-11/Occupied/2012-09-11_15_53_00#005.jpg", cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Erro ao carregar a imagem. Certifique-se de que o caminho está correto!" << std::endl;
            return -1;
        }

        // Instanciar a classe LBP
        ImageProcessing::LocalBinaryPattern lbp;

        // Calcular o LBP com P=8, R=1.0, método="default"
        int P = 8;
        double R = 1.0;
        std::string method = "default";
        cv::Mat lbp_image = lbp.calculate(image, P, R, method);

        // Normalizar a saída para visualização
        cv::normalize(lbp_image, lbp_image, 0, 255, cv::NORM_MINMAX, CV_8U);

        // Salvar e exibir a imagem LBP
        cv::imwrite("output_lbp.jpg", lbp_image);
        std::cout << "LBP calculado com sucesso. Resultado salvo em 'output_lbp.jpg'." << std::endl;

        // Exibir a imagem original e a imagem LBP
        cv::imshow("Imagem Original", image);
        cv::imshow("LBP", lbp_image);
        cv::waitKey(0);
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return -1;
    }

    // Carrega a imagem
    cv::Mat image = cv::imread("/home/pedro/projects/knn_lbp_opencv/build/output_lbp.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Erro ao carregar a imagem. Certifique-se de que o caminho está correto!" << std::endl;
        return -1;
    }

    // Parâmetros do histograma
    int histSize = 256; // Número de bins
    float range[] = { 0, 256 }; // Intervalo [0, 256)
    const float* histRange = { range };

    cv::Mat hist;

    // Calcular o histograma
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // Normalizar o histograma para caber na altura da imagem
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

    // Classe associada à imagem (exemplo: classe 1)
    int imageClass = 1;

    // Nome do arquivo CSV
    std::string csvFilename = "histogram_class_output.csv";

    // Abrir o arquivo para escrita
    std::ofstream csvFile(csvFilename);
    if (!csvFile.is_open()) {
        std::cerr << "Erro ao abrir o arquivo para salvar o histograma!" << std::endl;
        return -1;
    }

    int count = 0;
    // Escrever os valores do histograma no formato especificado
    for (int i = 0; i < histSize; i++) {
        csvFile << hist.at<float>(i); // Valor do bin
        count++;
        if (i < histSize - 1) {
            csvFile << ";"; // Adicionar separador entre os valores
        }
    }

    std::cout << "Total de caracteristicas= " << count << std::endl;

    // Adicionar a classe da imagem no final
    csvFile << ";" << imageClass << "\n";

    // Fechar o arquivo
    csvFile.close();

    std::cout << "Histograma salvo em: " << csvFilename << std::endl;

    return 0;
}

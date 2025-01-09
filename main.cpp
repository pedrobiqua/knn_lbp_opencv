#include <iostream>

#include "knn_lbp.hpp"

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
    using namespace knn_ml;

    // Dados de treino e teste
    std::vector<data_point> train_data = {
        data_point({ 1.0, 2.0 }, 0), data_point({ 2.0, 3.0 }, 0),
        data_point({ 3.0, 3.0 }, 0), data_point({ 6.0, 6.0 }, 1),
        data_point({ 7.0, 7.0 }, 1), data_point({ 8.0, 8.0 }, 1)
    };

    std::vector<data_point> test_data = {
        data_point({ 5.0, 5.0 }, -1), // Classe desconhecida
        data_point({ 2.5, 2.5 }, -1) // Classe desconhecida
    };

    // Criando o objeto KNN com a métrica de distância desejada (Ex: "manhattan",
    // "minkowski", "euclidean")
    knn_lbp knn("minkowski");

    // Treinando o modelo com os dados de treino
    knn.fit(train_data);

    // Realizando previsões com k = 3
    std::vector<int> predictions = knn.predict(test_data, 3);

    // Exibindo as previsões
    for (int i = 0; i < predictions.size(); ++i) {
        std::cout << "Classe prevista para o ponto de teste " << i + 1 << ": "
                  << predictions[i] << std::endl;
        test_data[i].label = predictions[i];
    }

    plot(train_data, "train.png");
    plot(test_data, "test.png");

    return 0;
}

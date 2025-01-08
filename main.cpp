#include <iostream>

#include "knn_lbp.hpp"

int main() {
  using namespace knn_ml;

  // Dados de treino e teste
  std::vector<data_point> train_data = {
      data_point({1.0, 2.0}, 0), data_point({2.0, 3.0}, 0),
      data_point({3.0, 3.0}, 0), data_point({6.0, 6.0}, 1),
      data_point({7.0, 7.0}, 1), data_point({8.0, 8.0}, 1)};

  std::vector<data_point> test_data = {
      data_point({5.0, 5.0}, -1),  // Classe desconhecida
      data_point({2.5, 2.5}, -1)   // Classe desconhecida
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
  }

  return 0;
}

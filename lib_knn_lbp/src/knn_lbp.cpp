#include "knn_lbp.hpp"

namespace knn_ml {

// Construtor
knn_lbp::knn_lbp(const std::string& metric) : distance_metric(metric) {}

// Destrutor
knn_lbp::~knn_lbp() {}

// Função para calcular a distância Euclidiana
double knn_lbp::euclidean_distance(const arma::rowvec& a,
                                   const arma::rowvec& b) {
  return arma::norm(a - b, 2);  // Norm 2 (distância Euclidiana)
}

// Função para calcular a distância Manhattan
double knn_lbp::manhattan_distance(const arma::rowvec& a,
                                   const arma::rowvec& b) {
  return arma::norm(a - b, 1);  // Norm 1 (distância Manhattan)
}

// Função para calcular a distância Minkowski
double knn_lbp::minkowski_distance(const arma::rowvec& a, const arma::rowvec& b,
                                   double p) {
  return std::pow(arma::norm(a - b, p), 1.0 / p);  // Distância Minkowski
}

// Função auxiliar para escolher a distância de acordo com a métrica
double knn_lbp::calculate_distance(const arma::rowvec& a,
                                   const arma::rowvec& b) {
  if (distance_metric == "euclidean") {
    return euclidean_distance(a, b);
  } else if (distance_metric == "manhattan") {
    return manhattan_distance(a, b);
  } else if (distance_metric == "minkowski") {
    double p = 3;  // Exemplo de valor de p para Minkowski
    return minkowski_distance(a, b, p);
  } else {
    throw std::invalid_argument("Métrica de distância desconhecida");
  }
}

// Método 'fit' para treinar o modelo (armazenar dados de treino)
void knn_lbp::fit(const std::vector<data_point>& train) { train_data = train; }

// Método 'predict' para fazer previsões
std::vector<int> knn_lbp::predict(const std::vector<data_point>& test_data,
                                  int k) {
  std::vector<int> predictions;

  // Para cada ponto de teste, predizer a classe
  for (const auto& test_point : test_data) {
    int predicted_class = classify(test_point, k);
    predictions.push_back(predicted_class);
  }

  return predictions;
}

// Função auxiliar para realizar a classificação de um ponto de teste
int knn_lbp::classify(const data_point& test_point, int k) {
  std::vector<std::pair<double, int>> distances;

  // Calcular a distância de cada ponto de treino para o ponto de teste
  for (const auto& train_point : train_data) {
    double dist = calculate_distance(test_point.features, train_point.features);
    distances.push_back(std::make_pair(dist, train_point.label));
  }

  // Ordenar as distâncias
  std::sort(distances.begin(), distances.end());

  // Contar as classes dos k vizinhos mais próximos
  std::map<int, int> class_count;
  for (int i = 0; i < k; i++) {
    class_count[distances[i].second]++;
  }

  // Encontrar a classe com maior contagem
  int predicted_class = -1;
  int max_count = 0;
  for (const auto& entry : class_count) {
    if (entry.second > max_count) {
      max_count = entry.second;
      predicted_class = entry.first;
    }
  }

  return predicted_class;
}

}  // namespace knn_ml

// ===========================================================================
/**
 * Funções para apenas testar se funciona as bibliotecas que vou usar
 * Armadilo: Lib para operações matemáticas
 * OpenCV: Lib de visão computacional
 */
void say_hello() { std::cout << "Hello, from knn_lbp!\n"; }

// Função para testar o OpenCV
void test_opencv() {
  // Cria uma imagem preta de 400x400 pixels
  cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);

  // Desenha um texto na imagem
  cv::putText(image, "OpenCV Test", cv::Point(50, 200),
              cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

  // Exibe a imagem em uma janela
  cv::imshow("OpenCV Test Window", image);

  // Espera por qualquer tecla para fechar a janela
  cv::waitKey(0);
}

void test_armadilo() {
  arma::mat A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  arma::mat B = {{9.0, 8.0, 7.0}, {6.0, 5.0, 4.0}, {3.0, 2.0, 1.0}};

  // Multiplicar as matrizes A e B
  arma::mat C = A * B;

  // Imprimir a matriz resultante
  std::cout << "Matriz A:" << std::endl;
  A.print();

  std::cout << "Matriz B:" << std::endl;
  B.print();

  std::cout << "Resultado da multiplicação (A * B):" << std::endl;
  C.print();

  // Calcular e imprimir o determinante de A
  double det = arma::det(A);
  std::cout << "Determinante de A: " << det << std::endl;
}

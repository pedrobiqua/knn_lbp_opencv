#ifndef KNN_LBP_H
#define KNN_LBP_H

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

// Namespace para KNN
namespace knn_ml {

// Estrutura para armazenar um ponto de dados
struct data_point {
  arma::rowvec features;  // Características do ponto
  int label;              // Rótulo (classe) do ponto

  data_point(arma::rowvec f, int l) : features(f), label(l) {}
};

// Classe KNN
class knn_lbp {
 private:
  // Variáveis para armazenar o conjunto de treino
  std::vector<data_point> train_data;
  std::string distance_metric;  // Métrica de distância

 public:
  // Construtor
  knn_lbp(const std::string& metric = "euclidean");

  // Destrutor
  ~knn_lbp();

  // Método 'fit' para treinar o modelo
  void fit(const std::vector<data_point>& train);

  // Método 'predict' para fazer previsões com o modelo
  std::vector<int> predict(const std::vector<data_point>& test_data, int k);

 private:
  // Função para calcular a distância Euclidiana
  double euclidean_distance(const arma::rowvec& a, const arma::rowvec& b);

  // Função para calcular a distância Manhattan
  double manhattan_distance(const arma::rowvec& a, const arma::rowvec& b);

  // Função para calcular a distância Minkowski
  double minkowski_distance(const arma::rowvec& a, const arma::rowvec& b,
                            double p);

  // Função auxiliar para escolher a distância de acordo com a métrica
  double calculate_distance(const arma::rowvec& a, const arma::rowvec& b);

  // Função auxiliar para realizar a classificação de um ponto de teste
  int classify(const data_point& test_point, int k);
};

}  // namespace knn_ml

void test_armadilo();
void test_opencv();

#endif  // KNN_LBP_H
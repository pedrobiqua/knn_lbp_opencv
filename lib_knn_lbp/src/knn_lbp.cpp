#include "knn_lbp.hpp"

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

#include "knn_lbp.hpp"

void say_hello()
{
    std::cout << "Hello, from knn_lbp!\n";
}

// Função para testar o OpenCV
void test_opencv()
{
    // Cria uma imagem preta de 400x400 pixels
    cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);

    // Desenha um texto na imagem
    cv::putText(image, "OpenCV Test", cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

    // Exibe a imagem em uma janela
    cv::imshow("OpenCV Test Window", image);

    // Espera por qualquer tecla para fechar a janela
    cv::waitKey(0);
}

#ifndef LBP_HPP
#define LBP_HPP

#include <cmath>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace ImageProcessing {
class LocalBinaryPattern {
public:
    // Construtor
    LocalBinaryPattern() = default;

    // Método principal para calcular o LBP
    cv::Mat calculate(const cv::Mat& image, int P, double R, const std::string& method = "default");

private:
    // Função auxiliar para verificar se a imagem é 2D
    void check_nD(const cv::Mat& image, int ndim);
};
}

#endif // LBP_HPP

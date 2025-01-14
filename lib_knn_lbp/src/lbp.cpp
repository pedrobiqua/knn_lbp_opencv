#include "lbp.hpp"

namespace ImageProcessing {

void LocalBinaryPattern::check_nD(const cv::Mat& image, int ndim)
{
    if (image.dims != ndim) {
        throw std::invalid_argument("A imagem deve ser 2D.");
    }
}

cv::Mat LocalBinaryPattern::calculate(const cv::Mat& image, int P, double R, const std::string& method)
{
    // Verificar se a imagem é 2D
    check_nD(image, 2);

    // Mapear os métodos de LBP para caracteres
    std::map<std::string, char> methods = {
        { "default", 'D' },
        { "ror", 'R' },
        { "uniform", 'U' },
        { "nri_uniform", 'N' },
        { "var", 'V' },
    };

    // Validar o método escolhido
    if (methods.find(method) == methods.end()) {
        throw std::invalid_argument("Método inválido: " + method);
    }

    char method_char = methods[method];

    // Aviso caso a imagem seja de ponto flutuante
    if (image.type() == CV_64F || image.type() == CV_32F) {
        std::cerr << "Aviso: Aplicar `local_binary_pattern` em imagens de ponto flutuante "
                  << "pode gerar resultados inesperados devido a pequenas diferenças numéricas "
                  << "entre pixels adjacentes. Recomenda-se usar imagens de tipo inteiro."
                  << std::endl;
    }

    // Converter a imagem para float64 (se necessário)
    cv::Mat image_contiguous;
    image.convertTo(image_contiguous, CV_64F);

    // Calcular o LBP
    cv::Mat output = cv::Mat::zeros(image.rows, image.cols, CV_64F);

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            double lbp_value = 0.0;

            // Cálculo circular para P vizinhos
            for (int p = 0; p < P; ++p) {
                double theta = 2.0 * M_PI * p / P;
                int x = static_cast<int>(i + R * std::cos(theta) + 0.5);
                int y = static_cast<int>(j - R * std::sin(theta) + 0.5);

                if (x >= 0 && x < image.rows && y >= 0 && y < image.cols) {
                    double center = image_contiguous.at<double>(i, j);
                    double neighbor = image_contiguous.at<double>(x, y);
                    if (neighbor >= center) {
                        lbp_value += std::pow(2, p);
                    }
                }
            }
            output.at<double>(i, j) = lbp_value;
        }
    }

    return output;
}
}

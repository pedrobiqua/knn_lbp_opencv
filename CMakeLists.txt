cmake_minimum_required(VERSION 3.28.0)
project(knn_lbp VERSION 0.1.0 LANGUAGES C CXX)

# Adicionar a biblioteca
add_library(knn_lbp lib_knn_lbp/src/knn_lbp.cpp)

# Definir os diretórios de inclusão da biblioteca
target_include_directories(knn_lbp PUBLIC lib_knn_lbp/include)

# Adicionar o executável que será usado para testar a biblioteca
add_executable(main_exec main.cpp)

# Linkar a biblioteca com o executável
target_link_libraries(main_exec PRIVATE knn_lbp)

# Configurar testes (opcional)
include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

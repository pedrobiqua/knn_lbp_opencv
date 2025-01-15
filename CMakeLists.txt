cmake_minimum_required(VERSION 3.28.0)
project(knn_lbp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Garantir um padrão de compilação (Debug se nenhum for especificado)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Encontre bibliotecas e pacotes necessários
find_package(PkgConfig REQUIRED)        # Para o OpenCV
pkg_check_modules(OpenCV REQUIRED opencv4)

find_package(Armadillo REQUIRED)        # Armadillo
find_package(LAPACK REQUIRED)           # LAPACK
find_package(BLAS REQUIRED)             # BLAS

# Adicionar a biblioteca knn_lbp
add_library(knn_lbp STATIC
    lib_knn_lbp/src/knn.cpp
    lib_knn_lbp/src/lbp.cpp
)

# Definir a macro ARMA_DONT_USE_WRAPPER
target_compile_definitions(knn_lbp PUBLIC ARMA_DONT_USE_WRAPPER)

# Incluir diretórios de cabeçalho
target_include_directories(knn_lbp PUBLIC
    lib_knn_lbp/include
    ${OpenCV_INCLUDE_DIRS}
    ${Armadillo_INCLUDE_DIRS}
)

# Linkar bibliotecas com a knn_lbp
target_link_libraries(knn_lbp PUBLIC
    ${OpenCV_LIBRARIES}
    ${Armadillo_LIBRARIES}
    LAPACK::LAPACK
    BLAS::BLAS
    pthread
)

# Adicionar o executável principal
add_executable(main_exec main.cpp)

# Linkar bibliotecas ao executável principal
target_link_libraries(main_exec PRIVATE
    knn_lbp
    pthread
    ${OpenCV_LIBRARIES}
    ${Armadillo_LIBRARIES}
    LAPACK::LAPACK
    BLAS::BLAS
)

# Adicionar flags específicas para o OpenCV
target_compile_options(main_exec PRIVATE ${OpenCV_CFLAGS_OTHER})

# Configurar testes (opcional)
include(CTest)
enable_testing()

# Configuração do empacotamento (opcional)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

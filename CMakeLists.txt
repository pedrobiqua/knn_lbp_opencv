cmake_minimum_required(VERSION 3.28.0)
project(knn_lbp VERSION 0.1.0 LANGUAGES C CXX)

# Encontre o OpenCV no sistema usando pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)

# Encontre o Armadillo
find_package(Armadillo REQUIRED)

# Encontre o LAPACK e BLAS
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

# Adicionar a biblioteca knn_lbp
add_library(knn_lbp STATIC lib_knn_lbp/src/knn.cpp lib_knn_lbp/src/lbp.cpp)

# Definir a macro ARMA_DONT_USE_WRAPPER antes de incluir o cabeçalho do Armadillo
target_compile_definitions(knn_lbp PUBLIC ARMA_DONT_USE_WRAPPER)

# Incluir os diretórios de cabeçalho
target_include_directories(knn_lbp PUBLIC
    lib_knn_lbp/include
    ${OpenCV_INCLUDE_DIRS}  # Diretórios do OpenCV
    ${Armadillo_INCLUDE_DIRS}  # Diretórios do Armadillo
)

# Linkar o OpenCV, Armadillo, LAPACK, BLAS, e pthread na biblioteca knn_lbp
target_link_libraries(knn_lbp PUBLIC
    ${OpenCV_LIBRARIES}     # OpenCV
    ${Armadillo_LIBRARIES}  # Armadillo
    LAPACK::LAPACK          # LAPACK
    BLAS::BLAS              # BLAS
    pthread                 # pthread
)

# Adicionar o executável principal
add_executable(main_exec main.cpp)

# Linkar a biblioteca knn_lbp com o executável principal
target_link_libraries(main_exec PRIVATE
    knn_lbp
    pthread                 # pthread
    ${OpenCV_LIBRARIES}     # OpenCV
    ${Armadillo_LIBRARIES}  # Armadillo
    LAPACK::LAPACK          # LAPACK
    BLAS::BLAS              # BLAS
)

# Adicionar flags de compilação (se necessário)
target_compile_options(main_exec PRIVATE ${OpenCV_CFLAGS_OTHER})

# Configurar testes (opcional)
include(CTest)
enable_testing()

# Configurações do CPack para empacotamento (opcional)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

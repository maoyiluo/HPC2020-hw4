CFLAGS = -Xcompiler -fopenmp -O3
CC = nvcc

all: inner_product, matrix_vector, jacobi

inner_product: p1_inner_product.cu
	${CC} -o p1_inner_product ${CFLAGS} p1_inner_product.cu

matrix_vector: p1_matrix_vector.cu
	${CC} -o p1_matrix_vector ${CFLAGS} p1_matrix_vector.cu

jacobi: p2_jacobi.cu
	${CC} -o p2 ${CFLAGS} p2_jacobi.cu 
#include <cuda.h>
#include <cstdkib.h>
using namespace std;

#define N 1024
#define TAILLE_BLOC_X 16
#define TAILLE_BLOC_Y 16

/*====================*/
/* KERNEL DECLARATION */
/*====================*/

__global__ void matmulKernel (float *d_A,
	   		      float *d_B,
			      float *d_C,
			      int n)
{
  unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int i = blockDim.y * blockIdy.y + threadIdy.y;

  if (i<n && j<n)
    {
      float temp=0;
      for (int k=0; k<n; k++)
	temp = temp + d_A[i*n+k] * d_B[k*n+j];

      d_C[i*n+j] = temp;
    }
}

/*=================*/
/* CPU DECLARATION */
/*=================*/

void matmulCPU (float *c_A,
		float *c_B,
		float *c_C,
		int n)
{
  int i,j,k;
  int s;

  for(i=0; i<n ; i++){
    for(j=0; j<n ; j++){
      s=0;
      for(k=0; k<n ; k++)
	s+=c_A[i*n+k]*c_B[k*n+j];
      c_C[i*n+j]=s;
    }
  }
}

/*==================*/
/* MAIN DECLARATION */
/*==================*/

int main()
{
  float *A, *B, *C, *C_ref_CPU, *d_A, *d_B, *d_C;
  int taille_totale = N*N*sizeof(float),i,j;

  /* Allocation CPU */
  A=(float*)malloc(taille_totale);
  B=(float*)malloc(taille_totale);
  C=(float*)malloc(taille_totale);

  fillMatriceFloat(A,N);
  fillMatriceFloat(B,N);

  /* Allocation GPU */
  cudaMalloc((void **) &d_A, taille_totale);
  cudaMalloc((void **) &d_B, taille_totale);
  cudaMalloc((void **) &d_C, taille_totale);

  /* Transferts CPU -> GPU */
  cudaMemcpy(d_A, A, taille_totale, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, taille_totale, cudaMemcpyHostToDevice);

  /* Lancement de kernel */
  dim3 threadsParBloc(TAILLE_BLOC_X, TAILLE_BLOC_Y);
  dim3 tailleGrille(ceil(N/(float) TAILLE_BLOC_X), ceil(N/(float) TAILLE_BLOC_Y));
  matmulKernel<<<tailleGrille, threadsParBloc>>>(d_A, d_B, d_C, N);

  /* Transferts GPU -> CPU */
  cudaMemcpy(d_C, C, taille_totale, cudaMemcpyDeviceToHost);

  /* Verification */
  matmulCPU(A, B, C_ref_CPU, N);
  for(i=0; i<N; i++)
    for(j=0; j<N; j++)
      if (fabsf(C[i*N+j]-C_ref_CPU[i*N+j]) > 0.001)
	printf("%4d %4d h %le d %le\n",i,j,C_ref_CPU[i*DIM+j],C[i*DIM+j]);

  /* Liberation memoire GPU et CPU */
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(A); free(B); free(C); free(C_ref_CPU);

  return 0;
}

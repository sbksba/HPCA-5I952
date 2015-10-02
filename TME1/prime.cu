#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 

#define N 10000000
#define TAILLE_BLOC_X 32
#define TAILLE_BLOC_Y 1
#define T 1
#define F 0

/*====================*/
/* KERNEL DECLARATION */
/*====================*/

__global__ void primeKernel (int *tabGPU, int n)
{
		unsigned int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
		unsigned int j = 2;
		while ((j*j)<=i)
		{
				if (i%j==0){
			   	/*printf("%d -- not prime\n",i);*/
			   	tabGPU[i-1]=F;
			   	return;
				}
				j++;
		}
		tabGPU[i-1]=T;
		/*printf("%d -- prime\n",i);*/
}

/*=================*/
/* CPU DECLARATION */
/*=================*/

void primeCPU (int *tabCPU, int n)
{
	int i,j;
	int est_premier;
	for (i=2; i<n; i++){
		est_premier=T;
		j=2;
		while ((j*j) <= n){
			if ((i%j) == 0)
			{
			   est_premier=F;
			   tabCPU[i-2]=F;
			   break;
			}
			j++;
		}
		tabCPU[i-2]=T;
	}
}

/*==================*/
/* MAIN DECLARATION */
/*==================*/
double my_gettimeofday()
{
	struct timeval tmp_time;
    	gettimeofday(&tmp_time, NULL);
      	return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}
      
int main()
{
	int *tabCPU, *tabGPU;
	double deb, fin;
	
	/* Allocation CPU */
	tabCPU = (int*)malloc(N*sizeof(int));
	
	/* Allocation GPU */
	cudaMalloc((void**)&tabGPU, N*sizeof(int));

	/* Transferts CPU -> GPU */
	cudaMemcpy(tabCPU, tabGPU, N, cudaMemcpyHostToDevice);
	
	/* Lancement de kernel */
	dim3 threadsParBloc(TAILLE_BLOC_X);
	dim3 nbBlocs(ceil(N/TAILLE_BLOC_X));
	
	deb = my_gettimeofday();
	primeKernel<<<nbBlocs, threadsParBloc>>>(tabGPU, N);

	/* Attente de la fin du calcul GPU */
	/*cudaDeviceSynchronize();*/

	/* Transferts GPU -> CPU */
	cudaMemcpy(tabCPU, tabGPU, N, cudaMemcpyDeviceToHost);
	fin = my_gettimeofday();

	printf("TIME GPU [%d] -- %g(s)\n",N,(fin-deb));

	/* Verification */
	deb = my_gettimeofday();
	primeCPU(tabCPU, N);
	fin = my_gettimeofday();
	printf("TIME CPU [%d] -- %g(s)\n",N,(fin-deb));

	return 0;
}

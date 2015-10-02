/**
 * Programmation GPU 
 * Université Pierre et Marie Curie
 * Calcul de convolution sur une image.
 */

/**
 * V0
 *
 */

#include <cuda.h>
#include <stdio.h>

extern "C" double my_gettimeofday();

/** 
 * Controle des erreurs CUDA et debugging. 
 */

#ifdef CUDA_DEBUG
#define CUDA_SYNC_ERROR() {						\
    cudaError_t sync_error;						\
    cudaDeviceSynchronize();						\
    sync_error = cudaGetLastError();					\
    if(sync_error != cudaSuccess) {					\
      fprintf(stderr, "[CUDA SYNC ERROR at %s:%d -> %s]\n",		\
	      __FILE__ , __LINE__, cudaGetErrorString(sync_error));	\
      exit(EXIT_FAILURE);						\
    }									\
  }
#else /* #ifdef CUDA_DEBUG */
#define CUDA_SYNC_ERROR()
#endif /* #ifdef CUDA_DEBUG */

#define CUDA_ERROR(cuda_call) {					\
    cudaError_t error = cuda_call;				\
    if(error != cudaSuccess){					\
      fprintf(stderr, "[CUDA ERROR at %s:%d -> %s]\n",		\
	      __FILE__ , __LINE__, cudaGetErrorString(error));	\
      exit(EXIT_FAILURE);					\
    }								\
    CUDA_SYNC_ERROR();						\
  }



/**
 * Retourne le quotient entier superieur ou egal a "a/b". 
 * D'apres : CUDA SDK 4.1 
 */

static int iDivUp(int a, int b){
  return ((a % b != 0) ? (a / b + 1) : (a / b));
}

__global__ void convolKernel(float* d_buf, float* d_buf_aux, int nbl, int nbc){

	  int j = blockDim.x*blockIdx.x + threadIdx.x;
	  int i = blockDim.y*blockIdx.y + threadIdx.y;

	  if (i<nbl && j<nbc)
	  {

	  //Copie depuis convol.c
	  //=====================
	  /*** filtre moyenneur CONVOL_MOYENNE2 (filtre moyenneur avec
	     * un poid central plus fort):
	     * Rq: pour les bords, moyenne avec uniquement les cases presentes */
	  float denominateur = 0.0f;
	  float numerateur = 0.0f;
	  float poids_central;
 	  if (i<nbl-1){
	     numerateur += d_buf[(i+1)*nbc+j]; ++denominateur;
	     if (j>0){     numerateur += d_buf[(i+1)*nbc+j-1]; ++denominateur; }
	     if (j<nbc-1){ numerateur += d_buf[(i+1)*nbc+j+1]; ++denominateur; }
	  }
	  if (j>0){     numerateur += d_buf[(i)*nbc+j-1]; ++denominateur; }
	  if (j<nbc-1){ numerateur += d_buf[(i)*nbc+j+1]; ++denominateur; }
	  if (i>0){
	       numerateur +=  d_buf[(i-1)*nbc+j]; ++denominateur;
	       if (j>0){     numerateur += d_buf[(i-1)*nbc+j-1]; ++denominateur; }
	       if (j<nbc-1){ numerateur += d_buf[(i-1)*nbc+j+1]; ++denominateur; }
	  }
														 poids_central = denominateur*0.5f; /* poids central = 50% autres poids */
	 numerateur   += poids_central*d_buf[(i)*nbc+j];
	 denominateur += poids_central;

	d_buf_aux[i*nbc+j] = numerateur/denominateur;
	}
}					   


/**
 * Effectue 'nbiter' convolutions sur GPU et retourne
 * le pointeur vers le buffer contenant la derniere convolution. 
 */

extern "C"
float *gpu_multiples_convolutions(float buf[], 
				  float buf_aux[], 
				  int nbl, 
				  int nbc,
				  int nbiter, 
				  int nbThreadsParBloc){
  
  /*** TODO ***/;
  float *d_buf, *d_buf_aux;
  int grilleX, grilleY;
  int taille_alloc = nbc * nbl * sizeof(float);

  cudaMalloc((void **) &d_buf, taille_alloc);
  cudaMalloc((void **) &d_buf_aux, taille_alloc);

  cudaMemcpy(d_buf, buf, taille_alloc, cudaMemcpyHostToDevice);
  cudaMemcpy(d_buf_aux, buf_aux, taille_alloc, cudaMemcpyHostToDevice);

  grilleX = ceil((float)nbc/(float)nbThreadsParBloc);
  grilleY = ceil((float)nbl/(float)nbThreadsParBloc);
  dim3 threads_par_bloc(nbThreadsParBloc, nbThreadsParBloc);
  dim3 taille_grille(grilleX, grilleY);

  int i;
  for(i=0; i<nbiter; i++){
 	convolKernel<<<taille_grille, threads_par_bloc>>>(d_buf, d_buf_aux, nbl, nbc);
  	cudaMemcpy(d_buf, d_buf_aux, taille_alloc, cudaMemcpyDeviceToDevice);
 }

 cudaMemcpy(buf, d_buf, taille_alloc, cudaMemcpyDeviceToHost);
 return buf;			  
}

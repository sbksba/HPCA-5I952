/**
 * Programmation GPU 
 * Université Pierre et Marie Curie
 * Calcul de convolution sur une image.
 */

/**
 * Avec sauvegarde de toutes les images. 
 *
 * V0
 *
 */

#include <cuda.h>
#include <stdio.h>

extern "C" double my_gettimeofday();

/* Pour sauvegarde des images Rasterfile : */
#include "rasterfile.h"
typedef struct {
  struct rasterfile file;  ///< Entête image Sun Raster
  unsigned char rouge[256],vert[256],bleu[256];  ///< Palette de couleur
  unsigned char *data;    ///< Pointeur vers l'image
} Raster;
extern "C" void convert_float2uchar_image(float *p_f, unsigned char*p_ua, int h, int w);
extern "C" void sauve_rasterfile(char *nom, Raster *r);


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
				  int nbThreadsParBloc, 
				  Raster r, 
				  char *nom_sortie){
  
  /*** TODO ***/;

}



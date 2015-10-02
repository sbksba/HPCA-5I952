
###----------------------------------------------------------------------------
### Debugging:
#DEBUGGING=y


###----------------------------------------------------------------------------
CC=gcc 
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart
NVCC=nvcc 

ifeq ($(DEBUGGING), y)
 CFLAGS=-g 
 CUDA_FLAGS = --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_35,code=sm_35 -g -G #-DCUDA_DEBUG 
else
 CFLAGS=-O3
 CUDA_FLAGS = --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_35,code=sm_35 -O3 #-DCUDA_DEBUG 
endif 

BINS= convol_v0   # convol_download_v0


###
all: ${BINS}

convol.o: convol.c rasterfile.h 
	${CC} -c ${CFLAGS} $< 


### gpu_convol_v* 
gpu_convol_v%.o: gpu_convol_v%.cu 
	${NVCC} -c ${CUDA_FLAGS} $< 

convol_v%: convol.o gpu_convol_v%.o
	${CC} ${CFLAGS} -o $@ $+ ${LIB} 


### gpu_convol_download_v* 
# Chaque resultat de convolution est recupere dans la memoire hote. 
convol_download.o: convol.c rasterfile.h 
	${CC} -DDOWNLOAD -c ${CFLAGS} -o $@ $< 

gpu_convol_download_v%.o: gpu_convol_download_v%.cu
	${NVCC} -c ${CUDA_FLAGS} $< 

convol_download_v%: convol_download.o gpu_convol_download_v%.o 
	${CC} ${CFLAGS} -o $@ $+ ${LIB} 


####
clean: 
	rm -f ${BINS} *~ *.o 





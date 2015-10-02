
###----------------------------------------------------------------------------
### Debugging:
#DEBUGGING=y


###----------------------------------------------------------------------------
CC=gcc 
LIB=-lm 
ifeq ($(DEBUGGING), y)
 CFLAGS=-g 
else
 CFLAGS=-O3
endif 

all: convol

convol: convol.c rasterfile.h 
	${CC} ${CFLAGS} -o $@ $< ${LIB} 

clean: 
	rm -f convol *~ *.o 





#!/bin/sh
#

PATTERN="Temps calcul :"
OUTPUT="$0.output"

if [ $# -ne 3 ] ; then
    echo "Usage : $0 <exec> <nom image SunRaster> <nbiter> "
    exit 1;
fi 

EXEC=./$1
FILENAME=$2
NBITER=$3
#echo "EXEC=${EXEC} FILENAME=${FILENAME} NBITER=${NBITER}"

rm -f ${OUTPUT}
best_t=0
best_perf=0
echo -n "Testing : "
for t in ` seq 32 32 1024 ` ; do
    echo -n "$t "  
    perf=$( ${EXEC} ${FILENAME} ${NBITER} $t | tee -a ${OUTPUT} | grep -F "${PATTERN}" | awk '{print $NF;}' )
    if [ -z "$perf" ] ; then 
	echo "Motif \"${PATTERN}\" non trouve dans la sortie de l'executable..." 
	exit 1  
    fi
    cmp=$(echo "$perf > $best_perf" | bc)
    if [ $cmp -eq 1 ] ; then
	best_perf=${perf}
	best_t=$t 
    fi
done
echo ""

echo "Meilleure taille de bloc : $best_t (nb convolutions/s : ${best_perf})"







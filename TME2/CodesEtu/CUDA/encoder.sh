#!/bin/bash
#
#set -x 

echo "Conversion en .jpg ..."
for i in `ls *.ras` ; do
    if (echo $i | grep -q Sukhothai) ; then 
	convert -crop 1533x2040+0+0 -resize 50% $i ${i/%.ras/.jpg}
    else
	convert $i ${i/%.ras/.jpg}
    fi 
done 

echo "Encodage ... "
mencoder `ls -v *.jpg` -ovc lavc -demuxer lavf -lavfdopts format=mjpeg -o sortie.avi 

echo "Effacement des fichiers .jpg ..."
for i in `ls *.ras` ; do
    rm ${i/%.ras/.jpg}
done 


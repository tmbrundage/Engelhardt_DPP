#!/bin/sh

rep=0
while [ $rep -lt 1000 ]
do
    python temp.py $rep
    rep=`expr $rep + 1`
done

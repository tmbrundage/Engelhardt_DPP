#!/bin/sh

rep=0
while [ $rep -lt 1000 ]
do
    python LRcompareUCI.py 200
    python LRcompareUCI.py 500
    rep=`expr $rep + 1`
done

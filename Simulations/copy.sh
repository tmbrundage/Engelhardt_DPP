#!/bin/sh

for dir in */
do
    for nDir in $dir*/
    do
	#sourceDir="~/__Engelhardt/FUCKINGGIT/$nDir"
	sourceDir="../../FUCKINGGIT/$nDir"
	folder="StandardRegressions/"
	fullSourceDir="$sourceDir$folder"
	localDir="$nDir$folder"
	#echo "$fullServerDir\n"
	#echo "$localDir\n"
	#echo "\n"
	#mkdir $localDir
	cp -a "$fullSourceDir" $localDir
    done
done

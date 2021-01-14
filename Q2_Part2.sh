#!/bin/bash
#$ -l h_rt=4:00:00  #time needed
#$ -P rse-com6012
#$ -q rse-com6012.q
#$ -pe smp 10 #number of cores
#$ -l rmem=2G #number of memery
#$ -o Q2_Part2.output #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M zfoobozuwa1@shef.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory



module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

export SPARK_LOCAL_DIRS=$TMPDIR

spark-submit --driver-memory 2G --executor-memory 2G --master local[10] Q2_Part2.py
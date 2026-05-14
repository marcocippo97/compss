#!/usr/bin/env bash

if [ -f $COMPSS_HOME/compssenv ]; then
  source $COMPSS_HOME/compssenv
fi

export R_LIBS_USER=$COMPSS_HOME/Bindings/RCOMPSs/user_libs:$R_LIBS_USER
export LD_LIBRARY_PATH=$COMPSS_HOME/Bindings/bindings-common/lib:$LD_LIBRARY_PATH

seed=1
numpoints=10000
dimensions=10
num_centres=10
fragments=10
mode="normal"
iterations=5
epsilon=1e-9
arity=2

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
stdout_file="RCOMPSs-kmeans-$timestamp.out"
stderr_file="RCOMPSs-kmeans-$timestamp.err"
touch $stdout_file
touch $stderr_file

compss_clean_procs

runcompss \
  --lang=r \
  kmeans.R \
  --seed $seed \
  --numpoints $numpoints \
  --dimensions $dimensions \
  --num_centres $num_centres \
  --fragments $fragments \
  --mode $mode \
  --iterations $iterations \
  --epsilon $epsilon \
  --arity $arity \
  --plot FALSE \
  --RCOMPSs \
  --Minimize \
  >>$stdout_file 2>>$stderr_file

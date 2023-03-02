#!/bin/bash
set -x 

declare -A single_batch

single_batch=( ["Combo/combo"]=64 ["NT3/nt3"]=2 ["P1B2/p1b2"]=64 ["ST1/sct"]=64 ["TC1/tc1"]=20)
GPU_TYPE=$1
NUM_GPU=$2

TESTCASES="Combo/combo NT3/nt3 P1B2/p1b2 ST1/sct TC1/tc1"
for ARCH in $TESTCASES
do
    python ${ARCH}_baseline_keras2.py \
    --batch_size $((NUM_GPU*single_batch[$ARCH])) --gpu-type $GPU_TYPE \
    --num-gpu $NUM_GPU --iter-limit 300 &&
    sleep 30
done

# python Pilot1/NT3/nt3_baseline_keras2.py \
# --batch_size $(($2*single_batch[NT3])) --gpu-type $1 \
# --num-gpu $2

# python Pilot1/Combo/combo_baseline_keras2.py \
# --batch_size $(($2*single_batch[Combo])) --gpu-type $1 \
# --num-gpu $2

# python Pilot1/P1B2/p1b2_baseline_keras2.py \
# --batch_size $(($2*single_batch[P1B2])) --gpu-type $1 \
# --num-gpu $2

# python Pilot1/ST1/st1_baseline_keras2.py \
# --batch_size $(($2*single_batch[ST1])) --gpu-type $1 \
# --num-gpu $2

# python Pilot1/TC1/tc1_baseline_keras2.py \
# --batch_size $(($2*single_batch[TC1])) --gpu-type $1 \
# --num-gpu $2

#!/bin/bash

python Pilot1/NT3/nt3_baseline_keras2.py \
--batch_size $(($2*single_batch[NT3])) --gpu-type $1 \
--num-gpu $2

python Pilot1/Combo/combo_baseline_keras2.py \
--batch_size $(($2*single_batch[Combo])) --gpu-type $1 \
--num-gpu $2

python Pilot1/P1B2/p1b2_baseline_keras2.py \
--batch_size $(($2*single_batch[P1B2])) --gpu-type $1 \
--num-gpu $2

python Pilot1/ST1/st1_baseline_keras2.py \
--batch_size $(($2*single_batch[ST1])) --gpu-type $1 \
--num-gpu $2

python Pilot1/TC1/tc1_baseline_keras2.py \
--batch_size $(($2*single_batch[TC1])) --gpu-type $1 \
--num-gpu $2

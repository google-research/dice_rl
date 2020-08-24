#!/bin/bash
BIN=blaze-bin/third_party/py/dice_rl/google/scripts/run_load_pytorch_policy
POLICYDIR=/cns/lu-d/home/offline-rl/public_data/d4rl/ope_target_policies/
LOGDIR=$HOME/tmp/ope/logs/

mkdir -p $LOGDIR

declare -A ENV_MAP
ENV_MAP["HalfCheetah-v2"]="cheetah"
ENV_MAP["Hopper-v2"]="hopper"
ENV_MAP["Walker2d-v2"]="walker2d"

blaze build -c opt --copt=-mavx third_party/py/dice_rl/google/scripts:run_load_pytorch_policy

for ENV in ${!ENV_MAP[@]}; do
  for DATASET in "" _random _medium
  do
    ENV_LOWER=${ENV_MAP["$ENV"]}
    echo $ENV $DATASET
    $BIN --env_name=$ENV \
         --target_policy=$POLICYDIR/${ENV_LOWER}${DATASET}_params.pkl \
         &> $LOGDIR/${ENV_LOWER}${DATASET}_params.log
  done
done

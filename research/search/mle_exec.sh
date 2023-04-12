#!/bin/sh
# GET_CONFIGS_READY FOR BASH FILE!
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -exp_dir|--experiment_dir)
    EXPERIMENT_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    -config|--config_fname)
    CONFIG_FNAME="$2"
    shift # past argument
    shift # past value
    ;;
    -seed|--seed_id)
    SEED_ID="$2"
    shift # past argument
    shift # past value
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

neb --config_fname ${CONFIG_FNAME}  --seed_id ${SEED_ID} --experiment_dir ${EXPERIMENT_DIR}

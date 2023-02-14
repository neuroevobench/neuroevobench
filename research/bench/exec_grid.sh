if [[ "$1" == "popsizes" ]]
then
    echo "Run tuned ES experiments"
    # OpenES
    # mle run configs/tuned.yaml \
    #     --purpose OpenES ant \
    #     --base_train_config configs/OpenES/train/brax/ant.yaml \
    #     --experiment_dir experiments/tuned/OpenES/ant
    # mle run configs/tuned.yaml \
    #     --purpose OpenES halfcheetah \
    #     --base_train_config configs/OpenES/train/brax/halfcheetah.yaml \
    #     --experiment_dir experiments/tuned/OpenES/halfcheetah
    # mle run configs/tuned.yaml \
    #     --purpose OpenES humanoid \
    #     --base_train_config configs/OpenES/train/brax/humanoid.yaml \
    #     --experiment_dir experiments/tuned/OpenES/humanoid
    # mle run configs/tuned.yaml \
    #     --purpose OpenES fetch \
    #     --base_train_config configs/OpenES/train/brax/fetch.yaml \
    #     --experiment_dir experiments/tuned/OpenES/fetch
    # SNES
    # mle run configs/tuned.yaml \
    #     --purpose SNES ant \
    #     --base_train_config configs/SNES/train/brax/ant.yaml \
    #     --experiment_dir experiments/tuned/SNES/ant
    # mle run configs/tuned.yaml \
    #     --purpose SNES halfcheetah \
    #     --base_train_config configs/SNES/train/brax/halfcheetah.yaml \
    #     --experiment_dir experiments/tuned/SNES/halfcheetah
    # mle run configs/tuned.yaml \
    #     --purpose SNES humanoid \
    #     --base_train_config configs/SNES/train/brax/humanoid.yaml \
    #     --experiment_dir experiments/tuned/SNES/humanoid
    # mle run configs/tuned.yaml \
    #     --purpose SNES fetch \
    #     --base_train_config configs/SNES/train/fetch.yaml \
    #     --experiment_dir experiments/tuned/SNES/fetch
    # PGPE
    # mle run configs/tuned.yaml \
    #     --purpose PGPE ant \
    #     --base_train_config configs/PGPE/train/brax/ant.yaml \
    #     --experiment_dir experiments/tuned/SNES/ant
    # mle run configs/tuned.yaml \
    #     --purpose PGPE halfcheetah \
    #     --base_train_config configs/PGPE/train/brax/halfcheetah.yaml \
    #     --experiment_dir experiments/tuned/PGPE/halfcheetah
    # mle run configs/tuned.yaml \
    #     --purpose PGPE humanoid \
    #     --base_train_config configs/PGPE/train/brax/humanoid.yaml \
    #     --experiment_dir experiments/tuned/PGPE/humanoid
    # mle run configs/tuned.yaml \
    #     --purpose PGPE fetch \
    #     --base_train_config configs/PGPE/train/brax/fetch.yaml \
    #     --experiment_dir experiments/tuned/PGPE/fetch
    # Sep_CMA_ES
    # mle run configs/tuned.yaml \
    #     --purpose Sep_CMA_ES ant \
    #     --base_train_config configs/Sep_CMA_ES/train/brax/ant.yaml \
    #     --experiment_dir experiments/tuned/Sep_CMA_ES/ant
    # mle run configs/tuned.yaml \
    #     --purpose Sep_CMA_ES halfcheetah \
    #     --base_train_config configs/Sep_CMA_ES/train/brax/halfcheetah.yaml \
    #     --experiment_dir experiments/tuned/Sep_CMA_ES/halfcheetah
    # mle run configs/tuned.yaml \
    #     --purpose Sep_CMA_ES humanoid \
    #     --base_train_config configs/Sep_CMA_ES/train/brax/humanoid.yaml \
    #     --experiment_dir experiments/tuned/Sep_CMA_ES/humanoid
    # mle run configs/tuned.yaml \
    #     --purpose Sep_CMA_ES fetch \
    #     --base_train_config configs/Sep_CMA_ES/train/brax/fetch.yaml \
    #     --experiment_dir experiments/tuned/Sep_CMA_ES/fetch
elif [[ "$1" == "modelsizes" ]]
then
    echo "Run Sep-CMA-ES Grid Experiments"
else
    echo "Provide valid argument to bash script"
fi
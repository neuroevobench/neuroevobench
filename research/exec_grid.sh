if [[ "$1" == "OpenES" ]]
then
    echo "Run Open ES Grid Experiments"
    mle run configs/OpenES/search.yaml \
        --purpose Grid OpenES F-MNIST \
        --base_train_config configs/OpenES/fmnist.yaml \
        --experiment_dir experiments/OpenES/fmnist
    mle run configs/OpenES/search.yaml \
        --purpose Grid OpenES Ant \
        --base_train_config configs/OpenES/ant.yaml \
        --experiment_dir experiments/OpenES/ant
    mle run configs/OpenES/search.yaml \
        --purpose Grid OpenES Humanoid \
        --base_train_config configs/OpenES/humanoid.yaml \
        --experiment_dir experiments/OpenES/humanoid
    mle run configs/OpenES/search.yaml \
        --purpose Grid OpenES SpaceInvaders \
        --base_train_config configs/OpenES/spaceinvaders.yaml \
        --experiment_dir experiments/OpenES/spaceinvaders
    mle run configs/OpenES/search.yaml \
        --purpose Grid OpenES Breakout \
        --base_train_config configs/OpenES/breakout.yaml \
        --experiment_dir experiments/OpenES/breakout
elif [[ "$1" == "Sep_CMA_ES" ]]
then
    echo "Run Sep-CMA-ES Grid Experiments"
    mle run configs/Sep_CMA_ES/search.yaml \
        --purpose Grid Sep_CMA_ES Ant \
        --base_train_config configs/Sep_CMA_ES/ant.yaml \
        --experiment_dir experiments/Sep_CMA_ES/ant
    mle run configs/Sep_CMA_ES/search.yaml \
        --purpose Grid Sep_CMA_ES Humanoid \
        --base_train_config configs/Sep_CMA_ES/humanoid.yaml \
        --experiment_dir experiments/Sep_CMA_ES/humanoid
else
    echo "Provide valid argument to bash script"
fi
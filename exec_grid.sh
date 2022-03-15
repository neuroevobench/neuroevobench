mle run configs/Open_ES/search_cart.yaml \
    --base_train_config configs/Open_ES/cartpole.yaml \
    --experiment_dir experiments/Open_ES/cartpole \
    --purpose Grid OpenES Cartpole
mle run configs/Open_ES/search_ant.yaml \
    --base_train_config configs/Open_ES/ant.yaml \
    --experiment_dir experiments/Open_ES/ant \
    --purpose Grid OpenES Ant
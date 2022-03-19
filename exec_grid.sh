if [[ "$1" == "OpenES" ]]
then
    echo "Run Open ES Grid Experiments"
    mle run configs/Open_ES/search_cart.yaml \
        --purpose Grid OpenES Cartpole
    mle run configs/Open_ES/search_mnist.yaml \
        --purpose Grid OpenES MNIST
    mle run configs/Open_ES/search_ant.yaml \
        --purpose Grid OpenES Ant
elif [[ "$1" == "PGPE" ]]
then
    echo "Run PGPE Grid Experiments"
    mle run configs/PGPE/search_cart.yaml \
        --purpose Grid PGPE Cartpole
    mle run configs/PGPE/search_mnist.yaml \
        --purpose Grid PGPE MNIST
    mle run configs/PGPE/search_ant.yaml \
        --purpose Grid PGPE Ant
elif [[ "$1" == "ARS" ]]
then
    echo "Run ARS Grid Experiments"
    mle run configs/ARS/search_cart.yaml \
        --purpose Grid ARS Cartpole
    mle run configs/ARS/search_mnist.yaml \
        --purpose Grid ARS MNIST
    mle run configs/ARS/search_ant.yaml \
        --purpose Grid ARS Ant
elif [[ "$1" == "Sep-CMA-ES" ]]
then
    echo "Run Sep-CMA-ES Grid Experiments"
    mle run configs/Sep_CMA_ES/search_cart.yaml \
        --purpose Grid Sep-CMA-ES Cartpole
    mle run configs/Sep_CMA_ES/search_mnist.yaml \
        --purpose Grid Sep-CMA-ES MNIST
    mle run configs/Sep_CMA_ES/search_ant.yaml \
        --purpose Grid Sep-CMA-ES Ant
elif [[ "$1" == "CMA-ES" ]]
then
    echo "Run CMA-ES Grid Experiments"
    mle run configs/CMA_ES/search_cart.yaml \
        --purpose Grid CMA-ES Cartpole
    mle run configs/CMA_ES/search_mnist.yaml \
        --purpose Grid CMA-ES MNIST
    mle run configs/CMA_ES/search_ant.yaml \
        --purpose Grid CMA-ES Ant
elif [[ "$1" == "SimpleGA" ]]
then
    echo "Run GA Grid Experiments"
    mle run configs/Simple_GA/search_cart.yaml \
        --purpose Grid GA Cartpole
    mle run configs/Simple_GA/search_mnist.yaml \
        --purpose Grid GA MNIST
    mle run configs/Simple_GA/search_ant.yaml \
        --purpose Grid GA Ant
else
    echo "Provide valid argument to bash script"
fi
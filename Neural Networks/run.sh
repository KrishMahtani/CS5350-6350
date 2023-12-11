#!/bin/bash

echo "running testing net, question 2 a"
python3 testing_net.py

echo "running neural net with random weight initialization, question 2 b"
python3 random_init.py

echo "running neural net with zeros weight initialization, question 2 c"
python3 zeroes_init.py

echo "running pytorch network, question 2 e"
python3 pytorch_net.py

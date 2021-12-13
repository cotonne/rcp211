# Deep A3C 

Ce répositorie contient le code pour le projet [RCP211](http://cedric.cnam.fr/vertigo/cours/RCP211/projet_S1.html).

> A3C : Apprenez un agent à jouer à pacman avec une approche de type Deep Asynchronous Advantage Actor Critic : Methods for Deep Reinforcement Learning . Vous pouvez considérer l’aspect asynchrone comme un bonus.

Basé sur [Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)

## Installation

```
python3.9 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar 
unzip ROMS.zip "ROMS/Pac-Man (1982) (Atari, Tod Frye) (CX2646) (PAL).bin"
ale-import-roms ROMS
```

## Execution

```
. venv/bin/activate
python main.py
```

# Références

 - [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
 - [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
 - [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
 - [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
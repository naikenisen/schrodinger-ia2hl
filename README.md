# Pont de Schrödinger Diffusif avec Applications à la Modélisation Générative basée sur le Score

Ce dépôt contient l'implémentation de l'article "Pont de Schrödinger Diffusif avec Applications à la Modélisation Générative basée sur le Score".

Si vous utilisez ce code, veuillez citer l'article :
```
    @article{de2021diffusion,
              title={Diffusion Schr$\backslash$" odinger Bridge with Applications to Score-Based Generative Modeling},
              author={De Bortoli, Valentin et Thornton, James et Heng, Jeremy et Doucet, Arnaud},
              journal={arXiv preprint arXiv:2106.01357},
              year={2021}
            }
```

Contributeurs
------------

*  Valentin De Bortoli  
*  James Thornton
*  Jeremy Heng
*  Arnaud Doucet

Qu'est-ce qu'un pont de Schrödinger ?
-----------------------------

Le problème du Pont de Schrödinger (SB) est un problème classique en mathématiques appliquées, contrôle optimal et probabilité ; voir [1, 2, 3]. En temps discret, il prend la forme dynamique suivante : on considère une densité de référence p(x<sub>0:N</sub>) décrivant le processus d'ajout de bruit aux données. On cherche à trouver p\*(x<sub>0:N</sub>) telle que p\*(x<sub>0</sub>) = p<sub>data</sub>(x<sub>0</sub>) et p\*(x<sub>N</sub>) = p<sub>prior</sub>(x<sub>N</sub>), tout en minimisant la divergence de Kullback-Leibler entre p\* et p. Dans ce travail, nous introduisons le **Pont de Schrödinger Diffusif** (DSB), un nouvel algorithme utilisant des approches de score-matching [4] pour approximer l'algorithme *Iterative Proportional Fitting*, une méthode itérative pour résoudre le problème SB. DSB peut être vu comme un raffinement des méthodes existantes de modélisation générative basée sur le score [5, 6].

![Pont de Schrödinger](schrodinger_bridge.png)

Installation
-------------

Ce projet peut être installé depuis son dépôt git.

1. Obtenez les sources :
    
    `git clone https://github.com/anon284/schrodinger_bridge.git`

ou, si `git` n'est pas disponible, téléchargez en ZIP depuis GitHub https://github.com/<repository>.
  
2. Installation :

```bash
pip install -r requirements.txt
```

3. Téléchargez des exemples de données :

    - CelebA : `python data.py --data celeba --data_dir './data/' `
    - MNIST :  `python data.py --data mnist --data_dir './data/' `

Comment utiliser ce code ?
---------------------

3. Entraînez les réseaux :
  - 2d :  `python main.py dataset=2d model=Basic num_steps=20 num_iter=5000`
  - mnist : `python main.py dataset=stackedmnist num_steps=30 model=UNET num_iter=5000 data_dir=<chemin vers le dossier data>`
  - celeba : `python main.py dataset=celeba num_steps=50 model=UNET num_iter=5000 data_dir=<chemin vers le dossier data>`

Les checkpoints et les images générées seront sauvegardés dans un nouveau dossier. Si la mémoire GPU est insuffisante, réduisez la taille du cache. Le dataset 2D doit être entraîné sur CPU. MNIST et CelebA ont été entraînés sur 2 GPU V100 à grande mémoire.

Références
----------

.. [1] Hans Föllmer
       *Champs aléatoires et processus de diffusion*
       In: École d'été de Probabilités de Saint-Flour 1985-1987

.. [2] Christian Léonard 
       *Un aperçu du problème de Schrödinger et de ses liens avec le transport optimal*
       In: Discrete & Continuous Dynamical Systems-A 2014

.. [3] Yongxin Chen, Tryphon Georgiou et Michele Pavon
       *Transport optimal dans les systèmes et le contrôle*
       In: Annual Review of Control, Robotics, and Autonomous Systems 2020

.. [4] Aapo Hyvärinen et Peter Dayan
       *Estimation de modèles statistiques non normalisés par score matching*
       In: Journal of Machine Learning Research 2005

.. [5] Yang Song et Stefano Ermon
       *Modélisation générative par estimation des gradients de la distribution des données*
       In: Advances in Neural Information Processing Systems 2019

.. [6] Jonathan Ho, Ajay Jain et Pieter Abbeel
       *Modèles probabilistes de diffusion pour la débruitage*
       In: Advances in Neural Information Processing Systems 2020

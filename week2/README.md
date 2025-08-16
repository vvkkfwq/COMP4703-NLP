Lab 1 install instructions.
1. Ensure that you have anaconda or miniconda installed and that conda is in your path.
2. conda create -n "COMP4703Lab1" python=3.8
3. conda activate COMP4703Lab1
4. pip install jupyterlab notebook numpy matplotlib
7. jupyter notebook or jupyter lab

An alternative is to use the requirements.txt file included to install the required
packages. This can be accomplished using ``pip install -r requirements.txt``.

If you are using Windows, I strongly advise you to install WSL2 on your machine so that you have a proper anaconda setup and to ensure that you can easily work in weekly virtual environments. I have included a cheat sheet on how to install WSL on windows. It should take no more than 10 minutes and is good experience for future work.

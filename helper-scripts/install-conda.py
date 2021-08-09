# try to get the bare minimum to get a new conda env working
conda_path = ''
try:
    conda_path = !which conda
finally:
    print('')

if (len(conda_path) == 0):
    print('installing miniconda')
    !wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh && bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -bfp /usr/local
    !conda update conda -y -q
    !source /usr/local/etc/profile.d/conda.sh
    !conda init
    !conda install -n root _license -y -q
else:
    print('found miniconda')

conda_envs = !conda env list
res = [i for i in conda_envs if 'open-mmlab' in i]
if (len(res) == 0):
    print('not found open-mmlab env', len(res))
    !conda create -y -q --name open-mmlab python=3.7
else:
    print('found open-mmlab env', len(res))
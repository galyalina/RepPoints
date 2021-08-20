pip uninstall torch -y

##Uninstall the current CUDA version
#apt-get --purge remove cuda nvidia* libnvidia-*
#dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
#apt-get remove cuda-*
#apt autoremove
#apt-get update
#
##Download CUDA 10.0
#wget  --no-clobber https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
##install CUDA kit dpkg
#dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
#sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
#apt-get update
#apt-get install cuda-10-0

#Get conda
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh && bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -bfp /usr/local
conda update conda -y -q
source /usr/local/etc/profile.d/conda.sh
conda init
conda install -n root _license -y -q
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

# install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.0 -c pytorch -y

# install the latest mmcv
pip install mmcv==0.2.14
pip install ipykernel
# install mmdetection
cd RepPoints
pip install -r requirements.txt
bash init.sh
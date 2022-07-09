# DistantHomologyDetection
Detecting functional relationships between proteins using DL

## Quick Tutorial:

1. Type ```git clone https://github.com/martinez-zacharya/DistantHomologyDetection``` in your home directory on the HPC
3. Download Miniconda by running ```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh``` and then ```sh ./Miniconda3-latest-Linux-x86_64.sh```.
4. Run ```conda env create -f environment.yml``` in the home directory of the repo to set up the proper conda environment.
5. Shift your current working directory to the scripts folder with ```cd scripts```.
6. Type ```vi tutorial_slurm``` to open the slurm file and then hit ```i```.
7. Change the email in the tutorial_slurm file to your email (You can use https://s3-us-west-2.amazonaws.com/imss-hpc/index.html to make your own slurm files in the future).
8. Save the file by first hitting escape and then entering ```:x``` to exit and save the file. 
9. You can view the arguments for the command line tool by typing ```python3 main.py -h```.
10. To run the tutorial analysis, make the tutorial slurm file exectuable with ```chmod +x tutorial_slurm.sh``` and then type ```sbatch tutorial_slurm.sh```.
11. You can now safely exit the ssh instance to the HPC if you want

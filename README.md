# DistantHomologyDetection
Detecting functional relationships between proteins using DL

## Quick Tutorial:

# Don't use this tutorial, not working anymore

1. Type ```git clone https://github.com/martinez-zacharya/DistantHomologyDetection``` in your home directory on the HPC
3. Download Miniconda by running ```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh``` and then ```sh ./Miniconda3-latest-Linux-x86_64.sh```.
4. Run ```conda env create -f environment.yml``` in the home directory of the repo to set up the proper conda environment.
5. Shift your current working directory to the scripts folder with ```cd scripts```.
6. Type ```vim tutorial_slurm``` to open the slurm file and then hit ```i```.
7. Change the email in the tutorial_slurm file to your email (You can use https://s3-us-west-2.amazonaws.com/imss-hpc/index.html to make your own slurm files in the future).
8. Save the file by first hitting escape and then entering ```:x``` to exit and save the file. 
9. Activate your conda environment by typing ```conda activate RemoteHomologyTransformer```.
10. You can view the arguments for the command line tool by typing ```python3 main.py -h```.
11. To run the tutorial analysis, run ```sbatch tutorial_slurm```.
12. Remember, don't run big jobs on the login nodes on the HPC, only submit them using slurm (If this is confusing, just let me know and I can explain more).
13. You can now safely exit the ssh instance to the HPC if you want

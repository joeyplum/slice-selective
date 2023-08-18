# slice-selective
A repository that contains simulations for 2D slice selective and non-selective RF excitations.

## Clone the repository.

1. Ensure SSH is setup for Git.
2. Navigate to folder you want to clone repository to.
3. `git clone git@github.com:joeyplum/slice-selective.git`


## Installation of environment.

Navigate to the folder containing this code inside the terminal. Run the following commands in sequence to create your environment.

1. `conda update -n base -c defaults conda`
2. `make conda`
3. `conda activate slice-selective`
4. `make pip`

## For activating display in WSL2 (optional):

Install XLaunch in Windows if not done already. 

1. `export DISPLAY=`grep -oP "(?<=nameserver ).+" /etc/resolv.conf`:0.0`
2. `export LIBGL_ALWAYS_INDIRECT=1`
3. Open XLaunch and set up (tick all three boxes)
4. Test with `xeyes`

## Uninstall.

To uninstall, run the following commands:

1. `conda activate`
2. `make clean`
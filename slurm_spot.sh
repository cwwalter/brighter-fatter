#!/bin/bash
# Job name:
#SBATCH --job-name=BF-spots
#
# Partition:
#SBATCH --partition=shared
#
# CPUs
#SBATCH --ntasks=1
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=00:05:00
#
# Output
#SBATCH --output=logs/spot-%a-perfect_seeing.log
#SBATCH --error=logs/spot-%a-perfect_seeing.log
#
# Array
#SBATCH --array=1-21
#
## Command(s) to run:

phosimdir=/global/u1/c/cwalter/PhoSim/phosim
workdir=/global/u1/c/cwalter/brighter-fatter

Initialdir=$phosimdir
Executable=$phosimdir/phosim

# Set the command file; First no effects
COMMAND_DIR=command-files
COMMAND_FILE=perfect_seeing

# We need to set the source file depending on the array numbrer
SOURCE_DIR=sources

case $SLURM_ARRAY_TASK_ID in
    1) SOURCE_FILE=1000e ;; 
    2) SOURCE_FILE=2000e ;;
    3) SOURCE_FILE=3000e ;;
    4) SOURCE_FILE=4000e ;;
    5) SOURCE_FILE=5000e ;;
    6) SOURCE_FILE=10000e ;;
    7) SOURCE_FILE=15000e ;;
    8) SOURCE_FILE=20000e ;;
    9) SOURCE_FILE=25000e ;;
    10) SOURCE_FILE=30000e ;;
    11) SOURCE_FILE=50000e ;;
    12) SOURCE_FILE=75000e ;;
    13) SOURCE_FILE=100000e ;;
    14) SOURCE_FILE=200000e ;;
    15) SOURCE_FILE=500000e ;;
    16) SOURCE_FILE=750000e ;;
    17) SOURCE_FILE=1000000e ;;
    18) SOURCE_FILE=1250000e ;;
    19) SOURCE_FILE=1500000e ;;
    20) SOURCE_FILE=1750000e ;;
    21) SOURCE_FILE=2000000e ;;
    *)
	echo "TASK_ID not defined!"
	exit
esac
	
SOURCE=$workdir/$SOURCE_DIR/$SOURCE_FILE
COMMANDS=$workdir/$COMMAND_DIR/$COMMAND_FILE
OPTIONS='-s R22_S11 -e 0'

Arguments="$SOURCE $OPTIONS -c $COMMANDS -w $workdir/work_spot -o $workdir/output"

echo RUN ON `date`
echo
echo EXECUTABLE: $Executable
echo SOURCE: $SOURCE
echo OPTIONS: $OPTIONS
echo
echo FULL COMMAND: $Executable $Arguments

cd $Initialdir
/usr/bin/time -p $Executable $Arguments

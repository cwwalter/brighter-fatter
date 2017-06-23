#!/bin/bash
# Job name:
#SBATCH --job-name=BF-flats
#
# Partition:
#SBATCH --partition=shared
#
# CPUs
#SBATCH --ntasks=1
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
# Output
#SBATCH --output=logs/flat-%a.log
#SBATCH --error=logs/flat-%a.log
#
# Array
#SBATCH --array=1-14,101-114

# Set directories
phosimdir=/global/u1/c/cwalter/PhoSim/phosim
workdir=/global/u1/c/cwalter/brighter-fatter
outputdir=/global/cscratch1/sd/cwalter/brighter-fatter

Initialdir=$phosimdir
Executable=$phosimdir/phosim

# Set the command file (upper third digit)
COMMAND_DIR=command-files

case $((SLURM_ARRAY_TASK_ID/100)) in
    0) COMMAND_FILE=perfect_seeing ;;
    1) COMMAND_FILE=dev_charge_sharing ;;
    *)
	echo "Command File not defined!"
	exit
esac

# Set the source file depending on the array number (lowest two digits)
SOURCE_DIR=sources

case $((SLURM_ARRAY_TASK_ID%100)) in
    1) SOURCE_FILE=flat10_0 ;; 
    2) SOURCE_FILE=flat10_1 ;;
    3) SOURCE_FILE=flat11_0 ;;
    4) SOURCE_FILE=flat11_1 ;;
    5) SOURCE_FILE=flat12_0 ;;
    6) SOURCE_FILE=flat12_1 ;;
    7) SOURCE_FILE=flat13_0 ;;
    8) SOURCE_FILE=flat13_1 ;;
    9) SOURCE_FILE=flat14_0 ;;
    10) SOURCE_FILE=flat14_1 ;;
    11) SOURCE_FILE=flat15_0 ;;
    12) SOURCE_FILE=flat15_1 ;;
    13) SOURCE_FILE=flat18_0 ;;
    14) SOURCE_FILE=flat18_1 ;;
    *)
	echo "TASK_ID not defined!"
	exit
esac
	
SOURCE=$workdir/$SOURCE_DIR/$SOURCE_FILE
COMMANDS=$workdir/$COMMAND_DIR/$COMMAND_FILE
OPTIONS='-s R22_S11 -e 0 -i lsst_flats'
OUTPUT="-w $outputdir/work_flat -o $outputdir/output"
Arguments="$SOURCE $OPTIONS -c $COMMANDS $OUTPUT"

echo JOB $SLURM_JOB_ID $SLURM_ARRAY_TASK_ID
echo RUN ON `date`
echo
echo EXECUTABLE: $Executable
echo SOURCE: $SOURCE
echo COMMANDS: $COMMANDS
echo OPTIONS: $OPTIONS
echo
echo FULL COMMAND: $Executable $Arguments

cd $Initialdir
/usr/bin/time -p $Executable $Arguments

#!/bin/bash

#SBATCH --nodes=1 # request one node

#SBATCH --cpus-per-task=8  # ask for 8 cpus

#SBATCH --mem=16G # Maximum amount of memory this job will be given, try to estimate this to the best of your ability. This asks for 128 GB of ram.

#SBATCH --gres=gpu:1 #If you just need one gpu, you're done, if you need more you can change the number
#SBATCH --nodelist singularity
#SBATCH --partition=gpu #specify the gpu partition
#asdas SBATCH --nodelist frost-3 gpu01 gpu03 matrix singularity frost-1 frost-2 frost-4 frost-5 frost-6 frost-7

#SBATCH --time=0-40:00:00 # ask that the job be allowed to run for 2 days, 2 hours, 30 minutes, and 2 seconds.

# everything below this line is optional, but are nice to have quality of life things

#SBATCH --output=vgg_fedprox_yes_new.%J.out # tell it to store the output console text to a file called job.<assigned job number>.out

#SBATCH --error=vgg_fedprox_yes_new.%J.err # tell it to store the error messages from the program (if it doesn't write them to normal console output) to a file called job.<assigned job muber>.err

#SBATCH --job-name="vgg_fedprox_yes_new" # a nice readable name to give your job so you know what it is when you see it in the queue, instead of just numbers

# under this we just do what we would normally do to run the program, everything above this line is used by slurm to tell it what your job needs for resources

# let's load the modules we need to do what we're going to do

cd /work/LAS/jannesar-lab/dphuong/aaai

# let's load the modules we need to do what we're going to do
#source /work/LAS/jannesar-lab/yusx/anaconda3/bin/activate /work/LAS/jannesar-lab/yusx/anaconda3/envs/gnnrl
source /work/LAS/jannesar-lab/dphuong/anaconda3/bin/activate /work/LAS/jannesar-lab/dphuong/anaconda3/envs/fl
#source activate Model_Compression
# the commands we're running are below

nvidia-smi

python experiments.py \
--model=vgg \
--dataset=cifar10 \
--alg=fedprox \
--lr=0.01 \
--batch-size=64 \
--epochs=10 \
--n_parties=10 \
--rho=0.9 \
--comm_round=100 \
--partition=noniid-labeldir \
--beta=0.1 \
--device='cuda' \
--datadir='./data/' \
--logdir='./logs/'  \
--noise=0 \
--sample=1 \
--init_seed=0 \
--train-flag \
--gate \
--ratio 0.7 \
--log_file_name='vgg_fedprox_yes_new' \
--mu=0.01 \
--dynamic-pruning \

#srun --nodes 1 --tasks 1 --cpus-per-task=8 --mem=64G --partition interactive --gres=gpu:1 --partition=gpu --time 8:00:00 --pty /usr/bin/bash

#chmod +x script-name-here.sh

#sbatch slurm.sh

#scontrol show job 308878
#scontrol show job 308875
#--gres=gpu:1
#--gres=gpu:v100-pcie-16G:1
#--dynamic-pruning \
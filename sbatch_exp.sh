#!/bin/bash

PM=$1


sbatch <<EOF
#!/bin/bash
#SBATCH --gres=gpu:t4:1
#SBATCH -c 10
#SBATCH --mem=40GB
#SBATCH --time=8:00:00
#SBATCH --job-name=PM_$PM
#SBATCH --output=exp.log
#SBATCH --qos=m2
#SBATCH --exclude=gpu096
# activate environment
module load pytorch1.13-cuda11.7-python3.7



python train-R1eps.py --total_epochs 120 \
				--lambda_P 0 \
				--lambda_PM $PM \
				--lambda_MSE 1 \
				--lambda_gp 10 \
				--bs 64 \
				--skip_fq 10 \
				--dim 128 \
				--z_dim 4 \
				--L 2 \
				--d_penalty 0 \
				--pre_path './saved_models/R1-eps|_dim_128|z_dim_4|L_2|lambda_gp_10|bs_64|dpenalty_0.0|lambdaP_0.0|lambdaPM_0.0|lambdaMSE_1.0' \
				--path './data/' 

#./saved_models/R1-eps|_dim_128|z_dim_4|L_2|lambda_gp_10|bs_64|dpenalty_0.0|lambdaP_0.0|lambdaPM_0.0|lambdaMSE_1.0' \

EOF

#!/bin/bash

sbatch <<EOF
#!/bin/bash
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -c 10
#SBATCH --mem=40GB
#SBATCH --time=12:00:00
#SBATCH --job-name=$1_mse
#SBATCH --output=exp.log
#SBATCH --qos=normal
#SBATCH --exclude=gpu096,gpu106,gpu071,gpu069,gpu062
# activate environment
module load pytorch2.0-cuda11.8-python3.9
#pytorch1.7.1-cuda11.0-python3.6 
#pytorch1.13-cuda11.7-python3.7


python train-inf-eps1-eps2.py --total_epochs 100 \
                                --lambda_JD 0 \
                                --lambda_MSE 1 \
                                --lambda_NEW 0 \
                                --lambda_FMD 0 \
                                --skip_fq 1 \
                                --eps_1 2 \
				--eps_2 $1\
                                --step 20 \
                                --single_bit 0 \
                                --pre_path 'None' \
				--pre_path_fixed 'saved_models/inf-eps/step_20/inf-2|lambdaJD_0.16|lambdaFMD_0.0|lambdaNEW_0.0|lambdaMSE_1.0|dataset_mmnist_4_axis_random_sample_step|buu_version' \
				--pre_trained 'JD' \
                                --pre_trained_lambda 0.16\
				--path './data/'\
				--dataset 'mmnist_4_axis_random_sample_step'\
				--learning_rate 5

#'./saved_models/inf-eps/inf-2|lambdaJD_0.0|lambdaFMD_0.0|lambdaNEW_0.0|lambdaMSE_1.0'

EOF

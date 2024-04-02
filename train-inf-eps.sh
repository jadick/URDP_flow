#!/bin/bash

sbatch <<EOF
#!/bin/bash
#SBATCH --gres=gpu:t4:1
#SBATCH -c 10
#SBATCH --mem=40GB
#SBATCH --time=16:00:00
#SBATCH --job-name=$1-fmd-inf-eps
#SBATCH --output=exp.log
#SBATCH --qos=normal
#SBATCH --exclude=gpu096,gpu106,gpu071,gpu069,gpu062
# activate environment
module load pytorch2.0-cuda11.8-python3.9
#pytorch1.7.1-cuda11.0-python3.6 
#pytorch1.13-cuda11.7-python3.7


python train-inf-eps.py --total_epochs 300 \
                                --lambda_JD 0 \
                                --lambda_MSE 1 \
                                --lambda_NEW 0 \
                                --lambda_FMD $1 \
                                --skip_fq 5 \
				--eps 2 \
				--step 15 \
				--single_bit 0 \
                                --pre_path './saved_models/inf-eps/step_15/inf-2|lambdaJD_0.0|lambdaFMD_0.0|lambdaNEW_0.0|lambdaMSE_1.0'\
                                --path './data/' 


#'./saved_models/inf-eps/inf-2|lambdaJD_0.0|lambdaFMD_0.0|lambdaNEW_0.0|lambdaMSE_1.0'

EOF

#!/bin/bash

#SBATCH --job-name=copying_ADHD_data
#SBATCH --output=output_copying_ADHD_data
#SBATCH --error=errors_copying_ADHD_data
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=16gb
#SBATCH --mail-user=bergerar@bc.edu
#SBATCH --mail-type=ALL

bucket_link='s3://fcp-indi/data/Projects/ADHD200'
destination='adhd200aliciasattempt' 
aws s3 cp $bucket_link $destination --recursive --no-sign-request


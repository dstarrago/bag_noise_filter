
#!/bin/bash

#$ -N NNIPF-0%

#$ -q muylarga

#$ -o array_output.$TASK_ID.dat

#$ -e array_error.$TASK_ID.dat

# Number of tasks
#$ -t 1-1

#$ -cwd

# DATASET[0]="33 WIR8"
# DATASET[1]="34 WIR9"

java -cp NoisyMIL.jar Utils.InstanceLevelNoiseFilterTest DATASET[$SGE_TASK_ID-1]


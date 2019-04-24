
#!/bin/bash

#$ -N BN1

#$ -q memoria

#$ -o output.dat

#$ -e error.dat

#$ -cwd

java -cp NoisyMIL.jar Utils.BagLevelNoiseFilterTest


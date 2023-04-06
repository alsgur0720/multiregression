## Create enviroments
conda env create -f multiregression.yaml

## Quick Run
python test.py --opt options/test/test_stage1.yml

In 'test_stage1.yml' file, there is kernel parameters of aniotropic Gaussian (sig1, sig2, theta).
Testers can modulate the parameters and observe our results.
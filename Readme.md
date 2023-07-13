
## multiregression weights
weights link : https://drive.google.com/file/d/1nO7Td6D2sTU0TqJYPNYeLRxXTVeV9UaS/view?usp=sharing

dir:
'../pretrained/300000_G.pt'

## Create enviroments
conda env create -f multiregression.yaml

## Quick Run
python test.py --opt options/test/test_stage1.yml

In 'test_stage1.yml' file, there is kernel parameters of aniotropic Gaussian (sig1, sig2, theta).
Testers can modulate the parameters and observe our results.

# Attention-based CNN for TYK2 Inhibitor Prediction

## Install

    conda env create -n tyk2 -f environment.yaml
    conda activate tyk2
 
## Process
**Train**

    python single_run.py -k train -d data/tyk2.json
    
**Test**

    python single_run.py -k test -d data/tyk2_test.json -m data/trained_models/model_TEST_BEST

**Visualize**

    python weight_vis.py -d data/tyk2.json -m data/trained_models/model_TEST_BEST
    
## Citaion

    Pham, H.N., & Le, T.H. (2019). Attention-based Multi-Input Deep Learning Architecture for Biological Activity Prediction: An Application in EGFR Inhibitors. ArXiv, abs/1906.05168.
    
    Attention-based Multi-Input Deep Learning Architecture for Biological Activity Prediction: An Application in EGFR Inhibitors. (2019, October 1). IEEE Conference Publication | IEEE Xplore. https://ieeexplore.ieee.org/abstract/document/8919265?casa_token=SM8E8X1L3EwAAAAA:BzcsOQ66CU8qGhlsOkxBgMwJrGPp_IJYDl0edjkCWpdGQwfS4zJbjb1jLt0ZeafA1sXdmdXasQ
    

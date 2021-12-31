# Attention-based CNN for TYK2 Inhibitor Prediction
<img src="https://i.ibb.co/jg5kzd5/egfr-architecture-new.jpg" width="700">

> **Abstract:**  
> Understand the existing egfr inhibitor prediction research process and apply it to tyk2 inhibitor prediction research <Br>  
> Reference: https://github.com/lehgtrung/egfr-att/tree/master/egfr/data
<Br>  
        
## How to Install
    
    conda env create -n tyk2 -f environment.yaml
    conda activate tyk2
<Br>  
    
## Usage
    
+ Train
    ```
    python single_run.py -k train -d data/tyk2.json
    ```
+ Test
    ```    
    python single_run.py -k test -d data/tyk2_test.json -m data/trained_models/model_TEST_BEST
    ```
+ Visualize Attension
    ```
    python weight_vis.py -d data/tyk2.json -m data/trained_models/model_TEST_BEST
    ```
## Citaion

 > Attention-based Multi-Input Deep Learning Architecture for Biological Activity Prediction: An Application in EGFR Inhibitors. (2019, October 1). IEEE Conference Publication | IEEE Xplore. https://ieeexplore.ieee.org/abstract/document/8919265?casa_token=SM8E8X1L3EwAAAAA:BzcsOQ66CU8qGhlsOkxBgMwJrGPp_IJYDl0edjkCWpdGQwfS4zJbjb1jLt0ZeafA1sXdmdXasQ  
    
 > Pham, H.N., & Le, T.H. (2019). Attention-based Multi-Input Deep Learning Architecture for Biological Activity Prediction: An Application in EGFR Inhibitors. ArXiv, abs/1906.05168.
    
    

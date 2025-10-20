# KA4GANC
This repository contains the implementation of the KA4GANC Model, a computational framework for inferring Gene Regulatory Networks (GRNs).

# Dependencies
- Python == 3.8 
- Pytorch == 1.6.0
- scikit-learn==1.0.2
- numpy==1.20.3
- scanpy==1.7.2
- gseapy==0.10.8
- pytorch_geometric

## Usage

__Preparing  for gene expression profiles and  gene-gene adjacent matrix__
   
   KA4GANC integrates gene expression matrix __(N×M)__ with prior gene topology __(N×N)__ to learn vertorized representations with supervision.  


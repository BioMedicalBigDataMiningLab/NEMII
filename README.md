# NEMII
Code and data for “A network embedding-based multiple information integration method for the MiRNA-disease association prediction”.

## Dataset

- data/miRNA_disease.csv is the miRNA_disease association matrix, which contain 4479 associations between 412 miRNAs and 314 diseases.

- data/miRNA_family.csv is the miRNA_family association matrix, which contain 278 miRNA-family associations between 412 miRNAs and 278 families.

- data/disease_similarity.csv is the disease similarity matrix of 314 diseases,which is calculated based on disease mesh descriptors.

## Code

- The directory SDNE contains the implementation of network embedding method Structural Deep Network Embedding.

- NEMII_main.py implements 5-fold CV of NEMII on our dataset.

Please kindly cite the paper if you use the code or the dataset. 

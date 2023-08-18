# MedicalMatch  

The code is for paper: 
## 1.Dataset
Experiments are conducted on three public datasets: **ACDC** , **Synapse** and **ISIC**.
- **ACDC**  
We evaluate our experiments on ACDC dataset under 1\% labeled, 3\% labeled and 10\% labeled, respectively.  
1. More details about the dataset split and implementation details will be released till acceptance.    
2. Refer to this [link](https://github.com/LiheYoung/UniMatch/tree/main/more-scenarios/medical) and download ACDC dataset.

- **ISIC**  
We divided the dataset into  and  images for training and validation, respectively. Then, we validate MedicalMatch under 3\% and 10\% labeled.
We will upload the processed dataset later.  


- **Synapse**  
We divided the dataset into  and  images for training and validation, respectively.  
1. Download from: [])  

## 2.Enviorments
- python 3.7
- pytorch 1.9.0
- torchvision 0.10.0

## 3.Train/Test  
**Train a Semi-Supervised Model**   
For example, we can train a model on ACDC dataset by:
```
python train_MedicalMatch.py
```
```
python test_MedicalMatch.py
```
  
**Note that all of our settings are the same with [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)**  

## 4.Reference
- [SSL4MIS]([https://github.com/Haochen-Wang409/U2PL](https://github.com/HiLab-git/SSL4MIS))
- [UniMatch](https://github.com/LiheYoung/UniMatch/tree/main/more-scenarios/medical)
- [TransUnet](https://github.com/Beckschen/TransUNet)

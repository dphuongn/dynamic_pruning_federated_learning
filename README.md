# Dynamic Pruning in Federated Learning with non-IID data

## Dataset
* FEMNIST
* Cifar-10
* ImageNet
<!-- ___ -->

## Model architectures
* VGG-11
* ResNet-32

## FL algorithm
* FedAvg
* FedNova
* FedProx
* SCAFFOLD

## Dependencies

Language used: 
* Python 3.7
* PyTorch 1.8.0 (cuda 11.1)
* Torchvision 0.9.0


## Usage

Here is one example to run this experiment:

```
nvidia-smi
python experiments.py \
--model=vgg \
--dataset=cifar10 \
--alg=fedavg \
--lr=0.01 \
--batch-size=64 \
--epochs=1 \
--n_parties=2 \
--mu=0.01 \
--rho=0.9 \
--comm_round=1 \
--partition=noniid-labeldir \
--beta=0.1 \
--device='cuda' \
--datadir='./data/' \
--logdir='./logs/'  \
--noise=0 \
--sample=1 \
--init_seed=0 \
--train-flag \
--gate \
--ratio 0.7 \
--log_file_name='vgg_fedavg_yes_test' \
--checkpoint_name='vgg_fednova_yes.pth.tar' \
--dynamic-pruning \
```

## Citation
```
@misc{yu2021dpfl,
    title={Adaptive Dynamic Pruning for Non-IID Federated Learning}, 
    author={Sixing Yu and Phuong Nguyen and Ali Anwar and Ali Jannesari},
    journal={arXiv preprint arXiv:2106.06921}
    year={2021}
}
```
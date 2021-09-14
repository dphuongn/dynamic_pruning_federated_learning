Dynamic Pruning in Federated Learning with non-IID data

How to run the experiments:

nvidia-smi
```
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
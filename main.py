from utilities.evaluate import test_network
from utilities.parameter import get_parameter
from utilities.train import train_network
from utilities.utils import save_network, load_network

args = get_parameter()
network = load_network(args)
if args.train_flag:
    network = train_network(network, args)
    save_network(network, args)
elif args.test_flag:
    test_network(network, args)
# print(network)

import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from sklearn import linear_model
import time

from lsm_weight_definitions import *
from lsm_models import LSM

parser = argparse.ArgumentParser(description='Train an LSM')

parser.add_argument('--time_steps', default=150, type=int, help='number of time steps in model')
parser.add_argument('--batch_size', default=256, type=int, help='batch size for training')
parser.add_argument('--tau_v', default=16.0, type=float, help='membrane time constant')
parser.add_argument('--tau_i', default=16.0, type=float, help='synaptic time constant')
parser.add_argument('--th', default=20, type=float, help='threshold for spiking')
parser.add_argument('--LqWin', default=27, type=float, help='LqWin')
parser.add_argument('--LqWlsm', default=2, type=float, help='LqWlsm')
parser.add_argument('--in_conn_density', default=0.15, type=float, help='in_conn_density')
parser.add_argument('--lam', default=9, type=float, help='d')
parser.add_argument('--inh_fr', default=0.2, type=float, help='inhibitory firing rate')
parser.add_argument('--Nx', default=10, type=int, help='Nx')
parser.add_argument('--Ny', default=10, type=int, help='Ny')
parser.add_argument('--Nz', default=10, type=int, help='Nz')

args = parser.parse_args()

def main():

    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                          transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.time_steps)])

    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)


    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    data, targets = next(iter(trainloader))
    print("data", data.shape)
    flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1))
    print("flat_data", flat_data.shape)

    in_sz = flat_data.shape[-1]

    #Set neuron parameters
    curr_prefac = np.float32(1/args.tau_i)
    alpha = np.float32(np.exp(-1/args.tau_i))
    beta = np.float32(1 - 1/args.tau_v)

    Win, Wlsm = initWeights1(LqWin=args.LqWin, LqWlsm=args.LqWlsm, in_conn_density=args.in_conn_density, in_size=in_sz, lam=args.lam, 
                             inh_fr=args.inh_fr, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz, init_Wlsm=True, W_lsm=None)

    N = Wlsm.shape[0]

    lsm_net = LSM(N, in_sz, np.float32(curr_prefac*Win), np.float32(curr_prefac*Wlsm), alpha=alpha, beta=beta, th=args.th).to(device)
    lsm_net.eval()

    #Run with no_grad for LSM
    with torch.no_grad():
        start_time = time.time()

        # initialize empty tensors
        in_train = torch.empty(0).to(device)
        lsm_out_train = torch.empty(0).to(device)
        lsm_label_train = torch.empty(0).to(device)

        for i, (data, targets) in enumerate(iter(trainloader)):

            if i%25 == 24:
                print("train batches completed: ", i)
            
            data = data.to(device).type(torch.float32)
            targets = targets.to(device)

            flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1)).to(device).type(torch.float32)

            spk_rec = lsm_net(data)
            lsm_out = torch.mean(spk_rec, dim=0)

            print("flat_data", flat_data.type())
            in_train = torch.cat((in_train, torch.mean(flat_data, dim=0)), dim=0)
            lsm_out_train = torch.cat((lsm_out_train, lsm_out), dim=0)
            lsm_label_train = torch.cat((lsm_label_train, targets), dim=0)

            print(i)

        end_time = time.time()


        print("running time of training epoch: ", end_time - start_time, "seconds")


        for i, (data, targets) in enumerate(iter(testloader)):

            if i%25 == 24:
                print("test batches completed: ", i)

            data = data.to(device).type(torch.float32)
            targets = targets.to(device)

            flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1)).to(device).type(torch.float32)

            lsm_net.eval()
            spk_rec = lsm_net(flat_data)
            lsm_out = torch.mean(spk_rec, dim=0)

            print("flat_data", flat_data.type())
            in_test = torch.cat((in_test, torch.mean(flat_data, dim=0)), dim=0)
            lsm_out_test = torch.cat((lsm_out_test, lsm_out), dim=0)
            lsm_label_test = torch.cat((lsm_label_test, targets), dim=0)

    print(lsm_out_train.shape)
    print(lsm_out_test.shape)

    print(in_train.shape)
    print(in_test.shape)

    print("mean in spiking (train) : ", np.mean(in_train))
    print("mean in spiking (test) : ", np.mean(in_test))

    print("mean LSM spiking (train) : ", np.mean(lsm_out_train))
    print("mean LSM spiking (test) : ", np.mean(lsm_out_test))

    print("training linear model:")
    clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6)
    clf.fit(lsm_out_train, lsm_label_train)

    score = clf.score(lsm_out_test, lsm_label_test)
    print("test score = " + str(score))

    # clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6)
    # clf.fit(in_train, lsm_label_train)

    # score = clf.score(in_test, lsm_label_test)
    # print("test score = " + str(score))


if '__main__' == __name__:
    main()
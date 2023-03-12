import pandas as pd
import h5py
from pathlib import Path  
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import dnaDataloader as dna
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lava.lib.dl.slayer as slayer
import os

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        neuron_params = {
                'threshold'     : 0.2,
                'current_decay' : 1,
                'voltage_decay' : 0.1,
                'requires_grad' : True,     
            }
        
        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Dense(neuron_params, 200, 256),
                slayer.block.cuba.Dense(neuron_params, 256, 200),
                ])

    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

class Settings:
    _INPUT_DIRECTORY    = None
    _INPUT_INDEX        = None  
    _EPOCHS             = None    

    def __new__(cls):
            if not hasattr(cls, 'instance'):
                    cls.instance = super(Settings, cls).__new__(cls)
            return cls.instance 
    def save_settings(self):
                dt_string = datetime.now().strftime("%d.%m.%Y_%H-%M-%S-%f")
                try:
                    f = open(f"./snn_logs/snnModel_{dt_string}_saved_args.csv", "w+")
                except FileNotFoundError as e: 
                    print("!!! Missing 'snn_logs' directory\n writing logs to this directory")
                    f = open(f"./snnModel_{dt_string}_saved_args.csv", "w+")
                f.write(f"run at {dt_string}")
                f.close()
        
if __name__ == "__main__":

    DEFAULT_EPOCHS = 20
    
    parser = argparse.ArgumentParser(description='''Runs SNN training \n''', 
                                    formatter_class=RawTextHelpFormatter )
    parser.add_argument('input_directory', help='Directory with input files for training\n',
                        type=str)
    parser.add_argument('input_index', help='index file listing the input->target pairings\n',
                        type=str)
    parser.add_argument('-e', '--epochs', help='''Number of epochs for training \n DEFAULT: {}\n'''.format(DEFAULT_EPOCHS),
                        type=int,
                        default=DEFAULT_EPOCHS)


    args = parser.parse_args()
    settings = Settings()
    ### user defined settings ###
    settings._INPUT_DIRECTORY = args.input_directory
    settings._INPUT_INDEX = args.input_index
    settings._EPOCHS = args.epochs
    settings.save_settings()
    print(f"reading files from  :{Path(settings._INPUT_DIRECTORY)}")
    print(f"reading index file  :{Path(settings._INPUT_INDEX)}")
    print(f"trainging for {settings._EPOCHS} epochs")

    trainingData = dna.expermentDataloader(settings._INPUT_INDEX,settings._INPUT_DIRECTORY)
    train_loader = DataLoader(dataset=trainingData, batch_size=3)

    trained_folder = 'Trained'
    os.makedirs(trained_folder, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {device}")

    net = Network().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
    error = slayer.loss.SpikeTime().to(device)
    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(net, error, optimizer, stats)
    
    for epoch in range(settings._EPOCHS):
        for i, (input, target) in enumerate(train_loader): # training loop
            output = assistant.train(input, target)
            print(f'\r[Epoch {epoch:3d}/{settings._EPOCHS}] {stats}', end='')

        if stats.training.best_loss:
            torch.save(net.state_dict(), trained_folder + '/network.pt')
        stats.update()
        stats.save(trained_folder + '/')
        
    torch.save(net, trained_folder + '/snnModel.pt')

"""
Created by Constantin Philippenko, 6th September 2021.
"""
import unittest

import torch
from torch import optim
from torch.optim import SGD

from src.deeplearning.DLParameters import cast_to_DL
from src.deeplearning.NnModels import LogisticReg
from src.deeplearning.Train import *
from src.machinery.PredefinedParameters import BiQSGD
from src.models.CompressionModel import SQuantization


class TrainTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.nb_devices = 10
        cls.input_size = 4
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = LogisticReg

        #### global model ##########
        cls.global_model = net(input_size=cls.input_size).to(cls.device)

        ############## client models ##############
        cls.client_models = [net(input_size=cls.input_size).to(cls.device) for i in range(cls.nb_devices)]
        for model in cls.client_models:
            model.load_state_dict(cls.global_model.state_dict())

        cls.shapes = [torch.Size([1,4])]

        cls.params = BiQSGD().define(cost_models=None,
                                     n_dimensions=cls.input_size,
                                     stochastic=False,
                                     nb_epoch=10000,
                                     nb_devices=cls.nb_devices,
                                     batch_size=10,
                                     fraction_sampled_workers=1,
                                     up_compression_model=SQuantization(1, norm=2),
                                     down_compression_model=SQuantization(1, norm=2))
        cls.params = cast_to_DL(cls.params, None, None, 0.1, 0, None)

        cls.optimizers = [optim.SGD(model.parameters(), lr=0.1, momentum=0) for model in cls.client_models]

    def test_initialize_gradients_to_zeros(self):
        initialize_gradients_to_zeros(self.global_model, self.shapes)

        index = 0
        for p in self.global_model.parameters():
            assert p.grad.equal(torch.zeros(self.shapes[index])), "Gradient are not equal to zeros"
            index += 1

    def test_server_aggregate_gradients(self):
        initialize_gradients_to_zeros(self.global_model, self.shapes)

        # Initialisation of local gradients
        i = 0
        for model in self.client_models:
            for p in model.parameters():
                p.grad = torch.zeros(self.shapes[0]) + i
            i+=1

        server_aggregate_gradients(self.global_model, self.client_models)
        for p in self.global_model.parameters():
            assert p.grad.equal(torch.zeros_like(p.grad) + 4.5)

    def test_server_compress_gradient(self):
        torch.manual_seed(10)

        # Initialisation of global gradients
        initialize_gradients_to_zeros(self.global_model, self.shapes)
        for p in self.global_model.parameters():
            p.grad.copy_(torch.tensor([[10, 1, 0, 10]]).type(p.grad.dtype))

        server_compress_gradient(self.global_model, self.params)
        for p in self.global_model.parameters():
            assert p.grad.equal(torch.tensor([[14.1774473190, 0, 0, 14.1774473190]]).type(p.grad.dtype)), "Up compression is not correct"

    def test_server_update_model(self):
        torch.manual_seed(10)

        # Initialisation of global model
        with torch.no_grad():
            for p in self.global_model.parameters():
                p.copy_(torch.tensor([[0, 1, 2, 3]]).type(p.dtype))
        # Initialisation of global gradient
        initialize_gradients_to_zeros(self.global_model, self.shapes)
        for p in self.global_model.parameters():
            p.grad = torch.tensor([[10, 0, 0, 10]]).type(p.grad.dtype)

        server_update_model(self.global_model, self.params)
        for p in self.global_model.parameters():
            assert p.equal(torch.tensor([[-1, 1, 2, 2]]).type(p.dtype)), "Update is not correct"

    def test_compress_model_and_combine_WITHOUT_down_memory(self):
        torch.manual_seed(10)
        client_index = 0

        # Initialisation of global model
        with torch.no_grad():
            for p in self.global_model.parameters():
                p.copy_(torch.tensor([[5, 10, 5, 0]]).type(p.dtype))

        compress_model_and_combine_with_down_memory(self.global_model, self.client_models[client_index],
                                                    self.optimizers[client_index], self.params, self.device)

        # Cheking value of client model
        for p in self.client_models[client_index].parameters():
            assert p.equal(torch.tensor([[0, 12.2474489212, 0, 0]]).type(p.dtype)), "Down compression is not correct"

        # Cheking that the central model is unchanged.
        for p in self.global_model.parameters():
            assert p.equal(torch.tensor([[5, 10, 5, 0]]).type(p.dtype)), "Global model should not have been updated"

    def test_compress_model_and_combine_WITH_down_memory(self):
        self.params.use_down_memory = True
        torch.manual_seed(10)
        client_index = 0
        model = self.client_models[client_index]

        # Initialisation of down memory
        for p in model.parameters():
            param_state = self.optimizers[client_index].state[p]
            param_state[down_memory_name] = torch.tensor([[5, 0, -10, -5]]).type(p.dtype)

        # Initialisation of global model
        with torch.no_grad():
            for p in self.global_model.parameters():
                p.copy_(torch.tensor([[5, 10, 5, 0]]).type(p.dtype))

        compress_model_and_combine_with_down_memory(self.global_model, self.client_models[client_index],
                                                    self.optimizers[client_index], self.params, self.device)

        # Cheking value of client model
        for p in self.client_models[client_index].parameters():
            assert p.equal(torch.tensor([[5, 18.7082862854, 8.7082862854, -5]]).type(p.dtype)), "Down compression is not correct"

        # Cheking that the central model is unchanged.
        for p in self.global_model.parameters():
            assert p.equal(torch.tensor([[5, 10, 5, 0]]).type(p.dtype)), "Global model should not have been updated"

        # Checking that the memory learning rate has been updated (alpha = 0.16666666666666)
        for p in model.parameters():
            param_state = self.optimizers[client_index].state[p]
            assert param_state[down_memory_name].equal(torch.tensor([[5, 3.1180477142332084, -6.881952285766792, -5]]).type(p.dtype)), \
                "The downmemory is not correct"

    def test_server_send_models_to_clients(self):
        # Initialisation of global model
        with torch.no_grad():
            for p in self.global_model.parameters():
                p.copy_(torch.tensor([[0, 1, 2, 3]]).type(p.dtype))

        # Initialisation of clients model to zeros
        with torch.no_grad():
            for model in self.client_models:
                for p in model.parameters():
                    p.copy_(torch.zeros_like(p))

        server_send_models_to_clients(self.global_model, self.client_models)
        for model in self.client_models:
            for p in model.parameters():
                assert p.equal(torch.tensor([[0, 1, 2, 3]]).type(p.dtype))

    def server_compress_model_and_send_to_clients(self):
        pass

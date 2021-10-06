"""
Created by Constantin Philippenko, 6th October 2021.
"""
from src.models.CostModel import RMSEModel, LogisticModel

batch_sizes = {"a9a": 50, "abalone": 50, "covtype": 10000, "gisette": 50, "madelon": 16, "mushroom": 4, "quantum": 400,
               "phishing": 50, "superconduct": 50, "w8a": 12,
               "synth_logistic": 1, "synth_linear_noised": 1, "synth_linear_nonoised": 1}

models =  {"a9a": LogisticModel, "abalone": RMSEModel, "covtype": RMSEModel, "gisette": RMSEModel,
           "madelon": LogisticModel, "mushroom": LogisticModel, "quantum": LogisticModel, "phishing": LogisticModel,
           "superconduct": RMSEModel, "w8a": LogisticModel,
           "synth_logistic": LogisticModel, "synth_linear_noised": RMSEModel, "synth_linear_nonoised": RMSEModel}
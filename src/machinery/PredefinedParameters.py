"""
Created by Philippenko, 7th July 2020.
"""

from src.machinery.GradientDescent import ArtemisDescent, SGD_Descent, DianaDescent, AGradientDescent
from src.machinery.Parameters import Parameters
from src.models.CompressionModel import CompressionModel, RandomSparsification, SQuantization
from src.utils.Constants import NB_EPOCH


class PredefinedParameters:
    """Abstract class to predefine (no customizable) parameters required by a given type of algorithms (e.g Artemis, QSGD ...)

    Keep high degree of customization.
    """

    def name(self) -> str:
        """Name of the predefined parameters.
        """
        return "empty"

    def type_FL(self) -> AGradientDescent:
        return ArtemisDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1.,
               use_averaging=False, stochastic=True, streaming=False, batch_size=1) -> Parameters:
        """Define parameters to be used during the descent.

        Args:
            n_dimensions: dimensions of the problem.
            nb_devices: number of device in the federated network.
            quantization_param: parameter of quantization.
            step_formula: lambda formul to compute the step size at each iteration.
            momentum: momentum coefficient.
            nb_epoch: number of epoch for the run.
            use_averaging: true if using Polyak-Rupper Averaging.
            cost_models: cost model of the problem (e.g least-square, logistic ...).
            stochastic: true if running stochastic descent.

        Returns:
            Build parameters.
        """
        pass


class VanillaSGD(PredefinedParameters):
    """Predefine parameters to run SGD algorithm in a federated settings.
    """

    def name(self):
        return "SGD"

    def type_FL(self) -> AGradientDescent:
        return SGD_Descent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          compression_model=compression_model,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          bidirectional=False,
                          cost_models=cost_models,
                          use_averaging=use_averaging,
                          use_memory=False
                          )


class Qsgd(VanillaSGD):
    """Predefine parameters to run QSGD algorithm.
    """

    def name(self) -> str:
        return r"QSGD"

    def type_FL(self) -> AGradientDescent:
        return DianaDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        return params


class Diana(VanillaSGD):
    """Predefine parameters to run Diana algorithm.
    """

    def name(self) -> str:
        return "Diana"

    def type_FL(self):
        return DianaDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.use_memory = True
        return params


class BiQSGD(Qsgd):
    """Predefine parameters to run BiQSGD algorithm.
    """

    def name(self) -> str:
        return "BiQSGD"

    def type_FL(self):
        return ArtemisDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.bidirectional = True
        return params

class Artemis(Diana):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "Artemis"

    def type_FL(self):
        return ArtemisDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.bidirectional = True
        return params

class RArtemis(Artemis):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "RArtemis"

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
               step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
               stochastic, streaming, batch_size)
        params.randomized = True
        return params

class RArtemisEF(RArtemis):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "RArtemisEF"

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
               step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
               stochastic, streaming, batch_size)
        params.error_feedback = True
        return params

KIND_COMPRESSION = [VanillaSGD(),
                    Qsgd(),
                    Diana(),
                    BiQSGD(),
                    Artemis()
                    ]

class Try1(PredefinedParameters):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "Mem-Feed"

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          compression_model=compression_model,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          cost_models=cost_models,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          use_double_memory=False,
                          compress_gradients=True,
                          use_memory=True,
                          error_feedback=True,
                          randomized=False
                          )

class RArtemis(PredefinedParameters):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "Mem-Rand"

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          compression_model=compression_model,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          cost_models=cost_models,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          use_double_memory=False,
                          compress_gradients=True,
                          use_memory=True,
                          error_feedback=False,
                          randomized=True
                          )

class Try3(PredefinedParameters):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "Feed-Rand"

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          compression_model=compression_model,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          cost_models=cost_models,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          use_double_memory=False,
                          compress_gradients=True,
                          use_memory=False,
                          error_feedback=True,
                          randomized=True
                          )

class Try4(PredefinedParameters):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "Mem-Feed-Rand"

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          compression_model=compression_model,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          cost_models=cost_models,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          use_double_memory=False,
                          compress_gradients=True,
                          use_memory=True,
                          error_feedback=True,
                          randomized=True
                          )

class BiQSGD_b(BiQSGD):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "BiQSGD_b"

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        parameters = super().define(cost_models, n_dimensions, nb_devices, RandomSparsification(10, n_dimensions, biased=True),
               step_formula, nb_epoch,  fraction_sampled_workers, use_averaging,
               stochastic, streaming, batch_size)
        return parameters

class BiQSGD_Feed_b(BiQSGD):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "BiQSGD-Feed_b"

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        parameters = super().define(cost_models, n_dimensions, nb_devices, RandomSparsification(10, n_dimensions, biased=True),
               step_formula, nb_epoch,  fraction_sampled_workers, use_averaging,
               stochastic, streaming, batch_size)
        parameters.error_feedback = True
        return parameters

class BiQSGD_unb(BiQSGD):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "BiQSGD-unb."

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        parameters = super().define(cost_models, n_dimensions, nb_devices, RandomSparsification(10, n_dimensions, biased=False),
               step_formula, nb_epoch,  fraction_sampled_workers, use_averaging,
               stochastic, streaming, batch_size)
        return parameters

class BiQSGD_Feed_unb(BiQSGD):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "BiQSGD-Feed-unb."

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        parameters = super().define(cost_models, n_dimensions, nb_devices, RandomSparsification(10, n_dimensions, biased=False),
               step_formula, nb_epoch,  fraction_sampled_workers, use_averaging,
               stochastic, streaming, batch_size)
        parameters.error_feedback = True
        return parameters


KIND_COMPRESSION_RANDOMIZED = [Artemis(), Try1()]#, Try2(), Try3(), Try4(), BiQSGD_Feed()]

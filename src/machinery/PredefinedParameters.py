"""
Created by Philippenko, 7th July 2020.
"""

from src.machinery.GradientDescent import ArtemisDescent, FL_VanillaSGD, DianaDescent
from src.machinery.Parameters import Parameters
from src.models.CostModel import ACostModel, RMSEModel
from src.utils.Constants import NB_EPOCH


class PredefinedParameters():
    """Abstract class to predefine (no customizable) parameters required by a given type of algorithms (e.g Artemis, QSGD ...)

    Keep high degree of customization.
    """

    def name(self) -> str:
        """Name of the predefined parameters.
        """
        return "empty"


    def type_FL(self):
        return ArtemisDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, quantization_param: int,
               step_formula=None, momentum: float = 0,
               nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1.,
               use_averaging=False, stochastic=True, streaming=False, batch_size=1):
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


class SGDWithoutCompression(PredefinedParameters):
    """Predefine parameters to run SGD algorithm in a federated settings.
    """

    def name(self):
        return "SGD"

    def type_FL(self):
        return FL_VanillaSGD

    def define(self, cost_models, n_dimensions: int, nb_devices: int, quantization_param: int = 0,
               step_formula=None, momentum: float = 0, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          quantization_param=0,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          bidirectional=False,
                          cost_models=cost_models,
                          use_averaging=use_averaging
                          )


class Qsgd(PredefinedParameters):
    """Predefine parameters to run QSGD algorithm.
    """

    def name(self) -> str:
        return r"QSGD"

    def type_FL(self):
        return DianaDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, quantization_param: int = 0,
               step_formula=None, momentum: float = 0, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          learning_rate=0,
                          verbose=False,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          cost_models=cost_models,
                          use_averaging=use_averaging,
                          bidirectional=False
                          )


class Diana(PredefinedParameters):
    """Predefine parameters to run Diana algorithm.
    """

    def type_FL(self):
        return DianaDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, quantization_param: int = 0,
               step_formula=None, momentum: float = 0, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          bidirectional=False,
                          cost_models=cost_models,
                          use_averaging=use_averaging
                          )

    def name(self) -> str:
        return "Diana"


class BiQSGD(PredefinedParameters):
    """Predefine parameters to run Bi-QSGD algorithm.
    """

    def name(self) -> str:
        return "BiQSGD"

    def type_FL(self):
        return ArtemisDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, quantization_param: int = 0,
               step_formula=None, momentum: float = 0, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          quantization_param=1,
                          learning_rate=0,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          cost_models=cost_models,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          double_use_memory=False,
                          compress_gradients=True
                          )


class Artemis(PredefinedParameters):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "Artemis"

    def define(self, cost_models, n_dimensions: int, nb_devices: int, quantization_param: int = 0,
               step_formula=None, momentum: float = 0, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          cost_models=cost_models,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          double_use_memory=False,
                          compress_gradients=True
                          )


class DoreVariant(PredefinedParameters):
    """Predefine parameters to run a variant of algorithm.
    This variant use
    """

    def name(self) -> str:
        return "Dore"

    def define(self, cost_models, n_dimensions: int, nb_devices: int, quantization_param: int = 0,
               step_formula=None, momentum: float = 0, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          cost_models=cost_models,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          double_use_memory=True,
                          compress_gradients=True
                          )


class SGDDoubleModelCompressionWithoutMem(PredefinedParameters):

    def define(self, cost_models, n_dimensions: int, nb_devices: int, quantization_param: int = 0,
               step_formula=None, momentum: float = 0, nb_epoch: int = NB_EPOCH,
               fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          cost_models=cost_models,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          double_use_memory=False,
                          compress_gradients=False
                          )


class SGDDoubleModelCompressionWithMem(PredefinedParameters):

    def define(self, cost_models, n_dimensions: int, nb_devices: int, quantization_param: int = 0,
               step_formula=None, momentum: float = 0, nb_epoch: int = NB_EPOCH,
               fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          fraction_sampled_workers=fraction_sampled_workers,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          streaming=streaming,
                          batch_size=batch_size,
                          cost_models=cost_models,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          double_use_memory=True,
                          compress_gradients=False
                          )


KIND_COMPRESSION = [SGDWithoutCompression(),
                    Qsgd(),
                    Diana(),
                    BiQSGD(),
                    Artemis()
                    ]

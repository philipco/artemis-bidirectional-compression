"""
Created by Philippenko, 7th July 2020.
"""

from src.machinery.GradientDescent import ArtemisDescent, SGD_Descent, DianaDescent, AGradientDescent, SympaDescent, \
    DownCompressModelDescent, FedAvgDescent
from src.machinery.Parameters import Parameters
from src.models.CompressionModel import *
from src.utils.Constants import NB_EPOCH


def build_compression_operator(biased, n_dimensions):
    if biased:
        return RandomSparsification(11, n_dimensions, biased=biased)
    else:
        return RandomSparsification(11, n_dimensions, biased=biased)

level = 10
level_quantiz=4

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

class Sympa(Artemis):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "Sympa"

    def type_FL(self):
        return SympaDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
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
        params.down_error_feedback = True
        return params

KIND_COMPRESSION = [VanillaSGD(),
                    Qsgd(),
                    Diana(),
                    BiQSGD(),
                    Artemis()
                    ]


############################################ - Error Feedback - ############################################

class BiQSGD_EF(BiQSGD):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "BiQSGD_EF"

    def type_FL(self):
        return ArtemisDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.down_error_feedback = True
        return params

class DoubleSqueeze(BiQSGD):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "DblSqz"

    def type_FL(self):
        return ArtemisDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.down_error_feedback = True
        params.up_error_feedback = True
        return params

class ArtemisEF(Artemis):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "Dore"

    def type_FL(self):
        return ArtemisDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.down_error_feedback = True
        return params

KIND_COMPRESSION_EF = [VanillaSGD(),
                       BiQSGD(),
                       BiQSGD_EF(),
                       Artemis(),
                       ArtemisEF()
                    ]

KIND_COMPRESSION_RAND = [VanillaSGD(),
                         BiQSGD(),
                         Artemis(),
                         ArtemisEF(),
                         RArtemis(),
                         RArtemisEF(),
                         ]

ARTEMIS_LIKE_ALGO = [VanillaSGD(),
                     BiQSGD(),
                     BiQSGD_EF(),
                     Artemis(),
                     ArtemisEF(),
                     RArtemis(),
                     RArtemisEF()]

################################### - Operators - ###################################


class Topk(PredefinedParameters):

    def name(self) -> str:
        return "Topk{0}".format(self.super_class.name())

    def __init__(self, super_class: PredefinedParameters) -> None:
        super().__init__()
        self.super_class = super_class
        self.biased = True

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        parameters = self.super_class.define(cost_models, n_dimensions, nb_devices,
                                    compression_model, step_formula, nb_epoch,
                                    fraction_sampled_workers, use_averaging,stochastic, streaming, batch_size)
        parameters.compression_model = TopKSparsification(level, n_dimensions)
        return parameters


class RandkBiased(PredefinedParameters):

    def name(self) -> str:
        return "RandkBsd{0}".format(self.super_class.name())

    def __init__(self, super_class: PredefinedParameters) -> None:
        super().__init__()
        self.super_class = super_class
        self.biased = True

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        parameters = self.super_class.define(cost_models, n_dimensions, nb_devices,
                                    compression_model, step_formula, nb_epoch,
                                    fraction_sampled_workers, use_averaging,stochastic, streaming, batch_size)
        parameters.compression_model = RandomSparsification(level, n_dimensions, biased=False)
        return parameters


class Randk(PredefinedParameters):

    def name(self) -> str:
        return "Randk{0}".format(self.super_class.name())

    def __init__(self, super_class: PredefinedParameters) -> None:
        super().__init__()
        self.super_class = super_class
        self.biased = False

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH,  fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        parameters = self.super_class.define(cost_models, n_dimensions, nb_devices,
                                    compression_model, step_formula, nb_epoch,
                                    fraction_sampled_workers, use_averaging,stochastic, streaming, batch_size)
        parameters.compression_model = RandomSparsification(level, n_dimensions, biased=False)
        return parameters


class Quantiz(PredefinedParameters):

    def name(self) -> str:
        return "Qtzd{0}".format(self.super_class.name())

    def __init__(self, super_class: PredefinedParameters) -> None:
        super().__init__()
        self.super_class = super_class
        self.biased = False

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1):
        parameters = self.super_class.define(cost_models, n_dimensions, nb_devices,
                                             compression_model, step_formula, nb_epoch,
                                             fraction_sampled_workers, use_averaging, stochastic, streaming, batch_size)
        parameters.compression_model = SQuantization(level_quantiz, n_dimensions)
        return parameters


RAND_K = [VanillaSGD(),
          Randk(BiQSGD()),
          Randk(BiQSGD_EF()),
          Randk(Artemis()),
          Randk(ArtemisEF()),
          Randk(RArtemis()),
          Randk(RArtemisEF())
          ]

RAND_K_BIASED = [VanillaSGD(),
          RandkBiased(BiQSGD()),
          RandkBiased(BiQSGD_EF()),
          RandkBiased(Artemis()),
          RandkBiased(ArtemisEF()),
          RandkBiased(RArtemis()),
          RandkBiased(RArtemisEF())
          ]

TOP_K = [VanillaSGD(),
          Topk(BiQSGD()),
          Topk(BiQSGD_EF()),
          Topk(Artemis()),
          Topk(ArtemisEF()),
          Topk(RArtemis()),
          Topk(RArtemisEF())
          ]

QUANTIZATION = [VanillaSGD(),
          Quantiz(BiQSGD()),
          Quantiz(BiQSGD_EF()),
          Quantiz(Artemis()),
          Quantiz(ArtemisEF()),
          Quantiz(RArtemis()),
          Quantiz(RArtemisEF())
          ]

################################################## Compress model ##################################################


class ModelCompr(Artemis):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "ModelCompr"

    def type_FL(self):
        return DownCompressModelDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        return params

class ModelComprMem(ModelCompr):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "ModelComprMem"

    def type_FL(self):
        return DownCompressModelDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.double_use_memory = True
        return params

class RandMCM(ModelComprMem):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "RandMCM"

    def type_FL(self):
        return DownCompressModelDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.randomized = True
        return params

class ModelComprEF(ModelCompr):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "ModelComprEF"

    def type_FL(self):
        return DownCompressModelDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.down_error_feedback = True
        return params

class RModelComprEF(ModelComprEF):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "RModelComprEF"

    def type_FL(self):
        return DownCompressModelDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.randomized = True
        return params


class RModelComprMem(ModelComprMem):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "RModelComprMem"

    def type_FL(self):
        return DownCompressModelDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.randomized = True
        return params

MODEL_COMPRESSION = [VanillaSGD(), Artemis(), ArtemisEF(), ModelCompr(), ModelComprMem(), ModelComprEF(),
                     RModelComprEF(), Sympa()]


################################################## Federated Learning ##################################################

class FedAvg(VanillaSGD):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "FedAvg"

    def type_FL(self):
        return FedAvgDescent

    def define(self, cost_models, n_dimensions: int, nb_devices: int, compression_model: CompressionModel,
               step_formula=None, nb_epoch: int = NB_EPOCH, fraction_sampled_workers: int = 1., use_averaging=False,
               stochastic=True, streaming=False, batch_size=1) -> Parameters:
        params = super().define(cost_models, n_dimensions, nb_devices, compression_model,
                                step_formula, nb_epoch, fraction_sampled_workers, use_averaging,
                                stochastic, streaming, batch_size)
        params.stochastic = True
        params.nb_local_update = 5
        return params

FL_ALGOS = [VanillaSGD(), FedAvg(), DoubleSqueeze(), ArtemisEF(), Artemis()]

ALGO_EF = [Qsgd(), Diana(), Artemis(), ArtemisEF(), DoubleSqueeze()]

STUDIED_ALGO = [VanillaSGD(), FedAvg(), ArtemisEF(), DoubleSqueeze(), Artemis(),
                RArtemis(), RArtemisEF(), Sympa(), ModelComprMem()]

GAME_OF_THRONES = [VanillaSGD(), Artemis(), RArtemis(), ModelCompr(), ModelComprMem(), RandMCM()]
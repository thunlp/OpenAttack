from .UtilClass import LayerNorm, Bottle, BottleLinear, \
    BottleLayerNorm, BottleSoftmax, Elementwise
from .Gate import ContextGateFactory
from .GlobalAttention import GlobalAttention
from .ConvMultiStepAttention import ConvMultiStepAttention
from .ImageEncoder import ImageEncoder
from .CopyGenerator import CopyGenerator, CopyGeneratorLossCompute
from .StructuredAttention import MatrixTree
from .Transformer import TransformerEncoder, TransformerDecoder
from .Conv2Conv import CNNEncoder, CNNDecoder
from .MultiHeadedAttn import MultiHeadedAttention
from .StackedRNN import StackedLSTM, StackedGRU
from .Embeddings import Embeddings
from .WeightNorm import WeightNormConv2d

from .SRU import check_sru_requirement
can_use_sru = check_sru_requirement()
if can_use_sru:
    from .SRU import SRU


# For flake8 compatibility.
__all__ = [GlobalAttention, ImageEncoder, CopyGenerator, MultiHeadedAttention,
           LayerNorm, Bottle, BottleLinear, BottleLayerNorm, BottleSoftmax,
           TransformerEncoder, TransformerDecoder, Embeddings, Elementwise,
           MatrixTree, WeightNormConv2d, ConvMultiStepAttention,
           CNNEncoder, CNNDecoder, StackedLSTM, StackedGRU, ContextGateFactory,
           CopyGeneratorLossCompute]

if can_use_sru:
    __all__.extend([SRU, check_sru_requirement])

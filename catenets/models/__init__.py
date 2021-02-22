from catenets.models.twostep_nets import TwoStepNet
from catenets.models.disentangled_nets import SNet3
from catenets.models.snet import SNet
from catenets.models.representation_nets import SNet1, SNet2
from catenets.models.tnet import TNet

SNET1_NAME = 'SNet1'
T_NAME = 'TNet'
SNET2_NAME = 'SNet2'
TWOSTEP_NAME = 'TwoStepNet'
SNET3_NAME = 'SNet3'
SNET_NAME = 'SNet'

ALL_MODELS = [T_NAME, SNET1_NAME, SNET2_NAME, SNET3_NAME, SNET_NAME,
                      TWOSTEP_NAME]
MODEL_DICT = {T_NAME: TNet, SNET1_NAME: SNet1, SNET2_NAME: SNet2, SNET3_NAME: SNet3,
              SNET_NAME: SNet, TWOSTEP_NAME: TwoStepNet}

__all__ = [T_NAME, SNET1_NAME, SNET2_NAME, SNET3_NAME, SNET_NAME, TWOSTEP_NAME]


def get_catenet(name):
    if name not in ALL_MODELS:
        raise ValueError('Model name should be in '
                         'catenets.models.ALL_MODELS.'
                         ' You passed {}'.format(name))
    return MODEL_DICT[name]

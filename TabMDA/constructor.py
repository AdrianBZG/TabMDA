import copy
import logging

from TabMDA.model import TabMDA
from utils import get_available_device
from tabpfn import TabPFNEncoder

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')

VALID_TABMDA_ENCODERS = ['TabPFN']


def construct_tabmda_model(encoder_name='TabPFN', classifier=None, freeze_encoder=True, device=get_available_device()):
    """
    Function to construct a TabMDA model with a given encoder and classifier.
    """
    if encoder_name not in VALID_TABMDA_ENCODERS:
        raise ValueError(f'Unsupported encoder {encoder_name}. Valid are: {VALID_TABMDA_ENCODERS}')

    if encoder_name == 'TabPFN':
        tabpfn = TabPFNEncoder(
            device=device,

            # An 'ensemble' is a permutation of the columns and of the labels. 
            #   For classification, it performs better to use more than one ensemble.
            #   For TabMDA, having more than one ensemble would introduce more stochasticity in the embeddings.
            N_ensemble_configurations=1,
            only_inference=True,
            no_grad=False,
            no_preprocess_mode=True
        )

        # Remove unnecessary modules to make model lighter
        del tabpfn.model[2].decoder
        del tabpfn.model[2].criterion

        # Unpack the tuple to keep only the encoder
        tabpfn_encoder = copy.deepcopy(tabpfn.model[2])
        del tabpfn.model

        return TabMDA(encoder=(tabpfn, tabpfn_encoder),
                      classifier=classifier,
                      freeze_encoder=freeze_encoder,
                      device=device)

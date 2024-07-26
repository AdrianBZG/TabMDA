import logging
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np

from utils import freeze_model, unfreeze_model, stack_embeddings_to_dataset, to_numpy, smote_augmentation_classwise
from tabpfn.scripts.transformer_prediction_interface import TabPFNEncoder

TABPFN_HIDDEN_DIM = 512

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


class TabMDA(nn.Module):
    def __init__(self, encoder, classifier, freeze_encoder=True, device='cpu'):
        """
        General module which performs (i) encoding and (ii) classification on the encoded data.

        Parameters:
        - encoder: A model that implements .encode() (e.g., TabPFNEncoder)
        - classifier: A model that implements .forward()
        """
        super().__init__()

        if not isinstance(encoder[0], TabPFNEncoder):
            raise ValueError(f'Unsupported encoder type {type(self.encoder_class)}')

        if classifier and not isinstance(classifier, nn.Module):
            raise ValueError(f'Unsupported classifier type {type(classifier)}. '
                             f'It has to be a subclass of torch.nn.Module')

        self.encoder_class = encoder[0]   # This is just the TabPFNEncoder class with the model and some methods. It's not the actual model
        self.encoder_module = encoder[1]  # Extract the actual nn.Module with the TabPFN model so that we can optimize it

        self.classifier = classifier
        
        self.freeze_encoder = freeze_encoder

        if not self.freeze_encoder:
            unfreeze_model(self.encoder_module)
        else:
            freeze_model(self.encoder_module)
            logging.info(f'[Model] Freezing the encoder')

        # === Move the model to the device===
        self.device = device
        self.to(self.device)

    def forward(self, x):
        return self.predict(x)

    def classify(self, x, as_probabilities=False):
        return self.classifier(x, as_probabilities=as_probabilities)

    def predict(self, x, as_probabilities=False):
        raise NotImplementedError("Not implemented. It must take the context as input. Use the `encode` + `classify` methods instead.")

    def encode(self, X_to_encode, X_context, y_context):
        """
        TabPFN expects the following arguments:
        - X_to_encode, X_context, y_context
        """
        return self.encoder_class.encode(X_to_encode=X_to_encode, 
                                         X_context=X_context, 
                                         y_context=y_context, 
                                         model=self.encoder_module)

    def encode_batch(self, batch, context_subsetting_params=None, smote_params=None):
        """
        Returns the encoded batch of size (B x num_contexts, embedding_size) 
            and the corresponding labels (B x num_contexts)
        """
        x, y = batch["x"], batch["y"]
        x_context = batch["x_context"][0]
        y_context = batch["y_context"][0]

        if context_subsetting_params and context_subsetting_params["num_contexts"] > 1:
            if isinstance(context_subsetting_params, dict):
                if all(key in context_subsetting_params.keys() for key in ["num_contexts", "context_size"]):
                    # ==== TrivialAugment style context subsetting ====
                    if context_subsetting_params["context_size"] == 0:
                        # Sample one context size for each of the contexts
                        context_sizes = [random.uniform(0.5, 0.99) for _ in range(context_subsetting_params["num_contexts"])]
                        x_batch_enc = torch.zeros((x.shape[0],
                                                   context_subsetting_params["num_contexts"],
                                                   TABPFN_HIDDEN_DIM))

                        for context_idx, context_size in enumerate(context_sizes):
                            x_context_enc = self.context_subsetting(X_train=x_context,
                                                                    y_train=y_context,
                                                                    num_contexts=1,
                                                                    context_size=context_size,
                                                                    X_to_encode=x)

                            x_batch_enc[:, context_idx] = x_context_enc[:, 0, :]

                        x_batch_enc, y_batch = stack_embeddings_to_dataset(x_batch_enc, batch["y"])
                    else:
                        x_batch_enc = self.context_subsetting(X_train=x_context,
                                                              y_train=y_context,
                                                              num_contexts=context_subsetting_params["num_contexts"],
                                                              context_size=context_subsetting_params["context_size"],
                                                              X_to_encode=x)
                        x_batch_enc, y_batch = stack_embeddings_to_dataset(x_batch_enc, batch["y"])

                    if smote_params:
                        if all(key in smote_params.keys() for key in ["k", "rounds"]):
                            x_batch_enc, y_batch = smote_augmentation_classwise(x_batch_enc, y_batch,
                                                                                k=smote_params["k"],
                                                                                rounds=smote_params["rounds"])
                        else:
                            raise ValueError(f'Unsupported SMOTE arguments {smote_params}')

                    return x_batch_enc, y_batch
                else:
                    raise ValueError(f'Unsupported context subsetting arguments {context_subsetting_params}')
            else:
                raise ValueError(f'Unsupported context subsetting arguments of type {type(context_subsetting_params)}')

        else:
            x_batch_enc = self.encode(X_to_encode=x, 
                                      X_context=x_context, 
                                      y_context=y_context)

            x_batch_enc, y_batch = stack_embeddings_to_dataset(x_batch_enc, batch["y"])

            if smote_params:
                if all(key in smote_params.keys() for key in ["k", "rounds"]):
                    x_batch_enc, y_batch = smote_augmentation_classwise(x_batch_enc, y_batch,
                                                                        k=smote_params["k"],
                                                                        rounds=smote_params["rounds"])
                else:
                    raise ValueError(f'Unsupported SMOTE arguments {smote_params}')

            return x_batch_enc, y_batch

    def context_subsetting(self,
        X_train,                # (np.ndarray) data to be encoded
        y_train,                # (np.ndarray) labels of the data to be encoded
        X_to_encode,            # data to be encoded: (one row), (list of rows), or (batch of B x num_features)
        num_contexts,           # (int) number of contexts to generate
        context_size = None,    # If int, size of each context. If float, proportion of the data to use as context. If None, use the maximum context size.
        seed = 42               # If not None, set the seed for reproducibility
    ):
        """
        Encode X by separately fitting the encoder on `num_contexts` random contexts of size `context_size` from `context[0]` as the context.

        Returns:
        - a tensor with shape (num_contexts, embedding_size) containing `num_contexts` embeddings of X,
        each using a different context
        """
        if context_size is None:
            context_size = len(X_train)
        elif context_size > 1:
            if context_size > len(X_train):
                raise ValueError(f'The context size ({context_size}) can not be bigger than the available '
                                 f'context size ({len(X_train)}).')

            context_size = int(context_size)
        else:  # A proportion of the data
            context_size = int(context_size * len(X_train))

        # === If X is one row, then convert to list
        if (isinstance(X_to_encode, np.ndarray) or isinstance(X_to_encode, torch.Tensor)) and len(X_to_encode.shape) == 1:
            X_to_encode = [X_to_encode]

        if isinstance(y_train, torch.Tensor):
            num_classes = len(torch.unique(y_train))
        else:
            num_classes = len(np.unique(y_train))

        # List of seeds used for sampling each context
        seeds_for_contexts = [seed + i for i in range(num_contexts)]

        # Generate an array of indices from 0 to the length of your dataset
        indices = np.arange(len(X_train))

        # === Encode the data using the random subcontexts ===
        output = []
        for i, _ in enumerate(range(num_contexts)):
            # === Sample a random context (prompt) (stratified sampling) ===
            if context_size == len(X_train):
                X_subcontext, y_subcontext = X_train, y_train
            else:
                indices_train, indices_test, _, _ = train_test_split(
                    indices, indices,
                    train_size=min(context_size, len(X_train) - num_classes),
                    random_state=seeds_for_contexts[i],
                    stratify=to_numpy(y_train)
                )

                X_subcontext = X_train[indices_train]
                y_subcontext = y_train[indices_train]

            # === Encode the data using the random context ===
            X_subcontext_encoded = self.encode(X_context=X_subcontext,
                                               X_to_encode=X_to_encode,
                                               y_context=y_subcontext)

            output.append(X_subcontext_encoded)

        output = torch.cat(output, 1)
        return output

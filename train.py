"""
Script for running training with TabMDA.
"""
import argparse

import wandb, logging, json

import numpy as np
import torch
import os

from TabMDA.constructor import construct_tabmda_model

from dataset.datasets import get_tabular_dataset, ManualStopError
from dataset.pytorch_dataset import PytorchDataset

from models.scikit_classifiers import create_scikit_classifier

from utils import GLOBAL_SEED, set_seed, get_available_device, to_numpy
from sklearn.metrics import balanced_accuracy_score, accuracy_score

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


NON_DIFFERENTIABLE_CLASSIFIERS = ['TabPFN', 'LogReg', 'KNN', 'DecisionTree', 'RandomForest', 'XGBoost']



if __name__ == "__main__":
    set_seed(GLOBAL_SEED)
    device = get_available_device()
    
    # ===== PARSE ARGUMENTS =====
    parser = argparse.ArgumentParser()

    # ----- Dataset -----
    parser.add_argument('--dataset',
                        default="vehicle",
                        choices=['qsar-biodeg', 'vehicle', 'texture', 'steel-plates-fault', 'MiceProtein', 'mfeat-fourier'],
                        required=True,
                        help='Name of the dataset to use')
    parser.add_argument('--repeat_id',
                        type=int,
                        required=True,
                        help='ID of the repeat to use (between 0, 1, .., 9)')
    parser.add_argument('--num_real_samples',
                        default=20,
                        type=int,
                        required=True,
                        choices=[20, 50, 100, 200, 500, -1],
                        help='Number of real samples per class to use. If -1, use all samples.')

    # ----- Model -----
    parser.add_argument('--augmentor_model',
                        default='none',
                        choices=[
                            "none",                 # Use the real samples directly (no augmentor model)
                            "tabmda_encoder",        # Use TabMDA either frozen, or fine-tuned without a downstream classifier
                        ],
                        type=str,
                        help='Name of the model to use')
    parser.add_argument('--when_to_augment_data',
                        default='once_at_initialisation',
                        choices=['once_at_initialisation', 'stochastic_every_batch'],
                        type=str,
                        help='When to encode the data using TabMDA. If "once_at_initialisation", the data is encoded once at the beginning.\
                              If "stochastic_every_batch", the data is encoded every mini-batch.')


    # ----- TabMDA Encoder -----
    # ==== The `freeze_encoder` is set automatically based on the augmentor_model. Specify only if you want to override the default behavior ====
    parser.set_defaults(freeze_encoder=None)
    parser.add_argument('--freeze_encoder',
                        action='store_true',
                        dest='freeze_encoder')
    parser.add_argument('--unfreeze_encoder',
                        action='store_false',
                        dest='freeze_encoder')


    # ----- CONTEXT SUBSETTING -----
    parser.add_argument('--context_size',       # It's shared across train/val/test
                        default=0.5,
                        type=float,
                        help='If int, the number of samples in the context.\
                              If float, the proportion of the training set to use as context.\
                              If 0, it will be randomly sampled in each batch between 0.5 and 0.9.')

    parser.add_argument('--num_contexts',       # For train
                        default=20,
                        type=int,
                        help='Number of contexts for train')

    # ----- Test-time agumetnation -----
    parser.add_argument('--num_contexts_val',   # For validation
                        default=1,
                        type=int,
                        help='Number of contexts for validation.\
                              If 1, then the context is the entire training set.\
                              Otherwise, the context is randomly sampled from the training set using the context_size.')

    parser.add_argument('--aggregation_over_contexts_val',
                        default='none',
                        choices=[
                            'none',  # No aggregation. So the validation set is `initial_val_size` * `num_contexts_val`
                            'mean'   # Aggregate the predictions over the `num_contexts_val`
                        ],
                        type=str,
                        help='Aggregation method over context for validation')
    
    parser.add_argument('--num_contexts_test',
                        default=1,
                        type=int,
                        help='Number of contexts for testing.\
                              If 1, then the context is the entire training set.\
                              Otherwise, the context is randomly sampled from the training set using the context_size.')
    
    parser.add_argument('--aggregation_over_contexts_test',
                        default='mean',
                        choices=['mean'],   # TEST MUST ALWAYS BE AGGREGATED, BECAUSE WE DON'T WANT TO MAKE PREDICTIONS ON MULTIPLE "FAKE" TEST SAMPLES
                        type=str,
                        help='Aggregation method over context for testing')
    
    # ----- SMOTE on TabMDA -----
    parser.add_argument('--smote_TabMDA',
                        action='store_true',
                        dest='smote_TabMDA',
                        help='True if you want to apply SMOTE augmentation on TabMDA.')
    parser.set_defaults(smote_TabMDA=False)

    parser.add_argument('--smote_rounds',
                        default=1,
                        type=int,
                        help='Rounds of SMOTE')
    
    parser.add_argument('--smote_k',
                        default=3,
                        type=int,
                        help='Number of K-nearest neighbors for SMOTE')



    # ----- Classifier (either end-to-end trained with TabMDA, or a separate downstream classifier) -----
    parser.add_argument('--classifier_model',
                        choices=NON_DIFFERENTIABLE_CLASSIFIERS,
                        type=str,
                        help='Name of the classifier to use')

    parser.add_argument('--tabpfn_ensembles',
                        default=1,
                        type=int,
                        help='Number of ensembles for TabPFN classifier. Max 32')


    # ----- Training -----
    parser.add_argument('--learning_rate',
                        default=1e-3,
                        type=float,
                        help='Learning rate for the optimizer')
    parser.add_argument('--train_epochs',
                        default=None,
                        type=int,
                        help='Number of epochs for training. Provide either this or --train_steps, not both.')
    parser.add_argument('--train_steps',
                        default=None,
                        type=int,
                        help='Number of steps for training. Provide either this or --train_epochs, not both.')
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        help='Batch size for training')
    parser.add_argument('--weight_decay',
                        default=0.0,
                        type=float,
                        help='Weight decay for the optimizer')
    parser.add_argument('--optimizer',
                        default="adamw",
                        type=str,
                        help='Optimizer for training')


    # ----- W&B logger -----
    parser.add_argument('--tags', 
                        default=[], 
                        nargs='+', 
                        type=str, 
                        help='Tags for wandb. Intented use-case is defining a tag-name for each set of experiments.')

    parser.add_argument('--enable_wandb',
                        action='store_true', 
                        dest='enable_wandb',
                        help='True if you want to use W&B logging.')
    parser.set_defaults(enable_wandb=True)

    parser.add_argument('--log_test_metrics_during_training',
                        action='store_true',
                        dest='log_test_metrics_during_training',
                        help='True if you want to log test metrics during training')
    parser.set_defaults(log_test_metrics_during_training=False)

    args = parser.parse_args()
    logging.info(f"Parameters: {json.dumps(dict(vars(args)), indent=4)}")
    

    # ==============================================================
    #                        CHECK ARGUMENTS
    # ==============================================================
    if args.classifier_model == 'TabPFN':
        assert args.augmentor_model == 'none', f"TabPFN is not compatible with augmentor model {args.augmentor_model}"

    # ====== ENCODER-SPECIFIC CHECKS ======
    if args.augmentor_model == "none":
        print(f"Training a {args.classifier_model} classifier on the original data.")
    elif args.augmentor_model == "tabmda_encoder":
        if args.freeze_encoder is None:
            args.freeze_encoder = True
        assert args.freeze_encoder, f"tabmda_encoder requires the encoder to be frozen"

        if args.classifier_model in NON_DIFFERENTIABLE_CLASSIFIERS:
            print(f"=========================\n"
                  f"Training a {args.classifier_model} classifier on the encoded data using TabMDA encoder.\n"
                  f"The data is encoded once (because there is no mini-batch training).\n"
                  f"=========================")
        else:
            raise ValueError(f"tabmda_encoder is not compatible with classifier model {args.classifier_model}")

    # ====== CONTEXT SUBSETTING CHECKS ======
    # ---- Train ----
    if args.context_size == 1:
        args.num_contexts = 1
        logging.warning(f'Context size is 1. Setting num_contexts to 1 (while the user provided args.num_contexts = {args.num_contexts})')

    # ---- Validation ----
    if args.num_contexts_val != 1:
        assert args.classifier_model in NON_DIFFERENTIABLE_CLASSIFIERS,\
            "Context subsetting for validation is implemented only for non-differentiable classifiers (e.g., scikit-learn models)"

    # ---- Test ----
    if args.num_contexts_test != 1:
        assert args.classifier_model in NON_DIFFERENTIABLE_CLASSIFIERS,\
            "Context subsetting for test is implemented only for non-differentiable classifiers (e.g., scikit-learn models)"

        assert args.aggregation_over_contexts_test!='none',\
            "Test set must always be aggregated over num_contexts_test, because we can't compute the test accuracy over 'fake' test samples."

    # SMOTE params a dict
    smote_params = {"k": args.smote_k, "rounds": args.smote_rounds} if args.smote_TabMDA else None

    # ==============================================================
    #                       INIT W&B LOGGER
    # ==============================================================
    if not args.enable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    
    logging.info(f"[WanDB] Initializating WandB...")
    wandb.init(project="tabmda")
    logging.info(f"[WanDB] Initialized WandB")

    wandb.config.update(args, allow_val_change=True)
    args.global_step = 0  # The global step keeps track of the number of steps (batch updates),
                          #     and is used for logging.


    # ===== PREPARE DATASET =====
    try:
        dataset = get_tabular_dataset(
            dataset = args.dataset,
            num_real_samples = args.num_real_samples if args.num_real_samples != -1 else None,
            repeat_id = args.repeat_id
        )
    except ManualStopError as e:
        logging.error("[Mannual stop error] " + str(e))
        args.tags = ["manual-stop"]
        wandb.config.update(args, allow_val_change=True)
        wandb.finish()
        exit(0)
        

    X_train, y_train = dataset['X_train'], dataset['y_train']
    X_val, y_val = dataset['X_val'], dataset['y_val']
    X_test, y_test = dataset['X_test'], dataset['y_test']
    
    # ==== Store some dataset information ====
    args.num_classes = len(set(y_train))
    args.num_features = X_train.shape[1]
    args.train_size = len(X_train)
    args.val_size = len(X_val)
    args.test_size = len(X_test)
    wandb.config.update(args, allow_val_change=True)


    # ==== DECIDE TO AUGMENT THE DATASET OR NOT ====
    if args.augmentor_model == 'tabmda_encoder' and args.when_to_augment_data=='once_at_initialisation':
        TabMDA_model = construct_tabmda_model(classifier=None,
                                              freeze_encoder=True,
                                              device=device)

        # === Make data as Tensor and put on the device ===
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.long).to(device)
        X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.long).to(device)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.long).to(device)

        # ======= Encode the data =======
        X_train_enc, y_train_enc = TabMDA_model.encode_batch(
            batch={"x": X_train,             "y": y_train,
                   "x_context": [X_train],   "y_context": [y_train]},
            context_subsetting_params={"num_contexts": args.num_contexts,
                                       "context_size": args.context_size},
            smote_params=smote_params
        )
        X_val_enc, y_val_enc = TabMDA_model.encode_batch(
            batch={"x": X_val,                "y": y_val,
                    "x_context": [X_train],   "y_context": [y_train]},
            context_subsetting_params={"num_contexts": args.num_contexts_val,
                                        "context_size": 1 if args.num_contexts_val == 1 else args.context_size},
            smote_params=None
        )
        X_test_enc, y_test_enc = TabMDA_model.encode_batch(
            batch={"x": X_test,               "y": y_test,
                    "x_context": [X_train],   "y_context": [y_train]},
            context_subsetting_params={"num_contexts": args.num_contexts_test,
                                        "context_size": 1 if args.num_contexts_test == 1 else args.context_size},
            smote_params=None
        )
        print(f"[Data] Original train shape: {X_train.shape}")
        print(f"[Data] Encoded train shape: {X_train_enc.shape}")

        print(f"[Data] Original val shape: {X_val.shape}")
        print(f"[Data] Encoded val shape: {X_val_enc.shape}")

        print(f"[Data] Original test shape: {X_test.shape}")
        print(f"[Data] Encoded test shape: {X_test_enc.shape}")

        args.train_size_after_TabMDA = len(X_train_enc)
        args.val_size_after_TabMDA = len(X_val_enc)
        args.test_size_after_TabMDA = len(X_test_enc)
        wandb.config.update(args, allow_val_change=True)


        # ----- Ensure the data is numpy -----
        X_train_enc, y_train_enc = to_numpy(X_train_enc), to_numpy(y_train_enc)
        X_val_enc, y_val_enc = to_numpy(X_val_enc), to_numpy(y_val_enc)
        X_test_enc, y_test_enc = to_numpy(X_test_enc), to_numpy(y_test_enc)

        # ==== Prepare PytorchDataset ====
        train_data = PytorchDataset(x=X_train_enc, y=y_train_enc, x_context=X_train, y_context=y_train, device=device)
        val_data   = PytorchDataset(x=X_val_enc,   y=y_val_enc,   x_context=X_train, y_context=y_train, device=device)
        test_data  = PytorchDataset(x=X_test_enc,  y=y_test_enc,  x_context=X_train, y_context=y_train, device=device)
    
    else:
        train_data = PytorchDataset(x=X_train, y=y_train, x_context=X_train, y_context=y_train, device=device)
        val_data   = PytorchDataset(x=X_val,   y=y_val,   x_context=X_train, y_context=y_train, device=device)
        test_data  = PytorchDataset(x=X_test,  y=y_test,  x_context=X_train, y_context=y_train, device=device)
    

    if args.classifier_model in NON_DIFFERENTIABLE_CLASSIFIERS:
        classifier = create_scikit_classifier(args, model_name=args.classifier_model)

        train_data.move_to_device(torch.device('cpu'))
        val_data.move_to_device(torch.device('cpu'))
        test_data.move_to_device(torch.device('cpu'))

        if args.classifier_model == "TabPFN":
            classifier.fit(train_data.x, train_data.y, overwrite_warning=True)
        else:
            classifier.fit(train_data.x, train_data.y)
        y_train_pred = to_numpy(classifier.predict(train_data.x))
        y_val_pred = to_numpy(classifier.predict(val_data.x))
        y_test_pred = to_numpy(classifier.predict(test_data.x))
        y_test_pred_proba = to_numpy(classifier.predict_proba(test_data.x))

        if args.augmentor_model == "none":
            pass
        elif args.augmentor_model == "taics_encoder":
            
            # ==============================================================
            #          SCIKIT CLASSIFIER on TAICS ENCODER
            # ==============================================================      

            match args.aggregation_over_contexts_test:
                case 'mean':
                    # === Average the predicted probabilities over the contexts ===
                    y_test_pred_proba_aggregated = y_test_pred_proba.reshape(-1, args.num_contexts_test, args.num_classes)
                    y_test_pred_proba_aggregated = np.mean(y_test_pred_proba_aggregated, axis=1)
                    y_test_pred = np.argmax(y_test_pred_proba_aggregated, axis=1)
                case _:
                    raise ValueError(f"Aggregation method {args.aggregation_over_contexts_test} is not supported for test set.")
        else:
            raise ValueError(f"The combination of TAICS model {args.augmentor_model} and classifier model {args.classifier_model} is not supported.")


        # ==============================================================
        #                       SAVE THE METRICS
        # ==============================================================      
        metrics = {
            'train/balanced_accuracy': balanced_accuracy_score(train_data.y, y_train_pred),
            'val/balanced_accuracy': balanced_accuracy_score(val_data.y, y_val_pred),
            'test/balanced_accuracy': balanced_accuracy_score(test_data.y, y_test_pred),
        }

        # === Save the test accuracy per class ===
        for class_id in range(args.num_classes):
            metrics[f'test/balanced_accuracy_class_{class_id}'] = accuracy_score(test_data.y[test_data.y==class_id], y_test_pred[test_data.y==class_id])

        wandb.log(metrics)
        logging.info(f'Metrics: {metrics}')

    wandb.finish()

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from TabPFN.tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
import torch


def create_scikit_classifier(args, model_name):
    match model_name:
        case 'LogReg':
            classifier = LogisticRegression(max_iter=1000, random_state=0)
        case 'KNN':
            classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        case 'DecisionTree':
            classifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=0)
        case 'RandomForest':
            classifier = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=2, random_state=0)
        case 'XGBoost':
            # --- Hyper-parameters following Synthcity ---
            classifier = XGBClassifier(
                n_estimators = 200, learning_rate = 0.3,
                n_jobs=2, verbosity=0, max_depth=3, random_state=0, device='cuda' if torch.cuda.is_available() else 'cpu')
        case 'TabPFN':
            classifier = TabPFNClassifier(
                device='cpu',
                N_ensemble_configurations=args.tabpfn_ensembles,
                only_inference=True,

                # === Settings to enable sampling from an EBM ===
                no_grad=True,			    # if False, it allows passing inputs as tensors and computing the gradient wrt the inputs
                no_preprocess_mode=False    # it requires to be True if no_grad is False
            )
        case _:
            raise ValueError(f"Model {model_name} not recognized")

    return classifier
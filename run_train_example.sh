python train.py\
    --dataset 'mfeat-fourier'\
    --num_real_samples -1 \
    --train_epochs 20\
    --context_size 0\
    --num_contexts 20\
    --repeat_id 0\
    --augmentor_model 'tabmda_encoder'\
    --classifier_model 'LogReg'\
    --log_test_metrics_during_training
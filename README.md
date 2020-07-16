# DML CIFAR10 / CIFAR 100

Refactored the code so that it is easier to scale.

Args:
```
--num_model: number of models to use
--num_clf: number of classifiers in each model
--diffaug: if ON, use different agumentation for each classifier
```

Full Args:
```
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of total epochs to run
  --schedule SCHEDULE [SCHEDULE ...]
                        Decrease learning rate at these epochs.
  --batch_size BATCH_SIZE
                        The size of batch
  --lr LR               initial learning rate
  --momentum MOMENTUM   momentum
  --weight_decay WEIGHT_DECAY
                        weight decay
  --cuda CUDA
  --t_name T_NAME
  --t_depth T_DEPTH
  --t_widen_factor T_WIDEN_FACTOR
  --t_alpha T_ALPHA
  --num_model NUM_MODEL
                        number of models
  --num_clf NUM_CLF     number of classifiers in a model
  --data_name DATA_NAME
  --num_class NUM_CLASS
  --dataset_path DATASET_PATH
  --diffaug             if ture, use differernt augmentation for different
                        models and classifiers
  --use_intra
  --no_intra
  --intra_step INTRA_STEP
  --intra_ratio INTRA_RATIO
  --intra_loss_type {l1,soft_l1}
  --use_inter
  --no_inter
  --inter_step INTER_STEP
  --inter_ratio INTER_RATIO
  --inter_loss_type {l1,soft_l1}
  --use_ensemble
  --no_ensemble
  --ensemble_step ENSEMBLE_STEP
  --ensemble_type ENSEMBLE_TYPE
  --ensemble_ratio ENSEMBLE_RATIO
  --ensemble_mode {average,batch_weighted,sample_weighted}
  --ensemble_temp ENSEMBLE_TEMP
```
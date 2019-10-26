import torch
import os
from models import SpeakerDiarization
from dataset import create_dataloders
from losses import CustomCrossEntropyLoss, HardTripletLoss
from augmentation import get_training_augmentation, get_valid_augmentation
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import (AccuracyCallback, AUCCallback, F1ScoreCallback,
                                   CriterionCallback, CriterionAggregatorCallback)
from catalyst.contrib.schedulers import OneCycleLRWithWarmup


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_file = 'txt with training data | path label'
valid_file = 'txt with validation data | path label'
data_folder = 'root folder'
meta_info_file = 'path to meta.csv'
bs = 20
num_workers = 8
use_sampler = False
feature_kind = 0
augmenter = get_training_augmentation()
crop = True


# parameters
lr = 0.001
num_classes = 5994
dim = 512

# spectrogram_parameters
n_fft = 512
hop_length = 160
win_length = 400
spectrogram_length = 250

transformer_parameters = {'n_fft': n_fft, 'win_length': win_length, 'hop_length': hop_length}

class_names = [str(i) for i in range(num_classes)]

model = SpeakerDiarization(dim, num_classes, use_attention=False)

augmenters = {'train': get_training_augmentation(), 'valid': get_valid_augmentation()}

loaders = create_dataloders(
    train_file,
    valid_file,
    data_folder,
    meta_info_file,
    bs=bs,
    num_classes=num_classes,
    num_workers=num_workers,
    augmenters=augmenters,

)

logdir = "./logs/classification_tutorial_0"
num_epochs = 10

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer, milestones=[9], gamma=0.3
# )

criterion = {
    "ce": CustomCrossEntropyLoss(),
    #"htl": HardTripletLoss(device='cuda'),
}

scheduler = OneCycleLRWithWarmup(
    optimizer,
    num_steps=num_epochs,
    lr_range=(0.001, 0.0001),
    warmup_steps=1
)

runner = SupervisedRunner(
    input_key='features',
    output_key=['embeddings', 'logits']
)

runner.train(
    model=model,
    logdir=logdir,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    # We can specify the callbacks list for the experiment;
    # For this task, we will check accuracy, AUC and F1 metrics
    callbacks=[
        AccuracyCallback(num_classes=num_classes),
        AUCCallback(
            num_classes=num_classes,
            input_key="targets_one_hot",
            class_names=class_names
        ),
        F1ScoreCallback(
            input_key="targets_one_hot",
            activation="Softmax"
        ),
        CriterionCallback(
            input_key="targets",
            prefix="loss_ce",
            criterion_key="ce"
        ),
        # CriterionCallback(
        #     input_key="targets",
        #     output_key="embeddings",
        #     prefix="loss_htl",
        #     criterion_key="htl"
        # ),
        # CriterionAggregatorCallback(
        #     prefix="loss",
        #     loss_keys=["loss_ce", "loss_htl"],
        #     loss_aggregate_fn="sum"
        # ),
    ],
    num_epochs=num_epochs,
    verbose=True
)
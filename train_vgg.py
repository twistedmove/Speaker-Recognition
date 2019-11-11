import argparse
import torch
from speaker_diarization import *
from catalyst.dl.runner import SupervisedWandbRunner as SupervisedRunner
from catalyst.dl.callbacks import (AccuracyCallback,
                                   CriterionCallback, CriterionAggregatorCallback)
from catalyst.contrib.schedulers import OneCycleLRWithWarmup
from collections import OrderedDict


parser = argparse.ArgumentParser(description="Train model for speaker diarization problem")
parser.add_argument("--data_folder", help="pre-training dataset", type=str)
parser.add_argument("--train_file", help="file with train split", type=str)
parser.add_argument("--valid_file", help="file with valid split", type=str)
parser.add_argument("--meta_info_file", help="meta information (e.g gender)", type=str)
parser.add_argument("--log_dir", help="log_dir", type=str)
# parameters

parser.add_argument("--num_epochs", help="number of epochs", type=int, default=120)
parser.add_argument("--batch_size", help="batch size", type=int, default=80)
parser.add_argument("--fp16", help="use fp16", type=bool, default=False)
parser.add_argument("--num_workers", help="num workers", type=int, default=4)
parser.add_argument("--feature_kind", help="feature kind", type=int, default=0)
parser.add_argument("--num_classes", help="num classes", type=int, default=5994)
parser.add_argument("--dim", help="dim", type=int, default=512)
parser.add_argument("--lr", help="lr", type=float, default=1e-3)
parser.add_argument("--triplet_loss", help="triplet loss", type=bool, default=False)
parser.add_argument("--one_hot_encoding", help="one hot encoding", type=bool, default=False)
parser.add_argument("--clamp", help="clamp", type=bool, default=False)
parser.add_argument("--scheduler", help="scheduler", type=str, default='MultiStepLR')

args = parser.parse_args()

# spectrogram_parameters
n_fft = 512
hop_length = 160
win_length = 400
spectrogram_length = 250

transformer_parameters = {'n_fft': n_fft, 'win_length': win_length, 'hop_length': hop_length}

class_names = [str(i) for i in range(args.num_classes)]

model = SpeakerDiarization(args.dim, args.num_classes, use_attention=False).to('cuda')

if args.clamp:
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, min=1e-8))

augmenters = {'train': get_training_augmentation(), 'valid': get_valid_augmentation()}


fp16 = None
if args.fp16:
    args.batch_size *= 2
    fp16 = dict(opt_level="O1")

loaders = create_dataloders(
    args.train_file,
    args.valid_file,
    args.data_folder,
    args.meta_info_file,
    one_hot_encoding=args.one_hot_encoding,
    bs=args.batch_size,
    num_classes=args.num_classes,
    num_workers=args.num_workers,
    augmenters=augmenters,

)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

criterion = OrderedDict({
    "ce": CustomCrossEntropyLoss(),
})

if args.triplet_loss:
    criterion["htl"] = HardTripletLoss(squared=True)

if args.scheduler == 'OneCycleLRWithWarmup':
    scheduler = OneCycleLRWithWarmup(
        optimizer,
        num_steps=args.num_epochs,
        lr_range=(args.lr, args.lr * 1e-2),
        warmup_steps=36,
    )
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[36, 72, 108], gamma=0.1
    )

runner = SupervisedRunner(
    input_key='features',
    output_key=['embeddings', 'logits']
)

callbacks = [
                AccuracyCallback(
                    num_classes=args.num_classes,
                    accuracy_args=[1],
                    activation="Softmax",
                ),
                CriterionCallback(
                    input_key="targets",
                    prefix="loss",
                    criterion_key="ce"
                ),

            ]

if args.triplet_loss:
    callbacks.extend([
        CriterionCallback(
            input_key="targets",
            output_key="embeddings",
            prefix="loss",
            criterion_key="htl"
        ),
        CriterionAggregatorCallback(
            prefix="loss",
            loss_keys=["ce", "htl"],
            loss_aggregate_fn="sum"
        )]
    )

runner.train(
    model=model,
    logdir=args.log_dir,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=callbacks,
    num_epochs=args.num_epochs,
    fp16=fp16,
    verbose=True
)

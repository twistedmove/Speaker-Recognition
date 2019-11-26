import argparse
import torch
import yaml
from speaker_diarization import *
from catalyst.dl.runner import SupervisedWandbRunner as SupervisedRunner
from catalyst.dl.callbacks import (AccuracyCallback,
                                   CriterionCallback, CriterionAggregatorCallback)
from catalyst.contrib.schedulers import OneCycleLRWithWarmup
from collections import OrderedDict


parser = argparse.ArgumentParser(description="Train model for speaker_diarization problem")
parser.add_argument("--data_folder", help="pre-training dataset", type=str,
                    default='/data_ssd/VoxCeleb2/vox2_dev_dataset/dev/aac')
parser.add_argument("--train_file", help="file with train split", type=str,
                    default='/data_ssd/VoxCeleb2/meta/voxlb2_train.txt')
parser.add_argument("--valid_file", help="file with valid split", type=str,
                    default='/data_ssd/VoxCeleb2/meta/voxlb2_val.txt')
parser.add_argument("--meta_info_file", help="meta information (gender e.g)", type=str,
                    default='/data_ssd/VoxCeleb2/vox2_meta.csv')
parser.add_argument("--log_dir", help="log_dir", type=str, default='/data_ssd/VoxCeleb2-one-logs/')
parser.add_argument("--resume", help="resume path", type=str, default=None)
# parameters

parser.add_argument("--num_epochs", help="number of epochs", type=int, default=120)
parser.add_argument("--batch_size", help="batch_size", type=int, default=80)
parser.add_argument("--fp16", help="use fp16", type=bool, default=False)
parser.add_argument("--num_workers", help="num_workers", type=int, default=4)
parser.add_argument("--feature_kind", help="feature_kind", type=int, default=0)
parser.add_argument("--num_classes", help="num_classes", type=int, default=5994)
parser.add_argument("--dim", help="dim", type=int, default=512)
parser.add_argument("--min_lr", help="lr", type=float, default=1e-4)
parser.add_argument("--max_lr", help="lr", type=float, default=1e-1)
parser.add_argument("--triplet_loss", help="triplet_loss", type=bool, default=False)
parser.add_argument("--one_hot_encoding", help="one_hot_encoding", type=bool, default=False)
parser.add_argument("--clamp", help="clamp", type=bool, default=False)
parser.add_argument("--scheduler", help="scheduler", type=str, default='MultiStepLR')
parser.add_argument('--attention', dest='attention', action='store_true')
parser.add_argument('--no-attention', dest='attention', action='store_false')
parser.set_defaults(attention=True)

args = parser.parse_args()

with open('spectrogram.yaml', 'r') as file:
    settings = yaml.safe_load(file)

# spectrogram_parameters
n_fft = settings['n_fft']
hop_length = settings['hop_length']
win_length = settings['win_length']
spectrogram_length = settings['spectrogram_length']
n_mels = settings['n_mels']

class_names = [str(i) for i in range(args.num_classes)]

model = SpeakerDiarization(args.dim, args.num_classes, use_attention=args.attention).to('cuda')

if args.clamp:
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, min=1e-8))

augmenters = {'train': get_training_augmentation(n_fft, hop_length, win_length, n_mels, spectrogram_length),
              'valid': get_valid_augmentation(n_fft, hop_length, win_length, n_mels, spectrogram_length)}


fp16 = None
if args.fp16:
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

optimizer = torch.optim.Adam(model.parameters(), lr=args.max_lr, weight_decay=1e-5)

criterion = OrderedDict({
    "ce": CustomCrossEntropyLoss(),
})

if args.triplet_loss:
    criterion["htl"] = HardTripletLoss(squared=True)

if args.scheduler == 'OneCycleLRWithWarmup':
    scheduler = OneCycleLRWithWarmup(
        optimizer,
        num_steps=args.num_epochs,
        lr_range=(args.max_lr, args.min_lr),
        warmup_steps=args.num_epochs // 4,
        momentum_range=(0.85, 0.95),
    )
else:
    step = len(range(0, args.num_epochs, 4))
    milestones = [step * i for i in range(1, 4)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
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

_callbacks = OrderedDict()
callback_names = ['accuracy', 'criterion_ce', 'criterion_htl', 'criterion_aggregator']

for i, c in enumerate(callbacks):
    _callbacks[callback_names[i]] = c

callbacks = _callbacks

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
    resume=args.resume,
    verbose=True
)
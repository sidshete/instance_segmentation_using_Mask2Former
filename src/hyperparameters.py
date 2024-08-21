# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
from hyperopt import hp, tpe, Trials, fmin, space_eval

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

#Global variable
training_samples = 110

#This variables will be rewrote during training stage
experiment_dir = str(0)
lr = 0.0001
batch = 1
epochs = 10
wd = 0.005
MAX_ITER = int(epochs * training_samples / batch)
STEPS = (int(0.89 * MAX_ITER), int(0.96 * MAX_ITER))
WARMUP_ITERS = 5
EVAL_PERIOD = int(0.0135 * MAX_ITER)





class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Instance segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    #Setup our parameters for optimization
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.IMS_PER_BATCH = batch
    cfg.SOLVER.MAX_ITER = MAX_ITER 
    cfg.SOLVER.STEPS = STEPS
    cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS
    cfg.SOLVER.WEIGHT_DECAY = wd
    cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
    cfg.OUTPUT_DIR = experiment_dir
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


# Set project path
project_dir = os.getcwd()
results_dir = os.path.join(project_dir, 'results_hyperparameter_optimization')
os.makedirs(results_dir, exist_ok=True)

# Define the search space for hyperparameters
space = {
    'lr': hp.uniform('lr', 1e-5, 1e-3),
    'batch': hp.choice('batch', [1, 2]),
    'epochs': hp.choice('epochs', [50]),
    'wd': hp.uniform('wd', 0, 5e-4),
}

#Extract total loss from training process
def total_loss_extract(file):
    with open(file) as metrics_file:
        raw_content = metrics_file.read()
    metrics_lines = raw_content.strip().split('\n')
    metrics_data = [json.loads(line) for line in metrics_lines]
    metrics_df = pd.DataFrame(metrics_data)
    for col in metrics_df.columns:
        if loss_type in col:
            return metrics_df[col].mean()

# Define the objective function for hyperparameter optimization
def objective(params):
    try:
        global lr
        global batch
        global epochs
        global wd
        lr = params['lr']
        batch = params['batch']
        epochs = params['epochs']
        wd = params['wd']

        #Calulate iterations metrics
        global MAX_ITER
        global STEPS
        global WARMUP_ITERS
        global EVAL_PERIOD
        MAX_ITER = int(epochs * training_samples / batch)
        STEPS = (int(0.89 * MAX_ITER), int(0.96 * MAX_ITER))
        WARMUP_ITERS = 10
        if(MAX_ITER < 100000):
            WARMUP_ITERS = 5
        EVAL_PERIOD = int(0.0135 * MAX_ITER)

        #Name of this experiment
        experiment_name = f"model_lr_{lr}_b{batch}_epochs{epochs}_wd{wd}"
        global experiment_dir
        experiment_dir = os.path.join(results_dir, experiment_name)
        
        # Create directory for this experiment
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Print the combination of hyperparameters
        print(f"Training with hyperparameters: {params}")

        args = default_argument_parser().parse_args()
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )

        '''
        Here should be written how to extract losses, that we will return 
        (maybe read metrics.json file and extract mean total_loss)
        '''
        return {
            'loss': total_loss_extract(os.path.join(experiment_dir, 'metrics.json')),
            'status': 'ok'
        }

    except Exception as e:
        # Log the exception and return a high loss value
        print(f"Exception occurred: {str(e)}")
        return {
            'loss': float('inf'),
            'status': 'fail'
        }

# Ensure the DETECTRON2_DATASETS environment variable is set
dataset_path = os.environ['DATA_PATH']
dataset_path = os.getenv('DETECTRON2_DATASETS', dataset_path)

# Register the COCO dataset
register_coco_instances("my_dataset_train", {}, os.path.join(dataset_path, "coco/annotations/instances_train2017.json"), os.path.join(dataset_path, "coco/train2017"))
register_coco_instances("my_dataset_val", {}, os.path.join(dataset_path, "coco/annotations/instances_val2017.json"), os.path.join(dataset_path, "coco/val2017"))



# Run hyperparameter optimization
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)

# Print best hyperparameters and retrain the model
print("Best hyperparameters:", best)
best_params = space_eval(space, best)
print("Retraining model with best hyperparameters...")
print(best_params)

# Define experiment name and directory for the best model
best_experiment_name = f"best_model_lr0_{best_params['lr0']:.1e}_lrf_{best_params['lrf']:.1e}_b{best_params['batch']}_epochs{best_params['epochs']}_wd{best_params['wd']:.1e}"
best_experiment_dir = os.path.join(results_dir, best_experiment_name)

# Create directory for the best experiment
os.makedirs(best_experiment_dir, exist_ok=True)

# Train the model with the best hyperparameters
lr = best_params['lr']
batch = best_params['batch']
epochs = best_params['epochs']
wd = best_params['wd']

#Calulate iterations metrics
MAX_ITER = int(epochs * training_samples / batch)
STEPS = (int(0.89 * MAX_ITER), int(0.96 * MAX_ITER))
WARMUP_ITERS = 10
if(MAX_ITER < 100000):
    WARMUP_ITERS = 5
EVAL_PERIOD = int(0.0135 * MAX_ITER)

# Ensure the DETECTRON2_DATASETS environment variable is set
dataset_path = os.environ['DATA_PATH']
dataset_path = os.getenv('DETECTRON2_DATASETS', dataset_path)

# Register the COCO dataset
register_coco_instances("my_dataset_train", {}, os.path.join(dataset_path, "coco/annotations/instances_train2017.json"), os.path.join(dataset_path, "coco/train2017"))
register_coco_instances("my_dataset_val", {}, os.path.join(dataset_path, "coco/annotations/instances_val2017.json"), os.path.join(dataset_path, "coco/val2017"))

args = default_argument_parser().parse_args()
launch(
    main,
    args.num_gpus,
    num_machines=args.num_machines,
    machine_rank=args.machine_rank,
    dist_url=args.dist_url,
    args=(args,),
)

print(f"Best model saved in {best_experiment_dir}")



 

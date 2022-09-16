#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict

import wandb
import yaml

from configs.all_configs import (
    RenderingArguments,
    ModelArguments,
    CustomTrainingArguments,
    DataTrainingArguments,
)
from typing import Any, Dict, Optional, Union, List
import numpy as np
import datasets
import torch
import transformers
from datasets import interleave_datasets, load_dataset

from configs.config_maps import MODEL_PROTOTYPE_CONFIGS, TRAINING_CONFIGS
from dataset_synthesis.document_syntesis import DocumentSynthesizer
from dataset_synthesis.synthetic_dataset import (
    SyntheticDatasetTransform,
    SyntheticDatasetTorch,
)
from pixel import (
    PIXELConfig,
    PIXELEmbeddings,
    PIXELForPreTraining,
    PIXELTrainerForPretraining,
    get_2d_sincos_pos_embed, process_remaining_strings, get_config_dict,
)
from transformers import HfArgumentParser, TrainingArguments, ViTFeatureExtractor, TrainerCallback, TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from submitit_pretrain import SubmititTrainingArguments

""" Pre-training a PIXEL model as an MAE (masked autoencoder)"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")

class VisualizationCallback(TrainerCallback):
    def __init__(self, args=None):
        self.args = args

    def _clip(self, img: torch.Tensor):
        img = torch.einsum("chw->hwc", img)
        img = torch.clip(img * 255, 0, 255)
        img = torch.einsum("hwc->chw", img)
        return img

    def _log_image(self, figures_to_log: dict):
        for k, v in figures_to_log.items():
            if k in ["reconstruction", "predictions", "masked_predictions"]:
                wandb.log({k: [wandb.Image(x) for x in v]})
            else:
                wandb.log({k: [wandb.Image(self._clip(x)) for x in v]})

    def on_evaluate(self, args, state: TrainerState, control, eval_dataloader=None, model=None, **kwargs):
        logger.info(f"logging images. Global rank: {state.is_world_process_zero}, local rank: {state.is_local_process_zero}")
        if not state.is_world_process_zero:
            return
        figures_to_log = defaultdict(list)
        batch = next(iter(eval_dataloader))
        batch = {k: v.to(args.device) for k, v in batch.items()}
        logger.info(f"visualizing {batch['attention_mask'].shape[0]} images")

        model.eval()
        with torch.inference_mode():
            outputs = model(pixel_values=batch["pixel_values"],
                            attention_mask=batch["attention_mask"],
                            patch_mask=batch["patch_mask"])
        for i in range(len(outputs["logits"])):  # TODO don't duplicate code
            predictions = model.unpatchify(outputs["logits"][i].unsqueeze(0)).detach().cpu().squeeze()
            mask = outputs["mask"][i].unsqueeze(0).detach().cpu()
            mask = mask.unsqueeze(-1).repeat(1, 1, 16 ** 2 * 3)
            mask = model.unpatchify(mask).squeeze()
            figures_to_log["mask"].append(mask)

            attention_mask = batch["attention_mask"][i].view(1,-1, 1).detach().cpu().repeat(1, 1, 16 ** 2 * 3)
            attention_mask = model.unpatchify(attention_mask).squeeze()
            figures_to_log["attention_mask"].append(attention_mask)

            original_image = model.unpatchify(model.patchify(batch["pixel_values"][i].unsqueeze(0))).detach().cpu().squeeze()
            figures_to_log["original_image"].append(original_image)

            im_masked = original_image * (1 - mask)
            figures_to_log["im_masked"].append(im_masked)

            figures_to_log["predictions"].append(predictions)

            masked_predictions = predictions * mask * attention_mask
            figures_to_log["masked_predictions"].append(masked_predictions)

            reconstruction = (
                    original_image * (1 - (torch.bitwise_and(mask == 1, attention_mask == 1)).long())
                    + predictions * mask * attention_mask
            )
            figures_to_log["reconstruction"].append(reconstruction)

        self._log_image(figures_to_log)
        model.train()


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    inputs = {"pixel_values": pixel_values, "attention_mask": attention_mask}
    if "patch_mask" in examples[0]:
        patch_mask = torch.stack([example["patch_mask"] for example in examples])
        inputs.update({"patch_mask": patch_mask})
    return inputs


def get_datasets(model_args, data_args, training_args, rendering_args):
    train_text_datasets = [
        load_dataset(
            d_name,
            d_config,
            split=d_split,
            use_auth_token=model_args.use_auth_token,
            cache_dir=d_cache,
        )
        for d_name, d_config, d_split, d_cache in zip(
            data_args.train_dataset_names,
            data_args.train_dataset_configs,
            data_args.train_splits,
            data_args.dataset_caches,
        )
    ]
    dataset_sizes = [ds._info.splits.total_num_examples for ds in train_text_datasets]
    combined_size = sum(dataset_sizes)
    dataset_sampling_probs = [d_size / combined_size for d_size in dataset_sizes]
    train_text_dataset = interleave_datasets(
        train_text_datasets,
        probabilities=dataset_sampling_probs,
        seed=training_args.seed,
    )
    rng = np.random.RandomState(training_args.seed)
    transform = SyntheticDatasetTransform(rendering_args, rng=rng)
    ds = DocumentSynthesizer(rendering_args, rng=rng)
    train_dataset = SyntheticDatasetTorch(
        train_text_dataset,
        transform=transform,
        args=rendering_args,
        document_synthesizer=ds,
        overfit=training_args.overfit,
        rng=rng
    )
    train_dataset.max_step = (training_args.max_steps * training_args.per_device_train_batch_size) / training_args.dataloader_num_workers
    train_dataset.warmup_steps = training_args.warmup_render_steps
    train_dataset.randomness_intensity_update_interval = training_args.randomness_intensity_update_interval
    return train_dataset, dataset_sampling_probs


def main(config_dict: Dict[str, Any] = None):
    # wandb.init(project="pixel",config=config_dict, name=config_dict["run_name"])
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            CustomTrainingArguments,
            RenderingArguments,
        )
    )
    if not config_dict:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            (
                model_args,
                data_args,
                training_args,
                rendering_args,
            ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            (
                model_args,
                data_args,
                training_args,
                rendering_args,
            ) = parser.parse_args_into_dataclasses()
    else:
        (
            model_args,
            data_args,
            training_args,
            rendering_args,
        ) = parser.parse_dict(config_dict)

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our datasets

    train_dataset, dataset_sampling_probs = get_datasets(model_args, data_args, training_args, rendering_args)
    logger.info("***** Interleaving training datasets *****")
    for d_name, d_config, d_split, d_sampling_prob, d_cache in zip(
        data_args.train_dataset_names,
        data_args.train_dataset_configs,
        data_args.train_splits,
        dataset_sampling_probs,
        data_args.dataset_caches,
    ):
        logger.info(
            f"\tDataset name = {d_name}, config = {d_config}, split = {d_split}, "
            f"sampling probability = {d_sampling_prob:.3f}, cache = {d_cache}"
        )

    validation_dataset = train_dataset.get_evaluation_set()

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token,
    }
    logger.info(f"Using dropout with probability {model_args.dropout_prob}")

    if model_args.config_name:
        config = PIXELConfig.from_pretrained(
            model_args.config_name,
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
    elif model_args.model_name_or_path:
        config = PIXELConfig.from_pretrained(
            model_args.model_name_or_path,
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
    else:
        config = PIXELConfig(
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # Adapt config
    config.update(
        {
            "mask_ratio": model_args.mask_ratio,
            "norm_pix_loss": model_args.norm_pix_loss,
            "architectures": [PIXELForPreTraining.__name__],
        }
    )

    # Create model
    if model_args.model_name_or_path:
        model = PIXELForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            **config_kwargs,
        )
    else:
        logger.info("Training new model from scratch")
        model = PIXELForPreTraining(config)

    # Load or create feature extractor
    if model_args.feature_extractor_name:
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            model_args.feature_extractor_name, **config_kwargs
        )
    elif model_args.model_name_or_path:
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        feature_extractor = ViTFeatureExtractor()

    # Adjust image size
    image_height = rendering_args.figure_size[0]
    image_width = rendering_args.figure_size[0]
    model.config.image_size = (image_height, image_width)
    model.image_size = (image_height, image_width)
    feature_extractor.size = (image_height, image_width)

    # Reinitialize embeddings
    model.vit.embeddings = PIXELEmbeddings(model.config)
    model.decoder.decoder_pos_embed = torch.nn.Parameter(
        torch.zeros((1, model_args.max_seq_length + 1, 512)), requires_grad=False
    )
    decoder_pos_embed = get_2d_sincos_pos_embed(
        model.decoder.decoder_pos_embed.shape[-1],
        int(model_args.max_seq_length ** 0.5),
        add_cls_token=True,
    )
    model.decoder.decoder_pos_embed.data.copy_(
        torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
    )

    logger.info("***** Final model config *****")
    logger.info(config)

    total_params = sum([p.numel() for p in model.parameters()])
    logger.info(f"Total parameters count: {total_params}")
    encoder_params = sum([p.numel() for p in model.vit.parameters()])
    logger.info(f"Encoder parameters count: {encoder_params}")
    encoder_embedding_params = sum(
        [p.numel() for p in model.vit.embeddings.parameters()]
    )
    logger.info(f"Encoder embeddings parameters count: {encoder_embedding_params}")
    decoder_params = sum([p.numel() for p in model.decoder.parameters()])
    logger.info(f"Decoder parameters count: {decoder_params}")

    # Get patch mask generator if span masking

    column_names = ["pixel_values", "num_patches", "mask"]
    image_column_name = column_names[0]

    total_train_batch_size = (
        training_args.train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = (
            training_args.base_learning_rate * total_train_batch_size / 256
        )

    # Initialize our trainer
    trainer = PIXELTrainerForPretraining(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset,
        data_collator=collate_fn,
    )

    if training_args.do_eval:
        logger.info(f"adding visualization callback")
        trainer.add_callback(VisualizationCallback())

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        # Also save feature extractor together with model and text renderer
        feature_extractor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    print("eval metrics:", metrics)
    # Write model card and (optionally) push to hub
    kwargs = {
        "tasks": "masked-auto-encoding",
        "dataset": "wikipedia + bookcorpus",
        "tags": ["masked-auto-encoding"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    trainer_parser = HfArgumentParser(SubmititTrainingArguments)
    args, remaining_strings = trainer_parser.parse_args_into_dataclasses(return_remaining_strings=True)

    args_dict = asdict(args)
    # Get run configuration
    if remaining_strings:
        remaining_strings_dict = process_remaining_strings(remaining_strings)
        args_dict.update(remaining_strings_dict)

    config_dict = get_config_dict(**args_dict)
    main(config_dict)

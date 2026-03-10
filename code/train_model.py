#!/usr/bin/env python3
"""
train_model.py – Main training script for spoken language identification.

Supports three modes via --mode flag:
  standard  : Fine-tune a pretrained audio model for language classification
  dann      : Train with DANN for speaker-invariant features
  dann_aug  : Train DANN with speed-perturbation augmentation

Usage examples:
  python train_model.py --mode standard --model_id facebook/w2v-bert-2.0 --lr 4e-5 --epochs 6
  python train_model.py --mode dann --model_id facebook/w2v-bert-2.0 --lr 4e-5 --epochs 6
  python train_model.py --mode dann_aug --model_id facebook/w2v-bert-2.0 --lr 4e-5 --epochs 6
"""
import argparse
import numpy as np
import torch
import wandb

from transformers import TrainingArguments, Trainer, set_seed
from huggingface_hub import login

from src.data import (
    load_and_prepare_dataset,
    get_label_mappings,
    build_preprocess_fn,
    encode_dataset,
    AudioDataCollator,
    DANNDataCollator,
)
from src.model import get_input_features_key, load_feature_extractor, load_classification_model
from src.dann import DANNModel, DANNTrainer
from src.utils import compute_metrics

import evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Spoken Language Identification Training")
    p.add_argument("--mode", choices=["standard", "dann", "dann_aug"], default="standard")
    p.add_argument("--model_id", type=str, default="facebook/w2v-bert-2.0")
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.15)
    p.add_argument("--max_duration", type=float, default=2.0,
                   help="Maximum audio duration in seconds")
    p.add_argument("--dropout", action="store_true", default=False,
                   help="Enable dropout regularisation")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--wandb_key", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="Indic-SLID")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # GPU check
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Optional logins
    if args.hf_token:
        login(token=args.hf_token)
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    # ── Data ──────────────────────────────────────────────────────
    input_features_key = get_input_features_key(args.model_id)
    feature_extractor = load_feature_extractor(args.model_id)
    train_ds, valid_ds = load_and_prepare_dataset()
    str_to_int, int_to_str = get_label_mappings(train_ds)
    num_labels = len(str_to_int)

    # we only need speaker ids when doing adversarial training
    speaker2id = None
    if args.mode in ("dann", "dann_aug"):
        speakers = sorted(set(train_ds["speaker_id"]))
        speaker2id = {s: i for i, s in enumerate(speakers)}
        num_speakers = len(speaker2id)
        print(f"Number of speakers: {num_speakers}")

    augment = args.mode == "dann_aug"
    preprocess_fn = build_preprocess_fn(
        feature_extractor, str_to_int, input_features_key,
        max_duration=args.max_duration,
        speaker2id=speaker2id,
        augment=augment,
    )

    keep_cols = ["speaker_id", "language"]
    train_ds_enc = encode_dataset(train_ds, preprocess_fn, keep_cols)
    valid_ds_enc = encode_dataset(valid_ds, preprocess_fn, keep_cols)

    # ── Model ─────────────────────────────────────────────────────
    if args.mode == "standard":
        # plain classification head on top of the pretrained encoder
        model, config = load_classification_model(
            args.model_id, num_labels, str_to_int, int_to_str,
            apply_dropout=args.dropout,
        )
        data_collator = AudioDataCollator(feature_extractor, input_features_key)
    else:
        # dual-head DANN: language + speaker classifiers sharing the encoder
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_id)
        model = DANNModel(args.model_id, num_labels, num_speakers, config=config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # DANN collator also batches speaker ids alongside the audio
        data_collator = DANNDataCollator(feature_extractor, input_features_key)

    # ── Training args ─────────────────────────────────────────────
    run_name = f"{args.mode}_{args.model_id.split('/')[-1]}_lr{args.lr}"
    wandb.init(project=args.wandb_project, name=run_name)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to="wandb",
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        # DANN batches carry extra keys (speaker_ids) that HF would drop by default
        remove_unused_columns=False if args.mode != "standard" else True,
    )

    # ── Trainer ───────────────────────────────────────────────────
    if args.mode == "standard":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds_enc,
            eval_dataset=valid_ds_enc,
            processing_class=feature_extractor,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = DANNTrainer(
            dann_model_ref=model,
            input_features_key=input_features_key,
            args=training_args,
            train_dataset=train_ds_enc,
            eval_dataset=valid_ds_enc,
            processing_class=feature_extractor,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    # ── Train & evaluate ──────────────────────────────────────────
    print(f"Starting {args.mode} training...")
    trainer.train()
    print("Final evaluation:")
    results = trainer.evaluate()
    print(results)
    wandb.finish()

    # save the final model — HF .save_pretrained for standard, raw state_dict for DANN
    save_dir = f"{args.output_dir}/final_model"
    if args.mode == "standard":
        model.save_pretrained(save_dir)
    else:
        torch.save(model.state_dict(), f"{save_dir}/dann_model.pt")
    feature_extractor.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
期刊版: shared/llm/qlora_train.py のコピーに --seed と train_meta.json 出力を追加したもの(他は同一)。
QLoRA (4-bit base + LoRA) による臨床NERのSFT。venv: /opt/llm/ft-venv

使い方:
    /opt/llm/ft-venv/bin/python qlora_train.py \
        --model llm-jp/llm-jp-3.1-1.8b-instruct4 \
        --data_dir ./sft_data --out_dir ./ft/llmjp-1.8b \
        --epochs 3 --lr 2e-4
"""
import os
import argparse
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", default="./sft_data")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42, help="SFTConfig の seed(既定 42 = LAU 論文と同一)")
    p.add_argument("--save_steps", type=int, default=0,
                   help=">0 なら N step ごとに checkpoint を保存し、out_dir に checkpoint-* があれば自動で再開(停電対策、70B 用)")
    p.add_argument("--device_map", default="single",
                   help="single=単一GPU({'':0}) / auto=複数GPUにモデル並列(70B等)")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    dev_map = {"": 0} if args.device_map == "single" else args.device_map
    import fcntl, time as _t
    _lock = open("/opt/llm/.elements_load.lock", "w"); print("waiting for load lock...", flush=True)
    fcntl.flock(_lock, fcntl.LOCK_EX); _t0 = _t.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map=dev_map,  # single=単一GPU / auto=モデル並列
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    fcntl.flock(_lock, fcntl.LOCK_UN); _lock.close(); print(f"model loaded in {_t.time()-_t0:.0f}s", flush=True)
    model.config.use_cache = False

    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])

    ds = load_dataset("json", data_files={
        "train": os.path.join(args.data_dir, "sft_train.jsonl"),
        "val": os.path.join(args.data_dir, "sft_val.jsonl")})

    # messages → chat template適用済みの "text" 列に変換(trl0.14の自動変換に頼らない)
    def to_text(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
    ds = ds.map(to_text, remove_columns=["messages"])

    cfg = SFTConfig(
        output_dir=args.out_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=20, save_total_limit=1,
        **({"save_strategy": "steps", "save_steps": args.save_steps} if args.save_steps else {"save_strategy": "epoch"}),
        eval_strategy="no",  # 学習中の評価は無効(epoch境界でOOMするため)。評価は別途テスト集で
        bf16=True, max_seq_length=args.max_seq_len, packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(model=model, args=cfg, peft_config=lora,
                         train_dataset=ds["train"], processing_class=tok)
    import json, time
    t0 = time.time()
    import glob
    ckpts = sorted(glob.glob(os.path.join(args.out_dir, "checkpoint-*")), key=lambda d: int(d.rsplit("-", 1)[1])) if args.save_steps else []
    ckpts = [c for c in ckpts if os.path.exists(os.path.join(c, "trainer_state.json"))]
    if ckpts:
        print(f"resuming from {ckpts[-1]}", flush=True)
        trainer.train(resume_from_checkpoint=ckpts[-1])
    else:
        trainer.train()
    json.dump({"model": args.model, "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
               "r": args.r, "alpha": args.alpha, "max_seq_len": args.max_seq_len,
               "batch": args.batch, "grad_accum": args.grad_accum, "device_map": args.device_map,
               "train_sec": round(time.time() - t0, 1), "n_train": len(ds["train"]),
               "log_history": trainer.state.log_history},
              open(os.path.join(args.out_dir, "train_meta.json"), "w"), indent=1)
    trainer.save_model(args.out_dir)  # LoRAアダプタのみ保存(tokenizerは保存しない:
    # 評価側の古いtransformersで読めない不整合を避ける。評価はbaseのtokenizerを使う)
    print(f"LoRA adapter saved to {args.out_dir}")


if __name__ == "__main__":
    main()

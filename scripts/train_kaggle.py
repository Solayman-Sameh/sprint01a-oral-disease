"""Two-process PyTorch experiment runner, embedded in the Kaggle Run All notebook."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import models, transforms


VERSION = "oral-ddp-20261029-v2"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class PadSquare:
    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        left, top = (side - width) // 2, (side - height) // 2
        return ImageOps.expand(image, (left, top, side-width-left, side-height-top), fill=0)


class OralCNN(nn.Module):
    def __init__(self, classes=6, dropout=0.3):
        super().__init__()
        blocks = []
        incoming = 3
        for outgoing in [32, 64, 128]:
            blocks += [nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False), nn.BatchNorm2d(outgoing),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            incoming = outgoing
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def create_model(name, dropout, weights_spec, pretrained=True):
    if name == "custom_cnn":
        return OralCNN(dropout=dropout)
    model = models.get_model(name, weights=None)
    if pretrained:
        state = torch.load(weights_spec[name]["path"], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    if name.startswith("resnet"):
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 6))
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 6)
        for layer in model.classifier.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = dropout
    return model


def image_transform(name, weights_spec, image_size, training):
    if name == "custom_cnn":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        interpolation = transforms.InterpolationMode.BILINEAR
    else:
        recipe = models.get_weight(weights_spec[name]["weights_enum"]).transforms()
        mean, std, interpolation = recipe.mean, recipe.std, recipe.interpolation
    # Adapt geometry to this task while preserving each weight's normalization/interpolation.
    steps = [PadSquare(), transforms.Resize((image_size, image_size), interpolation=interpolation)]
    if training:
        steps += [transforms.RandomHorizontalFlip(0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps), {"size": image_size, "mean": mean, "std": std,
                                      "interpolation": str(interpolation), "geometry": "pad_square_black_then_resize"}


class OralDataset(Dataset):
    def __init__(self, frame, root, transform):
        self.paths = frame["path"].tolist()
        self.labels = frame["class_idx"].astype(int).tolist()
        self.root, self.transform = Path(root), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root / self.paths[index]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), self.labels[index]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(frame, root, transform, batch, workers, device, sampler=None, seed=42):
    options = dict(batch_size=batch, num_workers=workers, pin_memory=device.type == "cuda",
                   sampler=sampler, shuffle=False, worker_init_fn=seed_worker,
                   generator=torch.Generator().manual_seed(seed))
    if workers:
        options.update(persistent_workers=True, multiprocessing_context="spawn", prefetch_factor=2)
    return DataLoader(OralDataset(frame, root, transform), **options)


def evaluate(model, batches, device):
    model.eval()
    truths, probabilities, total_loss, count = [], [], 0.0, 0
    with torch.inference_mode():
        for images, labels in batches:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # Score in FP32: a few heavily processed inputs can overflow FP16 evaluation activations.
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("Non-finite evaluation logits")
            total_loss += loss.item()
            count += len(labels)
            truths.extend(labels.cpu().tolist())
            probabilities.extend(logits.float().softmax(1).cpu().tolist())
    predictions = np.asarray(probabilities).argmax(axis=1)
    metrics = {"loss": total_loss/count, "accuracy": float(accuracy_score(truths, predictions)),
               "macro_f1": float(f1_score(truths, predictions, labels=list(range(6)), average="macro", zero_division=0))}
    return metrics, truths, predictions, probabilities


def make_optimizer(model, name, trial):
    if name == "custom_cnn":
        groups = [{"params": model.parameters(), "lr": trial["lr"]}]
    else:
        head = model.fc if name.startswith("resnet") else model.classifier
        head_ids = {id(p) for p in head.parameters()}
        groups = [{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": trial["lr"] * 0.1},
                  {"params": list(head.parameters()), "lr": trial["lr"]}]
    return torch.optim.AdamW(groups, weight_decay=trial["weight_decay"])


def synchronize(value, rank, world):
    if world == 1:
        return value
    message = [value if rank == 0 else None]
    dist.broadcast_object_list(message, src=0)
    return message[0]


def barrier(world):
    if world > 1:
        dist.barrier()


def run_trial(config, trial, frame, device, rank, world, deadline, preflight=False):
    name = trial["model"]
    run_dir = Path(config["output"]) / trial["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps({"version": VERSION, "trial": trial,
        "manifest": config["manifest_sha256"], "image_size": config["image_size"],
        "classes": config["classes"], "weights": {k:v["sha256"] for k,v in config["weights"].items()}}, sort_keys=True).encode()).hexdigest()
    completed = None
    if rank == 0 and not preflight and (run_dir / "result.json").is_file():
        completed = json.loads((run_dir / "result.json").read_text())
        if completed["signature"] != signature:
            raise RuntimeError(f"Existing output configuration differs: {run_dir}; use a new experiment directory.")
    completed = synchronize(completed, rank, world)
    if completed is not None:
        if rank == 0:
            print("Already completed:", trial["id"], flush=True)
        return completed, False

    seed_all(config["seed"] + rank)
    model = create_model(name, trial["dropout"], config["weights"]).to(device)
    optimizer = make_optimizer(model, name, trial)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial["epochs"])
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=device.type == "cuda")
    train_frame = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    val_frame = frame.loc[frame["split"].eq("validation")].reset_index(drop=True)
    train_transform, preprocessing = image_transform(name, config["weights"], config["image_size"], True)
    val_transform, _ = image_transform(name, config["weights"], config["image_size"], False)
    sampler = DistributedSampler(train_frame, num_replicas=world, rank=rank, shuffle=True, seed=config["seed"])
    batch_size = trial["batch_size_per_gpu"]
    train_batches = loader(train_frame, config["data_root"], train_transform, batch_size, config["workers_per_rank"], device, sampler, config["seed"]+rank)
    # Rank zero evaluates the entire non-padded validation set; no distributed evaluation duplicates.
    val_batches = loader(val_frame, config["data_root"], val_transform, batch_size, 0, device) if rank == 0 else None
    counts = train_frame["class_idx"].value_counts().reindex(range(6), fill_value=0).to_numpy()
    if not (counts > 0).all():
        raise ValueError("A training class is missing")
    weights = torch.tensor(len(train_frame)/(6*counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    start_epoch, best, stale, history, previous_seconds = 0, -1.0, 0, [], 0.0
    latest_path = run_dir / "latest.pt"
    if latest_path.is_file() and not preflight:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
        if checkpoint["signature"] != signature:
            raise RuntimeError("Resume signature differs; choose a new output directory")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best, stale = checkpoint["epoch"]+1, checkpoint["best"], checkpoint["stale"]
        history, previous_seconds = checkpoint["history"], checkpoint["training_seconds"]
        if stale >= config["patience"]:
            start_epoch = trial["epochs"]
    if world > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    base = model.module if isinstance(model, DDP) else model
    start_time = time.perf_counter()
    interrupted = False
    epochs = 1 if preflight else trial["epochs"]
    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        seed_all(config["seed"] + epoch*world + rank)
        sampler.set_epoch(epoch)
        model.train()
        total_loss, total_count, optimizer_updates = 0.0, 0, 0
        probe = base.fc[-1].weight if name.startswith("resnet") else base.classifier[-1].weight
        probe_before = probe.detach().clone() if preflight else None
        for batch_index, (images, labels) in enumerate(train_batches):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            for attempt in range(8 if preflight else 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Non-finite loss in {trial['id']}")
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                accepted = scaler.get_scale() >= previous_scale
                optimizer_updates += int(accepted)
                if accepted:
                    break
            total_loss += loss.item() * len(labels)
            total_count += len(labels)
            if preflight or (config.get("debug_max_batches") and batch_index+1 >= config["debug_max_batches"]):
                break
        totals = torch.tensor([total_loss, total_count, optimizer_updates], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(totals)
        barrier(world)
        if totals[2].item() == 0:
            raise FloatingPointError("No optimizer update succeeded; reduce the initial AMP scale or inspect gradients before training.")
        if preflight:
            if torch.equal(probe_before, probe.detach()):
                raise RuntimeError("Preflight did not change the classifier weights")
            base.eval()
            with torch.inference_mode():
                check_logits = base(images)
            if not torch.isfinite(check_logits).all().item():
                raise FloatingPointError("Non-finite FP32 logits after the preflight optimizer update")
            if rank == 0:
                print("PREFLIGHT PASS:", name, "world_size", world, "classifier weights changed; finite loss", float(totals[0]/totals[1]), flush=True)
            break
        if rank == 0:
            validation, _, _, _ = evaluate(base, val_batches, device)
            improved = validation["macro_f1"] > best + 1e-6
            if improved:
                best, stale = validation["macro_f1"], 0
                save_checkpoint(run_dir / "best.pt", {"state_dict":base.state_dict(), "signature":signature,
                    "trial":trial, "class_to_idx":{name:i for i,name in enumerate(config["classes"])},
                    "preprocessing":preprocessing, "validation":validation, "epoch":epoch+1,
                    "manifest_sha256":config["manifest_sha256"], "weights":config["weights"].get(name), "version":VERSION})
            else:
                stale += 1
            scheduler.step()
            row = {"epoch":epoch+1,"train_loss":float(totals[0]/totals[1]),"validation_loss":validation["loss"],
                   "validation_accuracy":validation["accuracy"],"validation_macro_f1":validation["macro_f1"],
                   "optimizer_updates":int(totals[2].item()/world),
                   "epoch_seconds":time.perf_counter()-epoch_start}
            history.append(row)
            save_checkpoint(latest_path, {"state_dict":base.state_dict(),"signature":signature,"optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"best":best,"stale":stale,
                "history":history,"training_seconds":previous_seconds+time.perf_counter()-start_time})
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            interrupted = time.monotonic() >= deadline and stale < config["patience"] and epoch+1 < epochs
            stop = stale >= config["patience"] or interrupted
            print(trial["id"], json.dumps(row), "[checkpoint saved]", flush=True)
            state = [stop,interrupted,best,stale]
        else:
            scheduler.step()
            state = None
        stop, interrupted, best, stale = synchronize(state, rank, world)
        if stop:
            break
    barrier(world)
    if preflight:
        return None, False
    result = None
    if rank == 0 and not interrupted:
        checkpoint = torch.load(run_dir / "best.pt",map_location="cpu",weights_only=True)
        result = {"id":trial["id"],"model":name,"signature":signature,"trial":trial,
                  "best_validation_macro_f1":best,"best_epoch":checkpoint["epoch"],
                  "validation_accuracy":checkpoint["validation"]["accuracy"],
                  "epochs_run":len(history),"training_seconds":previous_seconds+time.perf_counter()-start_time,
                  "parameters":sum(p.numel() for p in base.parameters()),"best_checkpoint":str(run_dir / "best.pt")}
        write_json(run_dir / "result.json",result)
    result = synchronize(result,rank,world)
    return result, interrupted


def final_reports(config, results, frame, device):
    output = Path(config["output"])
    trials = pd.DataFrame([{k:v for k,v in result.items() if k != "trial"} for result in results])
    trials.to_csv(output / "all_validation_trials.csv",index=False)
    winners = [max([r for r in results if r["model"] == name], key=lambda r:r["best_validation_macro_f1"])
               for name in dict.fromkeys(trial["model"] for trial in config["trials"])]
    winner = max(winners,key=lambda result:result["best_validation_macro_f1"])
    # This decision is recorded before looking at test predictions.
    write_json(output / "selected_model.json",{"criterion":"validation_macro_f1","trial":winner["id"],
               "model":winner["model"],"best_checkpoint":winner["best_checkpoint"],"selected_before_test":True})
    rows = []
    test_frame = frame.loc[frame["split"].eq("test")].reset_index(drop=True)
    for result in winners:
        checkpoint = torch.load(result["best_checkpoint"],map_location="cpu",weights_only=True)
        cache_file = output / (result["id"]+"_test_metrics.json")
        cache_signature = hashlib.sha256(Path(result["best_checkpoint"]).read_bytes()).hexdigest()
        if cache_file.is_file():
            saved = json.loads(cache_file.read_text())
            if saved["checkpoint_sha256"] == cache_signature:
                rows.append(saved)
                continue
        model = create_model(result["model"],result["trial"]["dropout"],config["weights"],pretrained=False).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        transform, _ = image_transform(result["model"],config["weights"],config["image_size"],False)
        batches = loader(test_frame,config["data_root"],transform,32,0,device)
        metrics, truths, predicted, probabilities = evaluate(model,batches,device)
        report = classification_report(truths,predicted,labels=list(range(6)),target_names=config["classes"],zero_division=0,output_dict=True)
        pd.DataFrame(report).T.to_csv(output / (result["id"]+"_per_class.csv"))
        pd.DataFrame(confusion_matrix(truths,predicted,labels=list(range(6))),index=config["classes"],columns=config["classes"]).to_csv(output / (result["id"]+"_confusion_matrix.csv"))
        predictions = test_frame[["path","class","class_idx","family_id"]].copy()
        predictions["prediction_idx"] = predicted
        for index,label in enumerate(config["classes"]):
            predictions["score_"+label] = np.asarray(probabilities)[:,index]
        predictions.to_csv(output / (result["id"]+"_test_predictions.csv"),index=False)
        model.eval()
        dummy = torch.zeros(1,3,config["image_size"],config["image_size"],device=device)
        times = []
        with torch.inference_mode():
            for index in range(35):
                if device.type == "cuda": torch.cuda.synchronize(device)
                start = time.perf_counter()
                model(dummy)
                if device.type == "cuda": torch.cuda.synchronize(device)
                if index >= 10: times.append((time.perf_counter()-start)*1000)
        saved = {"model":result["model"],"trial":result["id"],"validation_macro_f1":result["best_validation_macro_f1"],
                 "test_macro_f1":metrics["macro_f1"],"test_accuracy":metrics["accuracy"],
                 "parameters":result["parameters"],"training_seconds":result["training_seconds"],
                 "checkpoint_MB":Path(result["best_checkpoint"]).stat().st_size/1e6,
                 "batch1_model_only_median_ms":float(np.median(times)),"batch1_model_only_p95_ms":float(np.percentile(times,95)),
                 "latency_device":str(device),"checkpoint_sha256":cache_signature}
        saved["evaluation_and_latency_precision"] = "fp32"
        write_json(cache_file,saved)
        rows.append(saved)
        del model,checkpoint
        if device.type == "cuda": torch.cuda.empty_cache()
    pd.DataFrame(rows).sort_values("validation_macro_f1",ascending=False).to_csv(output / "model_comparison.csv",index=False)
    write_json(output / "experiment_status.json",{"status":"COMPLETE","trials":len(results),"models":len(winners),
               "selection":"validation only; test reports are final descriptive comparisons","version":VERSION})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--stage",choices=["preflight","train"],default="train")
    parser.add_argument("--device",choices=["auto","cpu"],default="auto")
    parser.add_argument("--rendezvous-file", help="Optional file-store rendezvous for local distributed verification")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    rank, world, local_rank = int(os.getenv("RANK",0)),int(os.getenv("WORLD_SIZE",1)),int(os.getenv("LOCAL_RANK",0))
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    if world > 1:
        rendezvous = {"init_method":Path(args.rendezvous_file).resolve().as_uri(), "rank":rank,"world_size":world} if args.rendezvous_file else {}
        dist.init_process_group("nccl" if use_cuda else "gloo",timeout=datetime.timedelta(minutes=40),**rendezvous)
    try:
        manifest = Path(config["manifest"])
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != config["manifest_sha256"]:
            raise RuntimeError("Split manifest changed after configuration")
        frame = pd.read_csv(manifest)
        if not frame["pixel_sha256"].is_unique or frame.groupby("family_id")["split"].nunique().max() != 1:
            raise ValueError("Manifest uniqueness/group invariants failed")
        if set(frame["class"]) != set(config["classes"]):
            raise ValueError("Class map differs from manifest")
        for label,index in {label:i for i,label in enumerate(config["classes"])}.items():
            if not frame.loc[frame["class"].eq(label),"class_idx"].eq(index).all():
                raise ValueError("Class indices differ from configured class order")
        deadline = time.monotonic()+config.get("max_training_hours",9)*3600
        if args.stage == "preflight":
            unique = {trial["model"]:trial for trial in config["trials"]}
            for trial in unique.values():
                run_trial(config,trial,frame,device,rank,world,deadline,preflight=True)
            if rank == 0:
                write_json(Path(config["output"])/"ddp_preflight.json",{"passed":True,"world_size":world,"device":str(device),"models":list(unique)})
        else:
            results = []
            for trial in config["trials"]:
                result,interrupted = run_trial(config,trial,frame,device,rank,world,deadline)
                if interrupted:
                    if rank == 0:
                        write_json(Path(config["output"])/"experiment_status.json",{"status":"TIME_BUDGET_REACHED","resume":"Rerun with this output directory preserved; latest checkpoints resume completed epochs."})
                    return
                results.append(result)
                barrier(world)
                if use_cuda: torch.cuda.empty_cache()
            if rank == 0:
                final_reports(config,results,frame,device)
                print("ALL MODELS COMPLETE. See model_comparison.csv and selected_model.json",flush=True)
            barrier(world)
    finally:
        if world > 1 and dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    main()

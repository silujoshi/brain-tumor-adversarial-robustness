import time
import torch
import torchattacks
from src.evaluate import evaluate_model

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)

def normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std

def get_attack_instance(model, name, eps):
    if name == "FGSM":
        return torchattacks.FGSM(model, eps=eps)
    elif name == "MI-FGSM":
        return torchattacks.MIFGSM(model, eps=eps, alpha=2/255, steps=10, decay=1.0)
    elif name == "NIFGSM":
        return torchattacks.NIFGSM(model, eps=eps, alpha=2/255, steps=10, decay=1.0)
    elif name == "RFGSM":
        return torchattacks.RFGSM(model, eps=eps, alpha=2/255, steps=10)
    elif name == "PGDL2":
        return torchattacks.PGDL2(model, eps=eps * 2.0, steps=10) 
    elif name == "PGD":
        return torchattacks.PGD(model, eps=eps, alpha=2/255, steps=40, random_start=True)
    else:
        raise ValueError(f"Unknown attack name: {name}")

def evaluate_attacks_with_eps(model, test_loader, classes, device, eps_list):
    attack_names = ["FGSM", "MI-FGSM", "NIFGSM", "RFGSM", "PGDL2", "PGD"]
    results = {}
    sample_cache = {}

    # 1. Evaluate Clean Data Baseline & Cache First Sample + Clean Prediction
    print("\n--- Evaluating on Clean Test Data ---")
    clean_preds, clean_targets = [], []
    first_batch_captured = False
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            
            if not first_batch_captured:
                single_img = inputs[0:1]
                single_label = labels[0].item()
                sample_cache["true_label"] = single_label
                sample_cache["Clean_img"] = single_img.detach().cpu()
                
                # Get clean prediction for this sample
                out = model(single_img)
                probs = torch.softmax(out, dim=1)
                conf, pred = torch.max(probs, dim=1)
                sample_cache["Clean_pred"] = pred.item()
                sample_cache["Clean_conf"] = conf.item() * 100
                first_batch_captured = True
                
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            clean_preds.extend(predicted.cpu().numpy())
            clean_targets.extend(labels.cpu().numpy())
            
    clean_metrics = evaluate_model(clean_targets, clean_preds, classes)
    results["Clean"] = {"time_seconds": 0.0, "metrics": clean_metrics}

    # 2. Loop Through Attacks & Epsilons
    for name in attack_names:
        for eps in eps_list:
            experiment_key = f"{name}_eps_{eps:.4f}"
            print(f"\n--- Running {name} Attack (Epsilon: {eps:.4f}) ---")
            
            attack = get_attack_instance(model, name, eps)
            adv_preds, adv_targets = [], []
            
            start_time = time.time()
            is_first_batch = True
            
            for batch in test_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                inputs = denormalize(inputs)
                adv_inputs = attack(inputs, labels)
                adv_inputs = normalize(adv_inputs)
                
                # Cache the adversarial version and its model prediction for the single sample
                if is_first_batch:
                    single_adv = adv_inputs[0:1]
                    sample_cache[f"{experiment_key}_img"] = single_adv.detach().cpu()
                    
                    with torch.no_grad():
                        out = model(single_adv)
                        probs = torch.softmax(out, dim=1)
                        conf, pred = torch.max(probs, dim=1)
                        sample_cache[f"{experiment_key}_pred"] = pred.item()
                        sample_cache[f"{experiment_key}_conf"] = conf.item() * 100
                    is_first_batch = False
                
                with torch.no_grad():
                    outputs = model(adv_inputs)
                    _, predicted = outputs.max(1)
                    
                adv_preds.extend(predicted.cpu().numpy())
                adv_targets.extend(labels.cpu().numpy())
                
            elapsed_time = time.time() - start_time
            print(f"[{experiment_key}] Generation Time: {elapsed_time:.4f} seconds")
            
            metrics = evaluate_model(adv_targets, adv_preds, classes)
            results[experiment_key] = {
                "attack": name,
                "epsilon": eps,
                "time_seconds": elapsed_time,
                "metrics": metrics
            }

    return results, sample_cache
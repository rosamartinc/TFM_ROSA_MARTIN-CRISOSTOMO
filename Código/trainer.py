
import numpy as np
import torch
from torch import nn
from metrics_ecg import compute_metrics_from_logits

def train(model, train_loader, valid_loader, fold_idx, out_dir, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    lr           = float(cfg.get('lr', 1e-3))
    weight_decay = float(cfg.get('weight_decay', 0.0))
    epochs       = int(cfg.get('epochs', 3))
    patience     = int(cfg.get('patience', 5))
    thresh       = float(cfg.get('threshold', 0.5))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    best_f1 = -1.0
    wait = 0
    model_tag = out_dir.name.replace('results_', '')

    for epoch in range(1, epochs + 1):
        # TRAIN
        model.train()
        total_loss, total = 0.0, 0
        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            if step % 5 == 0 or step == len(train_loader):
                print(f"[Epoch {epoch:02d}] Step {step}/{len(train_loader)} | Loss={loss.item():.4f}")
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
       
        train_loss = total_loss / max(1, total)

        # VALID
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device); yb = yb.to(device)
                all_logits.append(model(xb).cpu())
                all_targets.append(yb.cpu())

        logits  = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        m = compute_metrics_from_logits(logits, targets, thresh=thresh)

        # Evitar NaN 
        step_value = m["f1_macro"] if np.isfinite(m["f1_macro"]) else 0.0
        scheduler.step(step_value)

        auc_print = m['auroc_macro'] if np.isfinite(m['auroc_macro']) else float('nan')
        ap_print  = m['ap_macro']    if np.isfinite(m['ap_macro'])    else float('nan')
        print(f"[Fold {fold_idx}] Epoch {epoch:02d} | loss={train_loss:.4f} | "
              f"F1={m['f1_macro']:.3f} P={m['precision_macro']:.3f} R={m['recall_macro']:.3f} "
              f"AUC={auc_print:.3f} AP={ap_print:.3f}")

        # EARLY STOP y SAVE BEST 
        if np.isfinite(m["f1_macro"]) and (m["f1_macro"] > best_f1 + 1e-6):
            best_f1 = m["f1_macro"]; wait = 0
            torch.save({'model_state': model.state_dict()}, out_dir / f'{model_tag}_best_foldfixed.pt')
        else:
            wait += 1
            if wait >= patience:
                print(f"[Fold {fold_idx}] Early stopping.")
                break

    all_logits, all_targets = [], []
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.to(device)
            all_logits.append(model(xb).cpu())
            all_targets.append(yb)
    logits  = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return compute_metrics_from_logits(logits, targets, thresh=thresh)

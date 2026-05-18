"""Étape 6 — Entraînement

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 6 — Entraînement optimisé CPU
# ══════════════════════════════════════════════════════════
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time

# ── Paramètre clé : réduire pour aller plus vite ─────────
NUM_EPOCHS = 20   # Mettre 3 pour un test rapide, 20 pour un vrai entraînement

def train_model(model, model_name, num_epochs=NUM_EPOCHS):
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    best_weights = None
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_path = MODELS_DIR / f'{model_name.lower().replace("-","_")}_best.pth'

    for epoch in range(1, num_epochs + 1):
        t_start = time.time()

        # ── Train ──────────────────────────────────────────
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch}/{num_epochs} Train',
                    leave=False, ncols=90)
        for imgs, lbls in pbar:
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == lbls).sum().item()
            total    += imgs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        t_loss = run_loss / total
        t_acc  = correct  / total

        # ── Validation ─────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc='Val', leave=False, ncols=70):
                out       = model(imgs)
                v_loss   += criterion(out, lbls).item() * imgs.size(0)
                v_correct += (out.argmax(1) == lbls).sum().item()
                v_total  += imgs.size(0)
        v_loss = v_loss / v_total
        v_acc  = v_correct / v_total

        scheduler.step()
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)

        elapsed = time.time() - t_start
        is_best = v_acc > best_val_acc

        if is_best:
            best_val_acc = v_acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            # Sauvegarde intermédiaire — protège contre crash ou coupure
            torch.save(best_weights, best_path)

        marker = ' ✅' if is_best else ''
        print(f'[{model_name}] Epoch {epoch:>2}/{num_epochs} | '
              f'Train {t_acc:.3f}/{t_loss:.3f} | '
              f'Val {v_acc:.3f}/{v_loss:.3f} | '
              f'{elapsed:.0f}s{marker}')

    model.load_state_dict(best_weights)
    print(f'\n🏆 {model_name} — Meilleure Val Acc : {best_val_acc:.4f}')
    print(f'   Sauvegarde intermédiaire : {best_path}\n')
    return model, history

# ── Lancer les entraînements ─────────────────────────────
print('=' * 65)
print('ENTRAÎNEMENT EfficientNet-B0')
print('=' * 65)
eff_model, eff_history = train_model(eff_model, 'EfficientNet-B0')

print('=' * 65)
print('ENTRAÎNEMENT ResNet-50')
print('=' * 65)
res_model, res_history = train_model(res_model, 'ResNet-50')
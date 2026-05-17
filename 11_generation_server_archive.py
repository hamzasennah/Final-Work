"""Étape 11 — Génération de server.py

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 11 — Générer server.py dans le dossier du projet
# ══════════════════════════════════════════════════════════

server_code = f'''# server.py — AgroShield
# Lancer : python server.py
# Puis ouvrir index.html dans le navigateur

from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import json, io, os, time, random, datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ── Chemins ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
EFF_PTH    = MODELS_DIR / "efficientnet_b0.pth"
RES_PTH    = MODELS_DIR / "resnet50.pth"
META_PATH  = MODELS_DIR / "efficientnet_b0_meta.json"

# ── Chargement métadonnées ───────────────────────────────
with open(META_PATH) as f:
    META = json.load(f)

CLASSES     = META["classes"]
NUM_CLASSES = len(CLASSES)
print(f"{{NUM_CLASSES}} classes chargées")

# ── Chargement modèles ───────────────────────────────────
def load_efficientnet(path, n):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()

def load_resnet50(path, n):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()

print("Chargement des modèles (quelques secondes)...")
eff_model = load_efficientnet(EFF_PTH, NUM_CLASSES)
res_model = load_resnet50(RES_PTH, NUM_CLASSES)
print("✅ Modèles prêts")

# ── Preprocessing ────────────────────────────────────────
preprocess = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=META["mean"], std=META["std"]),
])

def predict(pil_image):
    tensor = preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        eff_probs = torch.softmax(eff_model(tensor), dim=1)[0]
        res_probs = torch.softmax(res_model(tensor), dim=1)[0]
    eff_top    = eff_probs.topk(3)
    res_top    = res_probs.topk(3)
    eff_pred   = CLASSES[eff_top.indices[0].item()]
    res_pred   = CLASSES[res_top.indices[0].item()]
    eff_conf   = float(eff_top.values[0])
    res_conf   = float(res_top.values[0])
    agree      = (eff_pred == res_pred)
    final_pred = eff_pred if eff_conf >= res_conf else res_pred
    final_conf = max(eff_conf, res_conf)
    parts      = final_pred.split("__")
    plant      = parts[0].replace("_", " ")
    disease    = parts[1].replace("_", " ") if len(parts) > 1 else "Healthy"
    top3_eff   = [{{"class": CLASSES[i].replace("__"," — ").replace("_"," "),
                   "confidence": float(eff_probs[i])}}
                 for i in eff_top.indices.tolist()]
    top3_res   = [{{"class": CLASSES[i].replace("__"," — ").replace("_"," "),
                   "confidence": float(res_probs[i])}}
                 for i in res_top.indices.tolist()]
    return {{
        "efficientnet"    : {{"prediction": eff_pred, "confidence": eff_conf, "top3": top3_eff}},
        "resnet"          : {{"prediction": res_pred, "confidence": res_conf, "top3": top3_res}},
        "agreement"       : agree,
        "agreement_score" : "2/2" if agree else "1/2",
        "plant"           : plant,
        "disease"         : disease,
        "confidence"      : final_conf,
        "is_healthy"      : "healthy" in final_pred.lower(),
    }}

# ── Capteurs simulés ─────────────────────────────────────
def get_sensors():
    # Sur Raspberry Pi : remplacer par lecture GPIO réelle
    return {{
        "temperature" : round(random.uniform(22, 34), 1),
        "humidity"    : round(random.uniform(45, 82), 1),
        "luminosity"  : round(random.uniform(150, 850), 1),
        "servo1_pos"  : "A" if random.random() > 0.7 else "B",
        "servo2_pos"  : "B",
        "servo3_pos"  : "A" if random.random() > 0.6 else "B",
        "mode"        : "simulation",
        "last_update" : time.time(),
    }}

# ── Base de données en mémoire ───────────────────────────
detections_db = []

def calc_spread(result):
    if result["is_healthy"] or result["confidence"] < 0.7:
        return 0
    HIGH = ["Late_blight", "Early_blight", "Bacterial_spot", "Yellow_Leaf_Curl"]
    return 10 if any(d in result["disease"] for d in HIGH) else 5

# ── Endpoints ────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({{"status": "ok", "mode": "simulation_pc", "timestamp": time.time()}})

@app.route("/api/sensors", methods=["GET"])
def sensors():
    data = get_sensors()
    data["alerts"] = {{
        "temperature": data["temperature"] > 28.0,
        "humidity"   : data["humidity"]    > 75.0,
        "luminosity" : data["luminosity"]  > 600.0,
    }}
    return jsonify(data)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({{"error": "Pas d\'image"}}), 400
    file    = request.files["image"]
    zone_id = request.form.get("zone_id", f"zone_{{int(time.time())}}") 
    lat     = float(request.form.get("lat", 33.5731))
    lng     = float(request.form.get("lng", -7.5898))
    try:
        img    = Image.open(io.BytesIO(file.read())).convert("RGB")
        result = predict(img)
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500
    spread = calc_spread(result)
    det = {{
        "zone_id"             : zone_id,
        "lat"                 : lat,
        "lng"                 : lng,
        "timestamp"           : datetime.datetime.now().isoformat(),
        "plant"               : result["plant"],
        "disease"             : result["disease"],
        "confidence"          : result["confidence"],
        "is_healthy"          : result["is_healthy"],
        "agreement_score"     : result["agreement_score"],
        "spread_radius_meters": spread,
    }}
    detections_db.append(det)
    return jsonify({{**result, "zone": det}})

@app.route("/api/disease_map", methods=["GET"])
def disease_map():
    return jsonify({{
        "detections"    : detections_db,
        "total_scanned" : len(detections_db),
        "infected_zones": sum(1 for d in detections_db if not d["is_healthy"]),
        "sensors"       : get_sensors(),
    }})

@app.route("/api/detections/clear", methods=["POST"])
def clear():
    detections_db.clear()
    return jsonify({{"success": True}})

if __name__ == "__main__":
    print("Serveur AgroShield démarré sur http://localhost:5000")
    print("Ouvrez index.html dans votre navigateur")
    print("Ctrl+C pour arrêter")
    app.run(host="0.0.0.0", port=5000, debug=False)
'''

server_path = PROJECT_DIR / 'server.py'
with open(server_path, 'w', encoding='utf-8') as f:
    f.write(server_code)

print(f'✅ server.py généré : {server_path}')
print()
print('Pour lancer le serveur :')
print(f'  cd {PROJECT_DIR}')
print(f'  python server.py')
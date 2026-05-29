# server.py - AgroShield dashboard
# Run: python server.py
# Open from this computer: http://localhost:5000
# Open from another device on the same network: http://<computer-ip>:5000

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import deque
import datetime
import io
import json
import random
import time

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
except Exception as exc:
    torch = None
    nn = None
    models = None
    T = None
    TORCH_IMPORT_ERROR = str(exc)
else:
    TORCH_IMPORT_ERROR = None

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent
RECEIVED_IMAGES_DIR = Path(r"C:\Users\pc\Desktop\reception des images")
ALERT_CHANNELS = {
    "email": {"enabled": False, "provider": "SMTP", "status": "simulation - config SMTP non renseignee"},
    "sms": {"enabled": False, "provider": "Twilio", "status": "simulation - identifiants Twilio non renseignes"},
}
MODELS_DIR = BASE_DIR / "models"
EFF_PTH_CANDIDATES = [
    MODELS_DIR / "efficientnet_b0.pth",
    MODELS_DIR / "efficientnet_b0_best.pth",
]
RES_PTH_CANDIDATES = [
    MODELS_DIR / "resnet_50.pth",
    MODELS_DIR / "resnet50.pth",
    MODELS_DIR / "resnet_50_best.pth",
]
META_CANDIDATES = [
    MODELS_DIR / "efficientnet_b0_meta.json",
    MODELS_DIR / "resnet_50_meta.json",
]

MODEL_STATUS = {
    "available": False,
    "efficientnet": False,
    "resnet": False,
    "message": "Mode demonstration: ajoutez le dossier models pour activer l'IA.",
}

DEFAULT_META = {
    "classes": [
        "Tomato__healthy",
        "Tomato__Late_blight",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Pepper__bell___healthy",
        "Pepper__bell___Bacterial_spot",
        "Potato__Early_blight",
    ],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

META = DEFAULT_META
CLASSES = META["classes"]
NUM_CLASSES = len(CLASSES)
eff_model = None
res_model = None
preprocess = None
USE_RESNET = False


def load_efficientnet(path, n):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def load_resnet50(path, n):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def first_existing(paths):
    return next((path for path in paths if path.exists()), None)


def initialise_models():
    global META, CLASSES, NUM_CLASSES, eff_model, res_model, preprocess, USE_RESNET

    if torch is None:
        MODEL_STATUS["message"] = f"PyTorch indisponible: {TORCH_IMPORT_ERROR}"
        return
    meta_path = first_existing(META_CANDIDATES)
    eff_path = first_existing(EFF_PTH_CANDIDATES)
    res_path = first_existing(RES_PTH_CANDIDATES)
    if not meta_path or not eff_path:
        return

    try:
        with open(meta_path, encoding="utf-8") as meta_file:
            META = json.load(meta_file)
        CLASSES = META["classes"]
        NUM_CLASSES = len(CLASSES)
        preprocess = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=META["mean"], std=META["std"]),
            ]
        )
        eff_model = load_efficientnet(eff_path, NUM_CLASSES)
        MODEL_STATUS.update(
            {
                "available": True,
                "efficientnet": True,
                "message": f"{NUM_CLASSES} classes chargees.",
            }
        )
        if res_path:
            res_model = load_resnet50(res_path, NUM_CLASSES)
            USE_RESNET = True
            MODEL_STATUS["resnet"] = True
            MODEL_STATUS["message"] = "EfficientNet-B0 et ResNet-50 charges."
    except Exception as exc:
        MODEL_STATUS["message"] = f"Chargement des modeles impossible: {exc}"
        eff_model = None
        res_model = None
        USE_RESNET = False


initialise_models()

CROP_THRESHOLDS = {
    "tomato": {
        "label": "Tomate",
        "temperature_normal": "21-27 C",
        "humidity_normal": "60-75 %",
        "luminosity_normal": "450-700 lx",
        "precipitation_normal": "0-8 mm/h",
        "temp_ideal": "21-27 C",
        "humidity_ideal": "60-75 %",
        "temp_on": 29.0,
        "temp_off": 26.0,
        "humidity_on": 82.0,
        "humidity_off": 72.0,
        "rain_on": 18.0,
        "rain_off": 8.0,
        "luminosity_on": 720.0,
        "source_note": "Seuils inspires des recommandations horticoles tomates: stress thermique au-dessus de 29-30 C et risque fongique quand l'humidite reste elevee.",
    },
    "pepper": {
        "label": "Poivron",
        "temperature_normal": "21-28 C",
        "humidity_normal": "60-75 %",
        "luminosity_normal": "450-680 lx",
        "precipitation_normal": "0-7 mm/h",
        "temp_ideal": "21-28 C",
        "humidity_ideal": "60-75 %",
        "temp_on": 30.0,
        "temp_off": 27.0,
        "humidity_on": 80.0,
        "humidity_off": 70.0,
        "rain_on": 16.0,
        "rain_off": 7.0,
        "luminosity_on": 700.0,
        "source_note": "Poivron: culture sensible aux exces de chaleur et aux eclaboussures favorisant les taches bacteriennes.",
    },
    "potato": {
        "label": "Pomme de terre",
        "temperature_normal": "16-22 C",
        "humidity_normal": "70-85 %",
        "luminosity_normal": "350-580 lx",
        "precipitation_normal": "0-9 mm/h",
        "temp_ideal": "16-22 C",
        "humidity_ideal": "70-85 %",
        "temp_on": 26.0,
        "temp_off": 23.0,
        "humidity_on": 85.0,
        "humidity_off": 75.0,
        "rain_on": 20.0,
        "rain_off": 9.0,
        "luminosity_on": 600.0,
        "source_note": "Pomme de terre: culture de climat frais; chaleur et humidite prolongee augmentent le risque de maladies.",
    },
    "default": {
        "label": "Culture standard",
        "temperature_normal": "21-27 C",
        "humidity_normal": "60-80 %",
        "luminosity_normal": "350-650 lx",
        "precipitation_normal": "0-8 mm/h",
        "temp_ideal": "21-27 C",
        "humidity_ideal": "60-80 %",
        "temp_on": 29.0,
        "temp_off": 26.0,
        "humidity_on": 80.0,
        "humidity_off": 70.0,
        "rain_on": 18.0,
        "rain_off": 8.0,
        "luminosity_on": 650.0,
        "source_note": "Seuil generique utilise si aucune culture precise n'est selectionnee.",
    },
}

CROP_DECISION_MODELS = {
    "tomato": {
        "base_temp": 10.0,
        "sowing_date": "2026-03-15",
        "gdd_stages": [(350, "Reprise vegetative"), (650, "Floraison"), (1050, "Nouaison"), (1500, "Recolte proche")],
        "soil_optimum": 68,
    },
    "pepper": {
        "base_temp": 12.0,
        "sowing_date": "2026-03-20",
        "gdd_stages": [(320, "Reprise vegetative"), (700, "Floraison"), (1100, "Fructification"), (1600, "Recolte proche")],
        "soil_optimum": 65,
    },
    "potato": {
        "base_temp": 7.0,
        "sowing_date": "2026-03-10",
        "gdd_stages": [(280, "Emergence"), (650, "Tubérisation"), (1000, "Grossissement"), (1350, "Maturite")],
        "soil_optimum": 72,
    },
    "default": {
        "base_temp": 10.0,
        "sowing_date": "2026-03-15",
        "gdd_stages": [(320, "Croissance"), (650, "Floraison"), (1050, "Production"), (1450, "Recolte proche")],
        "soil_optimum": 68,
    },
}

system_state = {
    "current_crop": "default",
    "mode": "repos",
    "plate_angle": 0,
    "servo_angle": 0,
    "last_change": 0,
}
HYSTERESIS_DELAY = 8
sensor_history = deque(maxlen=30)
detections_db = []
latest_analysis = None
processed_received_images = set()
latest_raspberry_payload = {"timestamp": 0, "data": None}

ZONES = {
    "A": {"label": "Zone A", "x_cm": (0, 20), "actuator": "module_gauche"},
    "B": {"label": "Zone B", "x_cm": (20, 40), "actuator": "module_central"},
    "C": {"label": "Zone C", "x_cm": (40, 60), "actuator": "module_droit"},
}

ADJACENT_ZONES = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}

DISEASE_POLICIES = {
    "late blight": {
        "risk": 0.95,
        "spread": "tres elevee",
        "action": "Traiter A, B et C; isoler les plants tres atteints et limiter la mouillure foliaire.",
        "scope": "all",
        "weather": "cool_wet",
        "mechanical_bias": "pluie",
        "source": "Late blight is strongly favored by cool, wet and humid weather.",
    },
    "bacterial spot": {
        "risk": 0.86,
        "spread": "elevee",
        "action": "Traiter la zone detectee et les zones voisines; reduire les eclaboussures et l'humidite foliaire.",
        "scope": "adjacent_high_all",
        "weather": "warm_wet_splash",
        "mechanical_bias": "pluie",
        "source": "Bacterial spot is favored by warm wet weather and rain splash dispersal.",
    },
    "yellowleaf": {
        "risk": 0.82,
        "spread": "elevee",
        "action": "Surveiller A, B et C; controler les aleurodes vecteurs et reduire le stress thermique.",
        "scope": "all",
        "weather": "warm_vector",
        "mechanical_bias": "chaleur",
        "source": "Tomato yellow leaf curl virus is vectored by whiteflies, with pressure increasing in warm crops.",
    },
    "yellow leaf": {
        "risk": 0.82,
        "spread": "elevee",
        "action": "Surveiller A, B et C; controler les aleurodes vecteurs et reduire le stress thermique.",
        "scope": "all",
        "weather": "warm_vector",
        "mechanical_bias": "chaleur",
        "source": "Tomato yellow leaf curl virus is vectored by whiteflies, with pressure increasing in warm crops.",
    },
    "mosaic virus": {
        "risk": 0.78,
        "spread": "elevee",
        "action": "Isoler la zone detectee, desinfecter les manipulations et inspecter toute la culture.",
        "scope": "all_if_confident",
        "weather": "contact_vector",
        "mechanical_bias": "repos",
        "source": "Mosaic viruses spread mainly by contact or vectors; climate is treated as a secondary stress factor.",
    },
    "septoria": {
        "risk": 0.76,
        "spread": "moyenne a elevee",
        "action": "Traiter la zone et les demi-zones adjacentes; reduire la mouillure foliaire.",
        "scope": "adjacent",
        "weather": "wet_splash",
        "mechanical_bias": "pluie",
        "source": "Septoria leaf spot spreads by rain splash and is favored by prolonged leaf wetness.",
    },
    "leaf mold": {
        "risk": 0.74,
        "spread": "moyenne a elevee",
        "action": "Traiter la zone et les demi-zones adjacentes; ameliorer l'aeration et baisser l'humidite.",
        "scope": "adjacent",
        "weather": "humid",
        "mechanical_bias": "pluie",
        "source": "Tomato leaf mold is favored by high relative humidity.",
    },
    "target spot": {
        "risk": 0.73,
        "spread": "moyenne a elevee",
        "action": "Traiter la zone detectee et la demi-zone voisine la plus exposee.",
        "scope": "adjacent",
        "weather": "warm_wet_splash",
        "mechanical_bias": "pluie",
        "source": "Foliar spot diseases intensify when foliage remains wet and splash dispersal occurs.",
    },
    "early blight": {
        "risk": 0.70,
        "spread": "moyenne",
        "action": "Traiter la zone detectee; ajouter les demi-zones voisines si confiance forte ou climat humide.",
        "scope": "confidence_adjacent",
        "weather": "warm_wet_splash",
        "mechanical_bias": "pluie",
        "source": "Early blight is favored by warm, wet weather and extended leaf wetness.",
    },
    "spider mites": {
        "risk": 0.64,
        "spread": "moyenne",
        "action": "Traiter la zone detectee et inspecter les zones voisines en climat chaud et sec.",
        "scope": "confidence_adjacent",
        "weather": "hot_dry",
        "mechanical_bias": "chaleur",
        "source": "Spider mite pressure typically rises under hot and dry stress.",
    },
}

CLIMATE_IMPACT_BY_CLASS = {
    "Pepper__bell___Bacterial_spot": {
        "label": "Poivron - tache bacterienne",
        "heat": {"level": "modere a eleve", "bonus": 0.10, "trigger": "warm", "action": "Limiter le stress thermique; eviter que le feuillage reste humide sous chaleur."},
        "rain": {"level": "eleve", "bonus": 0.22, "trigger": "wet", "action": "Incliner les plaques, reduire les eclaboussures et proteger les zones voisines."},
        "mechanical": {"heat": "chaleur", "rain": "pluie"},
        "rationale": "Les taches bacteriennes du poivron/tomate sont favorisees par temps chaud et humide et par les eclaboussures de pluie.",
    },
    "Pepper__bell___healthy": {
        "label": "Poivron sain",
        "heat": {"level": "stress physiologique", "bonus": 0.00, "trigger": "heat", "action": "Proteger seulement si le seuil chaleur du poivron est depasse."},
        "rain": {"level": "stress hydrique", "bonus": 0.00, "trigger": "rain", "action": "Canaliser l'exces d'eau seulement si pluie intense."},
        "mechanical": {"heat": "chaleur", "rain": "pluie"},
        "rationale": "Aucune maladie detectee: le climat pilote la protection de culture, pas la propagation.",
    },
    "Potato___Early_blight": {
        "label": "Pomme de terre - alternariose",
        "heat": {"level": "modere", "bonus": 0.08, "trigger": "warm", "action": "Surveiller si chaleur accompagnee d'humidite foliaire."},
        "rain": {"level": "modere a eleve", "bonus": 0.16, "trigger": "wet", "action": "Incliner si pluie/humidite elevee; traiter la zone et demi-zones voisines si risque monte."},
        "mechanical": {"heat": "chaleur", "rain": "pluie"},
        "rationale": "L'alternariose est favorisee par chaleur moderee a elevee, humidite et mouillure foliaire.",
    },
    "Potato___Late_blight": {
        "label": "Pomme de terre - mildiou",
        "heat": {"level": "faible a modere", "bonus": 0.04, "trigger": "stress", "action": "La chaleur seule n'est pas le facteur principal; surveiller le stress."},
        "rain": {"level": "tres eleve", "bonus": 0.28, "trigger": "wet", "action": "Incliner immediatement vers le canal et traiter A, B et C."},
        "mechanical": {"heat": "repos", "rain": "pluie"},
        "rationale": "Le mildiou de la pomme de terre progresse fortement en conditions humides, pluvieuses et avec mouillure prolongee.",
    },
    "Potato___healthy": {
        "label": "Pomme de terre saine",
        "heat": {"level": "stress physiologique", "bonus": 0.00, "trigger": "heat", "action": "Proteger si chaleur superieure au seuil pomme de terre."},
        "rain": {"level": "stress hydrique", "bonus": 0.00, "trigger": "rain", "action": "Canaliser si pluie intense pour eviter saturation."},
        "mechanical": {"heat": "chaleur", "rain": "pluie"},
        "rationale": "Aucune maladie detectee: action climatique classique.",
    },
    "Tomato_Bacterial_spot": {
        "label": "Tomate - tache bacterienne",
        "heat": {"level": "modere a eleve", "bonus": 0.10, "trigger": "warm", "action": "Reduire stress thermique et surveiller humidite."},
        "rain": {"level": "eleve", "bonus": 0.22, "trigger": "wet", "action": "Incliner les plaques pour limiter eclaboussures et extension aux zones voisines."},
        "mechanical": {"heat": "chaleur", "rain": "pluie"},
        "rationale": "La tache bacterienne est favorisee par temps chaud et humide et se disperse par eau/eclaboussures.",
    },
    "Tomato_Early_blight": {
        "label": "Tomate - alternariose",
        "heat": {"level": "modere", "bonus": 0.08, "trigger": "warm", "action": "Surveiller chaleur + humidite; reduire stress."},
        "rain": {"level": "modere a eleve", "bonus": 0.16, "trigger": "wet", "action": "Incliner en pluie/humidite; etendre surveillance aux demi-zones voisines."},
        "mechanical": {"heat": "chaleur", "rain": "pluie"},
        "rationale": "L'alternariose de la tomate augmente avec chaleur, humidite et mouillure foliaire.",
    },
    "Tomato_Late_blight": {
        "label": "Tomate - mildiou",
        "heat": {"level": "faible a modere", "bonus": 0.04, "trigger": "stress", "action": "La chaleur seule n'est pas le facteur principal; conserver la surveillance."},
        "rain": {"level": "tres eleve", "bonus": 0.28, "trigger": "wet", "action": "Incliner immediatement et traiter A, B et C."},
        "mechanical": {"heat": "repos", "rain": "pluie"},
        "rationale": "Le mildiou tomate/pomme de terre est surtout favorise par humidite, pluie et mouillure prolongee.",
    },
    "Tomato_Leaf_Mold": {
        "label": "Tomate - leaf mold",
        "heat": {"level": "modere si serre chaude et humide", "bonus": 0.06, "trigger": "warm_humid", "action": "Eviter de fermer si cela augmente l'humidite; favoriser aeration."},
        "rain": {"level": "eleve par humidite", "bonus": 0.20, "trigger": "humid", "action": "Reduire humidite foliaire; incliner si pluie/humidite depasse le seuil."},
        "mechanical": {"heat": "repos", "rain": "pluie"},
        "rationale": "La moisissure foliaire est fortement liee a l'humidite relative elevee.",
    },
    "Tomato_Septoria_leaf_spot": {
        "label": "Tomate - septoriose",
        "heat": {"level": "faible a modere", "bonus": 0.05, "trigger": "stress", "action": "La chaleur seule n'est pas prioritaire; surveiller si humidite simultanee."},
        "rain": {"level": "eleve", "bonus": 0.24, "trigger": "wet", "action": "Incliner vers le canal; limiter eclaboussures; traiter zone + demi-zone voisine."},
        "mechanical": {"heat": "repos", "rain": "pluie"},
        "rationale": "La septoriose se propage par eclaboussures et mouillure prolongee des feuilles.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "label": "Tomate - acariens tetranyques",
        "heat": {"level": "eleve", "bonus": 0.20, "trigger": "hot_dry", "action": "Couvrir en forte chaleur pour reduire stress; inspecter zones voisines."},
        "rain": {"level": "faible", "bonus": -0.04, "trigger": "rain", "action": "La pluie n'augmente pas directement la propagation; eviter humidite excessive."},
        "mechanical": {"heat": "chaleur", "rain": "repos"},
        "rationale": "Les acariens sont generalement favorises par conditions chaudes et seches.",
    },
    "Tomato__Target_Spot": {
        "label": "Tomate - target spot",
        "heat": {"level": "modere", "bonus": 0.08, "trigger": "warm", "action": "Surveiller si chaleur accompagnee d'humidite."},
        "rain": {"level": "eleve", "bonus": 0.20, "trigger": "wet", "action": "Incliner en pluie; traiter zone et demi-zone voisine."},
        "mechanical": {"heat": "chaleur", "rain": "pluie"},
        "rationale": "Les taches foliaires sont aggravees par humidite, pluie et mouillure des feuilles.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "label": "Tomate - Yellow Leaf Curl Virus",
        "heat": {"level": "eleve via vecteurs/stress", "bonus": 0.16, "trigger": "warm", "action": "Reduire stress thermique; renforcer surveillance A, B et C et controle aleurodes."},
        "rain": {"level": "faible a modere", "bonus": 0.04, "trigger": "stress", "action": "La pluie n'est pas le vecteur principal; maintenir surveillance globale."},
        "mechanical": {"heat": "chaleur", "rain": "repos"},
        "rationale": "Le virus est transmis par aleurodes; la chaleur et le stress de culture renforcent le risque operationnel.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "label": "Tomate - virus mosaïque",
        "heat": {"level": "faible indirect", "bonus": 0.03, "trigger": "stress", "action": "Action mecanique seulement si stress climatique depasse les seuils."},
        "rain": {"level": "faible indirect", "bonus": 0.03, "trigger": "stress", "action": "La pluie n'est pas le vecteur principal; privilegier hygiene et isolation."},
        "mechanical": {"heat": "repos", "rain": "repos"},
        "rationale": "La mosaïque se transmet surtout mecaniquement/contact; le climat agit surtout comme stress secondaire.",
    },
    "Tomato_healthy": {
        "label": "Tomate saine",
        "heat": {"level": "stress physiologique", "bonus": 0.00, "trigger": "heat", "action": "Couvrir si temperature/luminosite depasse le seuil tomate."},
        "rain": {"level": "stress hydrique", "bonus": 0.00, "trigger": "rain", "action": "Incliner si pluie intense pour canaliser l'eau."},
        "mechanical": {"heat": "chaleur", "rain": "pluie"},
        "rationale": "Aucune maladie detectee: seule la protection climatique normale est active.",
    },
}



def classify_mechanical_mode(sensors, crop="default"):
    th = CROP_THRESHOLDS.get(crop, CROP_THRESHOLDS["default"])
    current = system_state["mode"]

    rain_alert = sensors["precipitation"] >= th["rain_on"] or sensors["humidity"] >= th["humidity_on"]
    heat_alert = sensors["temperature"] >= th["temp_on"] or sensors["luminosity"] >= th["luminosity_on"]
    clear_rain = sensors["precipitation"] <= th["rain_off"] and sensors["humidity"] <= th["humidity_off"]
    clear_heat = sensors["temperature"] <= th["temp_off"] and sensors["luminosity"] < th["luminosity_on"] - 80

    if current == "pluie" and not clear_rain:
        return "pluie"
    if rain_alert:
        return "pluie"
    if current == "chaleur" and not clear_heat:
        return "chaleur"
    if heat_alert:
        return "chaleur"
    return "repos"


def latest_active_disease(max_age_seconds=1800):
    now = datetime.datetime.now()
    for detection in reversed(detections_db):
        if detection.get("is_healthy") or detection.get("out_of_domain") or detection.get("is_valid_leaf") is False:
            continue
        try:
            detected_at = datetime.datetime.fromisoformat(detection["timestamp"])
        except (KeyError, ValueError):
            return detection
        if (now - detected_at).total_seconds() <= max_age_seconds:
            return detection
    return None


def environmental_modifier(policy, sensors, crop="default"):
    th = CROP_THRESHOLDS.get(crop, CROP_THRESHOLDS["default"])
    weather = policy.get("weather", "")
    warm = sensors["temperature"] >= th["temp_on"] - 1
    hot = sensors["temperature"] >= th["temp_on"]
    humid = sensors["humidity"] >= th["humidity_on"]
    wet = sensors["precipitation"] >= th["rain_off"] or humid
    heavy_rain = sensors["precipitation"] >= th["rain_on"]
    dry = sensors["humidity"] <= 45 and sensors["precipitation"] <= 2

    score = 0.0
    reasons = []
    if weather == "cool_wet":
        score += 0.18 if wet else 0
        score += 0.08 if sensors["temperature"] <= th["temp_on"] else 0
        if wet:
            reasons.append("mouillure foliaire favorable")
    elif weather == "warm_wet_splash":
        score += 0.14 if warm else 0
        score += 0.18 if wet else 0
        score += 0.10 if heavy_rain else 0
        if wet:
            reasons.append("eclaboussures/pluie favorables")
    elif weather == "wet_splash":
        score += 0.18 if wet else 0
        score += 0.12 if heavy_rain else 0
        if wet:
            reasons.append("dispersion par gouttes d'eau")
    elif weather == "humid":
        score += 0.22 if humid else 0
        if humid:
            reasons.append("humidite relative elevee")
    elif weather == "warm_vector":
        score += 0.16 if warm else 0
        if warm:
            reasons.append("stress thermique et pression vecteurs")
    elif weather == "hot_dry":
        score += 0.18 if hot and dry else 0
        if hot and dry:
            reasons.append("climat chaud et sec")
    elif weather == "contact_vector":
        score += 0.06 if warm or humid else 0

    return min(0.30, score), reasons


def climate_state(sensors, crop="default"):
    th = CROP_THRESHOLDS.get(crop, CROP_THRESHOLDS["default"])
    heat = sensors["temperature"] >= th["temp_on"] or sensors["luminosity"] >= th["luminosity_on"]
    rain = sensors["precipitation"] >= th["rain_on"] or sensors["humidity"] >= th["humidity_on"]
    warm = sensors["temperature"] >= th["temp_on"] - 1
    wet = sensors["precipitation"] >= th["rain_off"] or sensors["humidity"] >= th["humidity_off"]
    hot_dry = sensors["temperature"] >= th["temp_on"] and sensors["humidity"] <= 45 and sensors["precipitation"] <= 2
    return {
        "heat": heat,
        "rain": rain,
        "warm": warm,
        "wet": wet,
        "hot_dry": hot_dry,
        "thresholds": th,
    }


def trigger_active(trigger, state):
    return {
        "heat": state["heat"],
        "rain": state["rain"],
        "warm": state["warm"],
        "wet": state["wet"],
        "humid": state["wet"],
        "warm_humid": state["warm"] and state["wet"],
        "hot_dry": state["hot_dry"],
        "stress": state["heat"] or state["rain"],
    }.get(trigger, False)


def climate_diagnosis_for_result(result, sensors, crop="default"):
    if result.get("out_of_domain") or result.get("is_valid_leaf") is False:
        return {
            "class_label": "Image hors dataset",
            "current_weather": {
                "temperature": sensors["temperature"],
                "humidity": sensors["humidity"],
                "precipitation": sensors["precipitation"],
                "luminosity": sensors["luminosity"],
                "heat_alert": False,
                "rain_alert": False,
            },
            "heat": {
                "active": False,
                "level": "non applicable",
                "risk_delta": 0,
                "action": "Aucune action maladie: image non reconnue comme feuille de tomate, poivron ou pomme de terre.",
            },
            "rain": {
                "active": False,
                "level": "non applicable",
                "risk_delta": 0,
                "action": "Aucune action maladie: verifier la prise de vue et recapturer une feuille.",
            },
            "base_risk": 0,
            "climate_delta": 0,
            "instant_risk": 0,
            "recommended_mechanism": "repos",
            "rationale": "L'image est rejetee par le controle hors domaine; elle ne pilote pas les plaques.",
        }
    profile = CLIMATE_IMPACT_BY_CLASS.get(result["predicted_class"])
    if profile is None:
        profile = {
            "label": result["predicted_class"],
            "heat": {"level": "non specialise", "bonus": 0.03, "trigger": "stress", "action": "Surveiller le stress climatique."},
            "rain": {"level": "non specialise", "bonus": 0.03, "trigger": "stress", "action": "Surveiller humidite et precipitation."},
            "mechanical": {"heat": "repos", "rain": "repos"},
            "rationale": "Classe non mappee explicitement: regle prudente.",
        }
    state = climate_state(sensors, crop)
    heat_active = trigger_active(profile["heat"]["trigger"], state)
    rain_active = trigger_active(profile["rain"]["trigger"], state)
    heat_bonus = profile["heat"]["bonus"] if heat_active else 0
    rain_bonus = profile["rain"]["bonus"] if rain_active else 0
    climate_bonus = max(-0.05, min(0.32, heat_bonus + rain_bonus))
    base_policy = disease_policy(result["disease"])
    base_risk = 0.0 if result["is_healthy"] else base_policy["risk"]
    instant_risk = round(max(0, min(0.98, base_risk + climate_bonus)), 2)

    recommended_mode = "repos"
    if rain_active and profile["mechanical"].get("rain") != "repos":
        recommended_mode = profile["mechanical"]["rain"]
    elif heat_active and profile["mechanical"].get("heat") != "repos":
        recommended_mode = profile["mechanical"]["heat"]
    elif state["rain"]:
        recommended_mode = "pluie"
    elif state["heat"]:
        recommended_mode = "chaleur"

    return {
        "class_label": profile["label"],
        "current_weather": {
            "temperature": sensors["temperature"],
            "humidity": sensors["humidity"],
            "precipitation": sensors["precipitation"],
            "luminosity": sensors["luminosity"],
            "heat_alert": state["heat"],
            "rain_alert": state["rain"],
        },
        "heat": {
            "active": heat_active,
            "level": profile["heat"]["level"],
            "risk_delta": round(heat_bonus, 2),
            "action": profile["heat"]["action"],
        },
        "rain": {
            "active": rain_active,
            "level": profile["rain"]["level"],
            "risk_delta": round(rain_bonus, 2),
            "action": profile["rain"]["action"],
        },
        "base_risk": round(base_risk, 2),
        "climate_delta": round(climate_bonus, 2),
        "instant_risk": instant_risk,
        "recommended_mechanism": recommended_mode,
        "rationale": profile["rationale"],
    }


def disease_informed_mechanical_mode(sensors, crop="default"):
    base_mode = classify_mechanical_mode(sensors, crop)
    detection = latest_active_disease()
    if not detection:
        return base_mode, []

    result_like = {
        "predicted_class": detection.get("predicted_class", ""),
        "disease": detection.get("disease", ""),
        "is_healthy": detection.get("is_healthy", False),
    }
    diagnosis = climate_diagnosis_for_result(result_like, sensors, crop)
    if diagnosis["climate_delta"] < 0.12:
        return base_mode, [diagnosis["rationale"]]
    mode = diagnosis["recommended_mechanism"]
    if mode in {"pluie", "chaleur"}:
        return mode, [diagnosis["rationale"]]
    if base_mode != "repos":
        return base_mode, [diagnosis["rationale"]]
    return base_mode, [diagnosis["rationale"]]


def apply_mechanical_state(mode):
    target = {
        "repos": {"plate_angle": 0, "servo_angle": 0},
        "chaleur": {"plate_angle": 90, "servo_angle": 90},
        "pluie": {"plate_angle": 45, "servo_angle": 55},
    }[mode]
    system_state.update({"mode": mode, **target, "last_change": time.time()})


def evaluate_actuator(sensors, crop="default"):
    next_mode, _ = disease_informed_mechanical_mode(sensors, crop)
    elapsed = time.time() - system_state["last_change"]
    th = CROP_THRESHOLDS.get(crop, CROP_THRESHOLDS["default"])
    urgent_rain = next_mode == "pluie" and (
        sensors["precipitation"] >= th["rain_on"] or latest_active_disease() is not None
    )
    urgent_heat = next_mode == "chaleur" and (
        sensors["temperature"] >= th["temp_on"] and latest_active_disease() is not None
    )
    if next_mode != system_state["mode"] and (elapsed >= HYSTERESIS_DELAY or urgent_rain or urgent_heat):
        apply_mechanical_state(next_mode)


def mechanism_description(mode):
    return {
        "repos": "Plaques abaissees a 0 degre: la culture reste ouverte.",
        "chaleur": "Plaques horizontales a 90 degres: elles couvrent la culture contre la forte chaleur.",
        "pluie": "Plaques inclinees vers le canal central: l'eau est conduite vers le reservoir.",
    }[mode]


def risk_level(score):
    if score >= 75:
        return "critique"
    if score >= 55:
        return "attention"
    return "normal"


def zone_adjustment(zone):
    return {"A": -4, "B": 0, "C": 5}.get(zone, 0)


def compute_zone_swi(data, crop="default"):
    model = CROP_DECISION_MODELS.get(crop, CROP_DECISION_MODELS["default"])
    soil = float(data.get("soil_moisture", model["soil_optimum"]))
    temp = float(data["temperature"])
    humidity = float(data["humidity"])
    rain = float(data["precipitation"])
    light = float(data["luminosity"])
    opt = model["soil_optimum"]
    thermal = max(0, temp - CROP_THRESHOLDS.get(crop, CROP_THRESHOLDS["default"])["temp_off"]) * 2.1
    dry_air = max(0, 58 - humidity) * 0.55
    light_load = max(0, light - 650) * 0.035
    rain_relief = min(18, rain * 0.7)
    base = max(0, opt - soil) * 1.35 + thermal + dry_air + light_load - rain_relief
    zones = {}
    for zone in ("A", "B", "C"):
        score = round(max(0, min(100, base + zone_adjustment(zone))))
        zones[zone] = {
            "score": score,
            "level": risk_level(score),
            "recommendation": "Irrigation automatique recommandee" if score >= 65 else ("Surveillance irrigation" if score >= 45 else "Hydratation correcte"),
        }
    return zones


def compute_gdd(data, crop="default"):
    model = CROP_DECISION_MODELS.get(crop, CROP_DECISION_MODELS["default"])
    try:
        sowing = datetime.date.fromisoformat(model["sowing_date"])
    except ValueError:
        sowing = datetime.date(datetime.datetime.now().year, 3, 15)
    today = datetime.datetime.now().date()
    days = max(1, (today - sowing).days)
    history = list(sensor_history)[-24:]
    mean_temp = sum(item["temperature"] for item in history) / len(history) if history else float(data["temperature"])
    daily_gdd = max(0, mean_temp - model["base_temp"])
    cumulative = round(daily_gdd * days, 1)
    stages = model["gdd_stages"]
    stage = stages[0][1]
    next_stage = stages[-1]
    for threshold, label in stages:
        if cumulative >= threshold:
            stage = label
        else:
            next_stage = (threshold, label)
            break
    remaining = max(0, next_stage[0] - cumulative)
    days_to_next = round(remaining / max(0.1, daily_gdd))
    return {
        "cumulative": cumulative,
        "daily": round(daily_gdd, 1),
        "stage": stage,
        "next_stage": next_stage[1],
        "days_to_next": days_to_next,
        "base_temp": model["base_temp"],
        "sowing_date": model["sowing_date"],
    }


def compute_ndvi_proxy(data):
    green = max(0, min(255, 80 + data.get("soil_moisture", 62) * 1.5 - data["temperature"] * 0.7))
    red = max(0, min(255, 70 + data["temperature"] * 1.4 + data["precipitation"] * 0.25))
    blue = max(0, min(255, 45 + data["humidity"] * 0.35))
    gcc = green / max(1, red + green + blue)
    ndvi_proxy = round(max(0, min(0.95, (gcc - 0.28) / 0.34)), 2)
    trend = "stable"
    if data["temperature"] >= CROP_THRESHOLDS.get(data.get("crop", "default"), CROP_THRESHOLDS["default"])["temp_on"] or data["humidity"] >= 88:
        trend = "baisse probable"
    elif data.get("soil_moisture", 65) >= 65 and data["temperature"] < 29:
        trend = "hausse legere"
    return {"value": ndvi_proxy, "gcc": round(gcc, 3), "trend": trend, "method": "proxy RGB Green Chromatic Coordinate"}


def compute_risk_forecast(data, crop="default", disease_context=None):
    th = CROP_THRESHOLDS.get(crop, CROP_THRESHOLDS["default"])
    base = 24
    if data["humidity"] >= th["humidity_on"]:
        base += 18
    if data["precipitation"] >= th["rain_on"]:
        base += 22
    if data["temperature"] >= th["temp_on"]:
        base += 14
    if disease_context:
        base += round(disease_context.get("combined_risk", 0) * 28)
    days = []
    for idx in range(1, 8):
        drift = (idx - 1) * 3
        rain_wave = 7 if idx in {2, 3, 4} and data["precipitation"] > th["rain_off"] else 0
        score = max(5, min(98, base + drift + rain_wave - idx))
        days.append({"day": idx, "risk": score, "level": risk_level(score)})
    peak = max(days, key=lambda item: item["risk"])
    action = "Traitement preventif conseille" if peak["risk"] >= 75 else ("Surveillance rapprochee" if peak["risk"] >= 55 else "Risque controle")
    return {"days": days, "peak_day": peak["day"], "peak_risk": peak["risk"], "action": action}


def build_alerts(data, analytics, disease_context=None):
    alerts = []
    for zone, info in analytics["swi"]["zones"].items():
        if info["score"] >= 75:
            alerts.append({"level": "critique", "trigger": "soil_moisture", "zone": zone, "message": f"Stress hydrique zone {zone}: {info['score']}/100", "action": "Irrigation automatique recommandee"})
        elif info["score"] >= 55:
            alerts.append({"level": "attention", "trigger": "soil_moisture", "zone": zone, "message": f"Stress hydrique modere zone {zone}", "action": "Verifier humidite du sol"})
    if analytics["forecast"]["peak_risk"] >= 75:
        alerts.append({"level": "critique", "trigger": "disease_forecast", "zone": disease_context.get("zone", "A") if disease_context else "A/B/C", "message": f"Risque maladie J+{analytics['forecast']['peak_day']}: {analytics['forecast']['peak_risk']}%", "action": analytics["forecast"]["action"]})
    elif analytics["forecast"]["peak_risk"] >= 55:
        alerts.append({"level": "attention", "trigger": "disease_forecast", "zone": disease_context.get("zone", "A") if disease_context else "A/B/C", "message": f"Risque maladie a surveiller: {analytics['forecast']['peak_risk']}%", "action": analytics["forecast"]["action"]})
    if data["mechanism"]["mode"] != "repos":
        alerts.append({"level": "info", "trigger": "actuator_command", "zone": "systeme", "message": f"Plaques en mode {data['mechanism']['mode']}", "action": data["actuator_command"]["reason"]})
    if not alerts:
        alerts.append({"level": "info", "trigger": "system_health", "zone": "systeme", "message": "Aucune alerte critique", "action": "Continuer la surveillance"})
    for alert in alerts:
        alert["delivery"] = {
            "email": ALERT_CHANNELS["email"],
            "sms": ALERT_CHANNELS["sms"],
            "payload": {
                "subject": f"AgroShield {alert['level'].upper()} - {alert['zone']}",
                "body": f"{alert['message']} | Action: {alert['action']} | Capteur: {alert['trigger']}",
            },
        }
    return alerts[:6]


def compute_decision_analytics(data, crop="default", disease_context=None):
    analytics = {
        "swi": {"zones": compute_zone_swi(data, crop), "method": "Indice operationnel derive humidite sol, temperature, humidite air, pluie et luminosite"},
        "gdd": compute_gdd(data, crop),
        "ndvi": compute_ndvi_proxy(data),
    }
    analytics["forecast"] = compute_risk_forecast(data, crop, disease_context)
    return analytics


def decorate_sensor_payload(data, crop="default"):
    crop = crop if crop in CROP_THRESHOLDS else "default"
    data["crop"] = crop
    evaluate_actuator(data, crop)
    th = CROP_THRESHOLDS.get(crop, CROP_THRESHOLDS["default"])
    active_detection = latest_active_disease()
    disease_context = None
    if active_detection:
        result_like = {
            "predicted_class": active_detection.get("predicted_class", ""),
            "disease": active_detection.get("disease", ""),
            "is_healthy": active_detection.get("is_healthy", False),
        }
        diagnosis = climate_diagnosis_for_result(result_like, data, crop)
        disease_context = {
            "disease": active_detection.get("disease"),
            "zone": active_detection.get("zone_id"),
            "base_risk": diagnosis["base_risk"],
            "weather_modifier": diagnosis["climate_delta"],
            "combined_risk": diagnosis["instant_risk"],
            "weather_reasons": [diagnosis["rationale"]],
            "mechanical_bias": diagnosis["recommended_mechanism"],
        }
    data["thresholds"] = th
    data["mechanism"] = {
        "mode": system_state["mode"],
        "plate_angle": system_state["plate_angle"],
        "servo_angle": system_state["servo_angle"],
        "description": mechanism_description(system_state["mode"]),
    }
    data["plates"] = {
        "angle": system_state["plate_angle"],
        "left": system_state["mode"] != "repos",
        "right": system_state["mode"] != "repos",
    }
    data["alerts"] = {
        "temperature": data["temperature"] >= th["temp_on"],
        "humidity": data["humidity"] >= th["humidity_on"],
        "precipitation": data["precipitation"] >= th["rain_on"],
        "luminosity": data["luminosity"] >= th["luminosity_on"],
    }
    data["disease_context"] = disease_context
    command_reason = "Seuils climatiques de la culture"
    command_source = "crop_thresholds"
    if disease_context and disease_context["combined_risk"] >= 0.70:
        command_reason = (
            f"Maladie detectee en zone {disease_context['zone']}: "
            f"risque maladie-climat {round(disease_context['combined_risk'] * 100)}%"
        )
        command_source = "disease_climate_loop"
    data["actuator_command"] = {
        "mode": system_state["mode"],
        "plate_angle": system_state["plate_angle"],
        "servo_angle": system_state["servo_angle"],
        "source": command_source,
        "reason": command_reason,
        "apply_on_raspberry": True,
    }
    data["analytics"] = compute_decision_analytics(data, crop, disease_context)
    data["alerts_center"] = build_alerts(data, data["analytics"], disease_context)
    sensor_history.append(data.copy())
    return data


def get_sensors():
    raspberry = latest_raspberry_payload.get("data")
    if raspberry and time.time() - latest_raspberry_payload.get("timestamp", 0) < 25:
        data = dict(raspberry)
        data["mode"] = "raspberry"
        data["last_update"] = latest_raspberry_payload["timestamp"]
        return decorate_sensor_payload(data, data.get("crop", system_state["current_crop"]))

    # The simulation deliberately creates occasional heat and rain peaks so the
    # interface visibly demonstrates all mechanical positions during a defense.
    event = random.choices(["normal", "heat", "rain"], weights=[0.58, 0.22, 0.20], k=1)[0]
    if event == "heat":
        temperature = random.uniform(29, 37)
        humidity = random.uniform(42, 68)
        precipitation = random.uniform(0, 5)
        luminosity = random.uniform(680, 920)
    elif event == "rain":
        temperature = random.uniform(20, 28)
        humidity = random.uniform(82, 96)
        precipitation = random.uniform(20, 48)
        luminosity = random.uniform(120, 430)
    else:
        temperature = random.uniform(22, 27)
        humidity = random.uniform(48, 72)
        precipitation = random.uniform(0, 7)
        luminosity = random.uniform(280, 620)

    data = {
        "temperature": round(temperature, 1),
        "humidity": round(humidity, 1),
        "precipitation": round(precipitation, 1),
        "luminosity": round(luminosity, 1),
        "soil_moisture": round(random.uniform(48, 78), 1),
        "reservoir_level": round(random.uniform(42, 92), 1),
        "mode": "simulation",
        "last_update": time.time(),
        "crop": system_state["current_crop"],
    }
    return decorate_sensor_payload(data, system_state["current_crop"])


def mock_predict():
    choices = [
        ("Tomato", "healthy", 0.93, True),
        ("Tomato", "Late blight", 0.88, False),
        ("Tomato", "Yellow Leaf Curl Virus", 0.84, False),
        ("Pepper bell", "Bacterial spot", 0.81, False),
        ("Potato", "Early blight", 0.79, False),
    ]
    plant, disease, confidence, healthy = random.choice(choices)
    return {
        "predicted_class": f"{plant}___{disease}",
        "efficientnet": {"prediction": f"{plant}__{disease}", "confidence": confidence, "top3": []},
        "resnet": {"prediction": f"{plant}__{disease}", "confidence": confidence - 0.03, "top3": []},
        "agreement": True,
        "agreement_score": "demo",
        "plant": plant,
        "disease": disease,
        "confidence": confidence,
        "is_healthy": healthy,
        "is_valid_leaf": True,
        "out_of_domain": False,
        "domain_status": {"accepted": True, "label": "mode demonstration", "reasons": []},
        "demo": True,
    }


def parse_class_name(class_name):
    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
    elif "__" in class_name:
        plant, disease = class_name.split("__", 1)
    else:
        parts = class_name.split("_", 1)
        if len(parts) == 2 and parts[0].lower() in {"tomato", "potato", "pepper"}:
            plant, disease = parts
        else:
            plant, disease = class_name, "Healthy"
    return plant.replace("__", " ").replace("_", " "), disease.replace("__", " ").replace("_", " ")


def probability_margin(probs):
    values = probs.topk(2).values.tolist()
    if len(values) < 2:
        return float(values[0]) if values else 0.0
    return float(values[0] - values[1])


def plant_pixel_evidence(pil_image):
    sample = pil_image.convert("RGB").resize((160, 160))
    hsv = sample.convert("HSV")
    pixels = list(hsv.getdata())
    total = max(1, len(pixels))
    plant_like = 0
    saturated = 0
    low_saturation = 0
    very_bright = 0
    for h, s, v in pixels:
        hue = h * 360 / 255
        if s >= 38 and v >= 30:
            saturated += 1
        if s <= 28:
            low_saturation += 1
        if v >= 230:
            very_bright += 1
        green_leaf = 45 <= hue <= 175 and s >= 35 and v >= 35
        yellow_or_brown_leaf = 18 <= hue < 55 and s >= 38 and 35 <= v <= 230
        reddish_brown_spot = (hue < 18 or hue >= 330) and s >= 45 and 35 <= v <= 210
        if green_leaf or yellow_or_brown_leaf or reddish_brown_spot:
            plant_like += 1
    return {
        "plant_pixel_ratio": round(plant_like / total, 4),
        "saturation_ratio": round(saturated / total, 4),
        "low_saturation_ratio": round(low_saturation / total, 4),
        "very_bright_ratio": round(very_bright / total, 4),
    }


def domain_status_for_image(pil_image, eff_conf, res_conf, eff_margin, res_margin, agree):
    visual = plant_pixel_evidence(pil_image)
    final_conf = max(eff_conf, res_conf)
    avg_margin = (eff_margin + res_margin) / 2
    plant_ratio = visual["plant_pixel_ratio"]
    saturation_ratio = visual["saturation_ratio"]
    reasons = []

    if plant_ratio < 0.035:
        reasons.append("l'image contient trop peu de pixels compatibles avec une feuille ou une culture")
    if plant_ratio < 0.08 and saturation_ratio < 0.16:
        reasons.append("l'image ressemble davantage a un objet/ecran/document qu'a une feuille")
    if not agree and final_conf < 0.78:
        reasons.append("EfficientNet-B0 et ResNet-50 ne sont pas assez coherents pour valider la classe")
    if avg_margin < 0.08 and final_conf < 0.88:
        reasons.append("la marge entre la meilleure classe et les classes voisines est insuffisante")

    accepted = not reasons
    if plant_ratio >= 0.12 and final_conf >= 0.82 and avg_margin >= 0.12:
        accepted = True
        reasons = []
    elif agree and plant_ratio >= 0.08 and final_conf >= 0.70 and avg_margin >= 0.10:
        accepted = True
        reasons = []

    return {
        "accepted": accepted,
        "label": "image foliaire valide" if accepted else "image hors dataset / non vegetale",
        "reasons": reasons,
        "plant_probability": round(min(1.0, plant_ratio / 0.18), 2),
        "model_confidence": round(final_conf, 4),
        "ensemble_agreement": agree,
        "average_margin": round(avg_margin, 4),
        **visual,
    }


def out_of_domain_prediction(domain_status, eff_pred, eff_conf, res_pred, res_conf, top3_eff, top3_res, agree):
    return {
        "predicted_class": "Hors_dataset_non_vegetal",
        "efficientnet": {"prediction": eff_pred, "confidence": eff_conf, "top3": top3_eff},
        "resnet": {"prediction": res_pred, "confidence": res_conf, "top3": top3_res},
        "agreement": agree,
        "agreement_score": "rejete",
        "plant": "Hors domaine",
        "disease": "Image non reconnue",
        "confidence": 0.0,
        "is_healthy": True,
        "is_valid_leaf": False,
        "out_of_domain": True,
        "domain_status": domain_status,
        "demo": False,
    }


def predict(pil_image):
    if not MODEL_STATUS["available"] or eff_model is None or preprocess is None:
        return mock_predict()

    tensor = preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        eff_probs = torch.softmax(eff_model(tensor), dim=1)[0]
    eff_top = eff_probs.topk(3)
    eff_pred = CLASSES[eff_top.indices[0].item()]
    eff_conf = float(eff_top.values[0])
    eff_margin = probability_margin(eff_probs)
    top3_eff = [
        {"class": CLASSES[i].replace("__", " - ").replace("_", " "), "confidence": float(eff_probs[i])}
        for i in eff_top.indices.tolist()
    ]

    if USE_RESNET and res_model is not None:
        with torch.no_grad():
            res_probs = torch.softmax(res_model(tensor), dim=1)[0]
        res_top = res_probs.topk(3)
        res_pred = CLASSES[res_top.indices[0].item()]
        res_conf = float(res_top.values[0])
        res_margin = probability_margin(res_probs)
        top3_res = [
            {"class": CLASSES[i].replace("__", " - ").replace("_", " "), "confidence": float(res_probs[i])}
            for i in res_top.indices.tolist()
        ]
        agree = eff_pred == res_pred
        final_pred = eff_pred if eff_conf >= res_conf else res_pred
        final_conf = max(eff_conf, res_conf)
        agree_score = "2/2" if agree else "1/2"
    else:
        res_pred = eff_pred
        res_conf = eff_conf
        res_margin = eff_margin
        top3_res = top3_eff
        agree = True
        final_pred = eff_pred
        final_conf = eff_conf
        agree_score = "1/1"

    domain_status = domain_status_for_image(pil_image, eff_conf, res_conf, eff_margin, res_margin, agree)
    if not domain_status["accepted"]:
        return out_of_domain_prediction(domain_status, eff_pred, eff_conf, res_pred, res_conf, top3_eff, top3_res, agree)

    plant, disease = parse_class_name(final_pred)
    return {
        "predicted_class": final_pred,
        "efficientnet": {"prediction": eff_pred, "confidence": eff_conf, "top3": top3_eff},
        "resnet": {"prediction": res_pred, "confidence": res_conf, "top3": top3_res},
        "agreement": agree,
        "agreement_score": agree_score,
        "plant": plant,
        "disease": disease,
        "confidence": final_conf,
        "is_healthy": "healthy" in final_pred.lower(),
        "is_valid_leaf": True,
        "out_of_domain": False,
        "domain_status": domain_status,
        "demo": False,
    }


def disease_policy(disease):
    disease_key = disease.lower().replace("_", " ")
    for key, policy in DISEASE_POLICIES.items():
        if key in disease_key:
            return policy
    return {
        "risk": 0.55,
        "spread": "moderee",
        "action": "Traiter la zone detectee et maintenir la surveillance.",
        "scope": "local",
        "weather": "contact_vector",
        "mechanical_bias": "repos",
        "source": "Regle conservatrice appliquee aux classes non specialisees.",
    }


def zones_for_policy(zone, confidence, policy):
    if policy["scope"] == "all":
        return ["A", "B", "C"]
    if policy["scope"] == "all_if_confident" and confidence >= 0.75:
        return ["A", "B", "C"]
    if policy["scope"] == "adjacent_high_all" and confidence >= 0.90:
        return ["A", "B", "C"]
    if policy["scope"] in {"adjacent", "adjacent_high_all"}:
        return sorted({zone, *ADJACENT_ZONES.get(zone, [])})
    if policy["scope"] == "confidence_adjacent" and confidence >= 0.80:
        return sorted({zone, *ADJACENT_ZONES.get(zone, [])})
    return [zone]


def treatment_segments(zone, confidence, policy):
    scope = policy["scope"]
    if scope == "all" or (scope == "all_if_confident" and confidence >= 0.75):
        return [
            {"zone": "A", "portion": "complete", "label": "Zone A complète"},
            {"zone": "B", "portion": "complete", "label": "Zone B complète"},
            {"zone": "C", "portion": "complete", "label": "Zone C complète"},
        ]
    if scope == "adjacent_high_all" and confidence >= 0.90:
        return [
            {"zone": "A", "portion": "complete", "label": "Zone A complète"},
            {"zone": "B", "portion": "complete", "label": "Zone B complète"},
            {"zone": "C", "portion": "complete", "label": "Zone C complète"},
        ]

    segments = [{"zone": zone, "portion": "complete", "label": f"Zone {zone} complète"}]
    adjacent = ADJACENT_ZONES.get(zone, [])
    if scope in {"adjacent", "adjacent_high_all"}:
        for adj in adjacent:
            side = "gauche" if adj > zone else "droite"
            segments.append({"zone": adj, "portion": "half", "label": f"moitié {side} de la zone {adj}"})
    elif scope == "confidence_adjacent" and confidence >= 0.80:
        for adj in adjacent:
            side = "gauche" if adj > zone else "droite"
            segments.append({"zone": adj, "portion": "half", "label": f"moitié {side} de la zone {adj}"})
    return segments


def get_treatment_zones(zone_id, confidence, is_healthy, disease=""):
    if is_healthy or confidence < 0.6:
        return {
            "zones": [],
            "actuators": [],
            "priority": "surveillance",
            "propagation_probability": 0,
            "spread_level": "faible",
            "recommendation": "Aucun traitement immédiat; continuer la surveillance.",
            "segments": [],
            "treatment_summary": "Surveillance uniquement.",
        }
    zone = zone_id.upper() if zone_id.upper() in ZONES else "A"
    policy = disease_policy(disease)
    segments = treatment_segments(zone, confidence, policy)
    zones_to_treat = sorted({segment["zone"] for segment in segments})
    actuators = sorted({ZONES[z]["actuator"] for z in zones_to_treat if z in ZONES})
    propagation = round(min(0.98, policy["risk"] * (0.75 + 0.25 * confidence)), 2)
    summary = " + ".join(segment["label"] for segment in segments)
    return {
        "zones": zones_to_treat,
        "actuators": actuators,
        "priority": "intervention",
        "propagation_probability": propagation,
        "spread_level": policy["spread"],
        "recommendation": policy["action"],
        "segments": segments,
        "treatment_summary": summary,
    }


def analyze_image_object(img, zone_id="A", lat=33.5731, lng=-7.5898, source="manual", image_url=None, filename=None):
    global latest_analysis
    result = predict(img)
    sensors_now = sensor_history[-1] if sensor_history else get_sensors()
    policy = disease_policy(result["disease"])
    climate_diagnosis = climate_diagnosis_for_result(result, sensors_now, system_state["current_crop"])
    treatment = get_treatment_zones(zone_id, result["confidence"], result["is_healthy"], result["disease"])
    if result.get("out_of_domain") or result.get("is_valid_leaf") is False:
        treatment.update(
            {
                "priority": "rejet",
                "recommendation": "Image hors dataset: recapturer une feuille de tomate, poivron ou pomme de terre avant diagnostic.",
                "treatment_summary": "Aucune zone a traiter.",
            }
        )
    elif not result["is_healthy"]:
        boosted = round(min(0.98, treatment["propagation_probability"] + climate_diagnosis["climate_delta"]), 2)
        treatment["propagation_probability"] = boosted
        treatment["weather_modifier"] = climate_diagnosis["climate_delta"]
        treatment["weather_reasons"] = [
            f"Chaleur: {climate_diagnosis['heat']['level']}",
            f"Pluie/humidite: {climate_diagnosis['rain']['level']}",
        ]
        treatment["policy_source"] = policy.get("source", "")
        if boosted >= 0.84 and treatment["zones"] != ["A", "B", "C"]:
            treatment["recommendation"] += " Conditions actuelles favorables: etendre la surveillance a toute la culture."
    det = {
        "zone_id": zone_id.upper(),
        "lat": lat,
        "lng": lng,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "plant": result["plant"],
        "disease": result["disease"],
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "is_healthy": result["is_healthy"],
        "is_valid_leaf": result.get("is_valid_leaf", True),
        "out_of_domain": result.get("out_of_domain", False),
        "domain_status": result.get("domain_status", {}),
        "agreement_score": result["agreement_score"],
        "treatment": treatment,
        "source": source,
        "image_url": image_url,
        "filename": filename,
    }
    detections_db.append(det)
    latest_analysis = {**result, "zone": det, "treatment": treatment, "climate_diagnosis": climate_diagnosis}
    return latest_analysis


def analyze_received_image_path(path, zone_id="A", source="raspberry_usb_upload"):
    img = Image.open(path).convert("RGB")
    return analyze_image_object(
        img,
        zone_id=zone_id,
        source=source,
        image_url=f"/received_images/{path.name}",
        filename=path.name,
    )


def scan_received_images_once(zone_id="A"):
    RECEIVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [
        path for path in RECEIVED_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    path = candidates[0]
    signature = f"{path.name}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
    if signature in processed_received_images:
        return None
    result = analyze_received_image_path(path, zone_id=zone_id)
    processed_received_images.add(signature)
    return result


def mark_received_image_processed(path):
    try:
        signature = f"{path.name}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
        processed_received_images.add(signature)
    except OSError:
        pass


@app.route("/")
def index():
    return send_file(str(BASE_DIR / "index.html"))


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(str(BASE_DIR / "assets"), filename)


@app.route("/received_images/<path:filename>")
def received_images(filename):
    return send_from_directory(str(RECEIVED_IMAGES_DIR), filename)


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_status": MODEL_STATUS,
            "resnet": USE_RESNET,
            "timestamp": time.time(),
            "access": "0.0.0.0:5000",
        }
    )


@app.route("/api/sensors")
def sensors():
    return jsonify(get_sensors())


@app.route("/api/raspberry/sensors", methods=["POST"])
def raspberry_sensors():
    payload = request.get_json(silent=True) or {}
    required = ("temperature", "humidity", "precipitation", "luminosity")
    missing = [key for key in required if key not in payload]
    if missing:
        return jsonify({"error": f"Champs manquants: {', '.join(missing)}"}), 400
    crop = payload.get("crop", system_state["current_crop"])
    if crop in CROP_THRESHOLDS:
        system_state["current_crop"] = crop
    data = {
        "temperature": round(float(payload["temperature"]), 1),
        "humidity": round(float(payload["humidity"]), 1),
        "precipitation": round(float(payload["precipitation"]), 1),
        "luminosity": round(float(payload["luminosity"]), 1),
        "soil_moisture": round(float(payload.get("soil_moisture", 0)), 1),
        "reservoir_level": round(float(payload.get("reservoir_level", 0)), 1),
        "crop": system_state["current_crop"],
        "pi_id": payload.get("pi_id", "raspberry-pi"),
    }
    latest_raspberry_payload.update({"timestamp": time.time(), "data": data})
    return jsonify({"received": True, **decorate_sensor_payload(data, system_state["current_crop"])})


@app.route("/api/history")
def history():
    return jsonify({"history": list(sensor_history)})


def build_report_payload(audience="technician"):
    latest = sensor_history[-1] if sensor_history else get_sensors()
    detections = list(detections_db)[-10:]
    audience = "farmer" if audience == "farmer" else "technician"
    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "project": "AgroShield - Centrale Casablanca - Groupe PLBD 3",
        "audience": audience,
        "summary": {
            "crop": latest.get("crop", system_state["current_crop"]),
            "mechanism": latest.get("mechanism", {}),
            "alerts": latest.get("alerts_center", []),
            "analytics": latest.get("analytics", {}),
            "last_detection": latest_analysis,
            "sensors": {
                "temperature": latest.get("temperature"),
                "humidity": latest.get("humidity"),
                "precipitation": latest.get("precipitation"),
                "luminosity": latest.get("luminosity"),
                "soil_moisture": latest.get("soil_moisture"),
                "reservoir_level": latest.get("reservoir_level"),
            },
        },
        "detections": detections,
        "recommendation": "Rapport PDF genere automatiquement pour suivi agronomique terrain.",
        "notification_channels": ALERT_CHANNELS,
    }


def pdf_escape(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def report_lines(payload):
    summary = payload["summary"]
    analytics = summary.get("analytics", {})
    mechanism = summary.get("mechanism", {})
    sensors_now = summary.get("sensors", {})
    lines = [
        "AgroShield - Rapport agronomique automatise",
        f"Centrale Casablanca - Groupe PLBD 3",
        f"Generation: {payload['generated_at']}",
        "",
        "1. Etat global",
        f"Culture suivie: {summary.get('crop', '--')}",
        f"Mode plaques: {mechanism.get('mode', '--')} | angle plaques {mechanism.get('plate_angle', '--')} deg | servo {mechanism.get('servo_angle', '--')} deg",
        f"Description: {mechanism.get('description', '--')}",
        "",
        "2. Capteurs instantanes",
        f"Temperature: {sensors_now.get('temperature', '--')} C",
        f"Humidite air: {sensors_now.get('humidity', '--')} %",
        f"Precipitation: {sensors_now.get('precipitation', '--')} mm/h",
        f"Luminosite: {sensors_now.get('luminosity', '--')} lx",
        f"Humidite sol: {sensors_now.get('soil_moisture', '--')} %",
        f"Reservoir: {sensors_now.get('reservoir_level', '--')} %",
        "",
        "3. Indicateurs agronomiques",
    ]
    for zone, info in (analytics.get("swi", {}).get("zones", {}) or {}).items():
        lines.append(f"SWI Zone {zone}: {info.get('score', '--')}/100 - {info.get('level', '--')} - {info.get('recommendation', '--')}")
    gdd = analytics.get("gdd", {})
    lines.append(f"GDD cumule: {gdd.get('cumulative', '--')} degC.j | stade: {gdd.get('stage', '--')} | prochaine etape: {gdd.get('next_stage', '--')} (~{gdd.get('days_to_next', '--')} j)")
    ndvi = analytics.get("ndvi", {})
    lines.append(f"NDVI proxy RGB/GCC: {ndvi.get('value', '--')} | GCC {ndvi.get('gcc', '--')} | tendance: {ndvi.get('trend', '--')}")
    forecast = analytics.get("forecast", {})
    lines.append(f"Risque maladie 7 jours: pic J+{forecast.get('peak_day', '--')} a {forecast.get('peak_risk', '--')}% | {forecast.get('action', '--')}")
    lines.extend(["", "4. Alertes et actions recommandees"])
    alerts = summary.get("alerts", [])
    if alerts:
        for alert in alerts:
            lines.append(f"{alert.get('level', '--').upper()} - {alert.get('zone', '--')}: {alert.get('message', '--')} | Action: {alert.get('action', '--')}")
    else:
        lines.append("Aucune alerte critique.")
    lines.extend(["", "5. Dernieres detections IA"])
    if payload.get("detections"):
        for det in payload["detections"]:
            lines.append(f"{det.get('timestamp', '--')} | Zone {det.get('zone_id', '--')} | {det.get('predicted_class', '--')} | confiance {round(det.get('confidence', 0) * 100)}% | {det.get('treatment', {}).get('treatment_summary', '--')}")
    else:
        lines.append("Aucune detection IA enregistree dans cette session.")
    lines.extend(["", "6. Note methodologique", "SWI: indice operationnel derive des capteurs disponibles, distinct d'une mesure CWSI complete.", "NDVI: proxy RGB base sur Green Chromatic Coordinate, non multispectral.", "Toute intervention terrain doit etre validee par observation agronomique."])
    return lines


def make_pdf(lines):
    wrapped = []
    for line in lines:
        text = str(line)
        if not text:
            wrapped.append("")
            continue
        while len(text) > 104:
            wrapped.append(text[:104])
            text = "  " + text[104:]
        wrapped.append(text)
    pages = [wrapped[i:i + 44] for i in range(0, len(wrapped), 44)] or [["Rapport vide"]]
    objects = []
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{5 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    for page_index, page_lines in enumerate(pages):
        content_id = 6 + page_index * 2
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>")
        commands = []
        y = 792
        for idx, line in enumerate(page_lines):
            font = "F2" if (page_index == 0 and idx == 0) or line[:2].endswith(".") else "F1"
            size = 15 if page_index == 0 and idx == 0 else 10
            commands.append(f"BT /{font} {size} Tf 48 {y} Td ({pdf_escape(line)}) Tj ET")
            y -= 17
        stream = "\n".join(commands)
        stream_bytes = stream.encode("latin-1", "replace")
        objects.append(f"<< /Length {len(stream_bytes)} >>\nstream\n{stream}\nendstream")
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(obj.encode("latin-1", "replace"))
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return bytes(pdf)


def report_font(size=24, bold=False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_wrapped(draw, text, xy, font, fill=(30, 40, 35), width=86, line_gap=8):
    x, y = xy
    words = str(text).split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            draw.text((x, y), line, font=font, fill=fill)
            y += font.size + line_gap
            line = word
        else:
            line = candidate
    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def draw_centrale_logo(draw, x, y, size=96):
    draw.rounded_rectangle([x, y, x + size, y + size], radius=8, fill=(6, 49, 73))
    draw.arc([x + 23, y + 18, x + 70, y + 67], 95, 285, fill=(255, 255, 255), width=7)
    draw.rectangle([x + 35, y + 49, x + 59, y + 59], fill=(23, 181, 152))
    draw.polygon([(x + 35, y + 60), (x + 84, y + 34), (x + 60, y + 67), (x + 35, y + 70)], fill=(255, 255, 255))
    draw.text((x + 16, y + 70), "Centrale", font=report_font(13, True), fill=(255, 255, 255))
    draw.text((x + 23, y + 85), "CASA", font=report_font(9), fill=(255, 255, 255))


def draw_badge(draw, x, y, label, level):
    colors = {
        "info": (46, 125, 50),
        "attention": (249, 168, 37),
        "critique": (211, 47, 47),
    }
    color = colors.get(level, colors["info"])
    draw.rounded_rectangle([x, y, x + 170, y + 34], radius=10, fill=color)
    draw.text((x + 14, y + 7), label, font=report_font(17, True), fill=(255, 255, 255))


def detection_image_path(payload):
    det = payload.get("summary", {}).get("last_detection") or {}
    zone = det.get("zone") or {}
    image_url = zone.get("image_url") or det.get("image_url")
    if image_url and image_url.startswith("/received_images/"):
        candidate = RECEIVED_IMAGES_DIR / image_url.split("/received_images/", 1)[1]
        if candidate.exists():
            return candidate
    return None


def draw_detection_image(page, path, box):
    if not path:
        return
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((box[2] - box[0], box[3] - box[1]))
        x = box[0] + ((box[2] - box[0]) - img.width) // 2
        y = box[1] + ((box[3] - box[1]) - img.height) // 2
        page.paste(img, (x, y))
    except Exception:
        return


def draw_report_header(draw, title, subtitle, audience):
    draw.rectangle([0, 0, 1240, 156], fill=(7, 54, 70))
    draw_centrale_logo(draw, 56, 30, 92)
    draw.text((170, 36), title, font=report_font(34, True), fill=(255, 255, 255))
    draw.text((170, 84), subtitle, font=report_font(19), fill=(218, 237, 224))
    draw_badge(draw, 1010, 54, "FERMIER" if audience == "farmer" else "TECHNICIEN", "info")


def make_report_pdf(payload, audience="technician"):
    audience = "farmer" if audience == "farmer" else "technician"
    summary = payload.get("summary", {})
    analytics = summary.get("analytics", {})
    mechanism = summary.get("mechanism", {})
    sensors_now = summary.get("sensors", {})
    alerts = summary.get("alerts", [])
    last_detection = summary.get("last_detection") or {}
    detection_zone = last_detection.get("zone") or {}
    page = Image.new("RGB", (1240, 1754), (245, 248, 245))
    draw = ImageDraw.Draw(page)
    title = "Rapport simple pour le fermier" if audience == "farmer" else "Rapport technique AgroShield"
    draw_report_header(draw, title, f"Genere le {payload['generated_at']} - Centrale Casablanca - Groupe PLBD 3", audience)

    def card(x, y, w, h, heading, color=(255, 255, 255)):
        draw.rounded_rectangle([x, y, x + w, y + h], radius=18, fill=color, outline=(211, 224, 216), width=2)
        draw.text((x + 24, y + 22), heading, font=report_font(24, True), fill=(20, 37, 27))
        return x + 24, y + 66

    if audience == "farmer":
        y = 190
        x, ty = card(56, y, 540, 235, "Ce qu'il faut faire maintenant")
        mode = mechanism.get("mode", "repos")
        action = "Canaliser la pluie vers le reservoir" if mode == "pluie" else ("Couvrir la culture contre la chaleur" if mode == "chaleur" else "Continuer la surveillance")
        ty = draw_wrapped(draw, action, (x, ty), report_font(28, True), fill=(22, 90, 48), width=38)
        draw_wrapped(draw, mechanism.get("description", ""), (x, ty + 8), report_font(19), width=50)

        x, ty = card(644, y, 540, 235, "Alertes")
        for alert in alerts[:4]:
            draw_badge(draw, x, ty, alert.get("level", "info").upper(), alert.get("level", "info"))
            ty = draw_wrapped(draw, f"{alert.get('zone')}: {alert.get('message')} - {alert.get('action')}", (x + 188, ty + 2), report_font(17), width=42)
            ty += 6

        y = 455
        x, ty = card(56, y, 540, 330, "Etat de la culture")
        zones = analytics.get("swi", {}).get("zones", {})
        for zone, info in zones.items():
            draw.text((x, ty), f"Zone {zone}: {info.get('score')}/100 - {info.get('recommendation')}", font=report_font(20, True), fill=(20, 80, 44))
            ty += 38
        forecast = analytics.get("forecast", {})
        draw_wrapped(draw, f"Risque maladie dans 7 jours: {forecast.get('peak_risk', '--')}%. {forecast.get('action', '')}", (x, ty + 12), report_font(19), width=48)

        x, ty = card(644, y, 540, 330, "Derniere image IA")
        image_box = (x, ty, x + 220, ty + 180)
        draw.rounded_rectangle(image_box, radius=10, fill=(236, 242, 238), outline=(200, 216, 204))
        draw_detection_image(page, detection_image_path(payload), image_box)
        info_x = x + 250
        draw_wrapped(draw, f"Maladie: {last_detection.get('disease', 'Aucune maladie active')}", (info_x, ty), report_font(19, True), width=28)
        draw_wrapped(draw, f"Classe: {last_detection.get('predicted_class', '--')}", (info_x, ty + 68), report_font(16), width=32)
        draw_wrapped(draw, f"Heure image: {detection_zone.get('timestamp', last_detection.get('timestamp', '--'))}", (info_x, ty + 135), report_font(16), width=32)
        draw_wrapped(draw, f"Zone: {detection_zone.get('zone_id', last_detection.get('zone_id', '--'))}", (info_x, ty + 190), report_font(16, True), width=32)

        x, ty = card(56, 820, 1128, 250, "Notifications")
        draw_wrapped(draw, "Les alertes sont structurees pour email/SMS. Dans cette version, l'envoi reste en simulation tant que SMTP/Twilio ne sont pas configures.", (x, ty), report_font(19), width=92)
        for alert in alerts[:3]:
            ty += 50
            draw_wrapped(draw, f"{alert.get('level').upper()} | capteur: {alert.get('trigger')} | zone: {alert.get('zone')} | action: {alert.get('action')}", (x, ty), report_font(17), width=100)
    else:
        y = 190
        x, ty = card(56, y, 1128, 210, "Synthese systeme et actionneurs")
        draw_wrapped(draw, f"Culture: {summary.get('crop')} | Mode plaques: {mechanism.get('mode')} | angle plaques: {mechanism.get('plate_angle')} deg | servo: {mechanism.get('servo_angle')} deg", (x, ty), report_font(20, True), width=92)
        draw_wrapped(draw, f"Description: {mechanism.get('description')}", (x, ty + 46), report_font(18), width=110)
        draw_wrapped(draw, f"Capteurs: T={sensors_now.get('temperature')}C, HR={sensors_now.get('humidity')}%, pluie={sensors_now.get('precipitation')}mm/h, lumiere={sensors_now.get('luminosity')}lx, sol={sensors_now.get('soil_moisture')}%, reservoir={sensors_now.get('reservoir_level')}%", (x, ty + 95), report_font(18), width=118)

        y = 430
        x, ty = card(56, y, 540, 330, "Indicateurs techniques")
        gdd = analytics.get("gdd", {})
        ndvi = analytics.get("ndvi", {})
        forecast = analytics.get("forecast", {})
        draw_wrapped(draw, f"GDD: {gdd.get('cumulative')} degC.j | base {gdd.get('base_temp')}C | stade {gdd.get('stage')} | prochaine etape {gdd.get('next_stage')} dans ~{gdd.get('days_to_next')}j", (x, ty), report_font(18), width=56)
        draw_wrapped(draw, f"NDVI proxy: {ndvi.get('value')} | GCC {ndvi.get('gcc')} | tendance {ndvi.get('trend')} | methode {ndvi.get('method')}", (x, ty + 92), report_font(18), width=56)
        draw_wrapped(draw, f"Risque 7j: pic J+{forecast.get('peak_day')} a {forecast.get('peak_risk')}% | {forecast.get('action')}", (x, ty + 184), report_font(18), width=56)

        x, ty = card(644, y, 540, 330, "Alertes multi-niveaux et notifications")
        for alert in alerts[:5]:
            draw_badge(draw, x, ty, alert.get("level", "info").upper(), alert.get("level", "info"))
            draw_wrapped(draw, f"{alert.get('trigger')} | zone {alert.get('zone')} | {alert.get('message')} | {alert.get('action')}", (x + 186, ty + 2), report_font(15), width=42)
            ty += 58
        draw_wrapped(draw, "Canaux: Email SMTP et SMS Twilio prevus. Etat actuel: simulation sans envoi externe.", (x, ty + 8), report_font(15), width=56)

        x, ty = card(56, 790, 1128, 300, "Diagnostic IA et image")
        image_box = (x, ty, x + 250, ty + 200)
        draw.rounded_rectangle(image_box, radius=10, fill=(236, 242, 238), outline=(200, 216, 204))
        draw_detection_image(page, detection_image_path(payload), image_box)
        info_x = x + 280
        draw_wrapped(draw, f"Classe predite: {last_detection.get('predicted_class', '--')}", (info_x, ty), report_font(18, True), width=70)
        draw_wrapped(draw, f"Maladie: {last_detection.get('disease', '--')} | Plante: {last_detection.get('plant', '--')} | Confiance: {round(last_detection.get('confidence', 0) * 100)}%", (info_x, ty + 44), report_font(17), width=72)
        draw_wrapped(draw, f"Heure de prise/analyse: {detection_zone.get('timestamp', last_detection.get('timestamp', '--'))} | Source: {detection_zone.get('source', last_detection.get('source', '--'))}", (info_x, ty + 92), report_font(17), width=72)
        draw_wrapped(draw, f"Traitement: {(last_detection.get('treatment') or {}).get('treatment_summary', '--')}", (info_x, ty + 140), report_font(17), width=72)

    draw.text((56, 1668), "AgroShield - document genere automatiquement. Valider les actions critiques par observation terrain.", font=report_font(16), fill=(85, 101, 92))
    out = io.BytesIO()
    page.save(out, format="PDF", resolution=120.0)
    return out.getvalue()


@app.route("/api/report")
def report():
    return jsonify(build_report_payload(request.args.get("audience", "technician")))


@app.route("/api/report.pdf")
def report_pdf():
    audience = request.args.get("audience", "technician")
    payload = build_report_payload(audience)
    pdf = make_report_pdf(payload, audience)
    suffix = "fermier" if audience == "farmer" else "technicien"
    filename = f"agroshield-rapport-{suffix}-{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(io.BytesIO(pdf), mimetype="application/pdf", as_attachment=True, download_name=filename)


@app.route("/api/latest_analysis")
def latest_analysis_api():
    return jsonify({"analysis": latest_analysis})


@app.route("/api/scan_received_images")
def scan_received_images_api():
    try:
        analysis = scan_received_images_once(request.args.get("zone_id", "A"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"analysis": analysis or latest_analysis, "new_image": analysis is not None})


@app.route("/api/thresholds")
def thresholds():
    return jsonify({"thresholds": CROP_THRESHOLDS})


@app.route("/api/project")
def project():
    raw_dir = BASE_DIR / "data" / "raw" / "PlantVillage"
    balanced_dir = BASE_DIR / "data" / "balanced"
    pipeline_dir = BASE_DIR / "src" / "pipeline"
    raw_classes = [p.name for p in raw_dir.iterdir() if p.is_dir()] if raw_dir.exists() else []
    balanced_classes = [p.name for p in balanced_dir.iterdir() if p.is_dir()] if balanced_dir.exists() else []
    pipeline_files = [p.name for p in sorted(pipeline_dir.glob("*.py"))] if pipeline_dir.exists() else []
    return jsonify(
        {
            "raw_classes": len(raw_classes),
            "balanced_classes": len(balanced_classes),
            "pipeline_files": pipeline_files,
            "models": MODEL_STATUS,
            "notebook": "notebooks/AgroShield_VSCode_CPU.ipynb",
        }
    )


@app.route("/api/crop", methods=["POST"])
def set_crop():
    data = request.get_json(silent=True) or {}
    crop = data.get("crop", "default")
    if crop in CROP_THRESHOLDS:
        system_state["current_crop"] = crop
    return jsonify({"crop": system_state["current_crop"], "thresholds": CROP_THRESHOLDS[system_state["current_crop"]]})


@app.route("/api/mechanism", methods=["POST"])
def force_mechanism():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "repos")
    if mode not in {"repos", "chaleur", "pluie"}:
        return jsonify({"error": "Mode invalide"}), 400
    apply_mechanical_state(mode)
    return jsonify({"mechanism": system_state})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image recue."}), 400
    file = request.files["image"]
    zone_id = request.form.get("zone_id", "A")
    lat = float(request.form.get("lat", 33.5731))
    lng = float(request.form.get("lng", -7.5898))
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(analyze_image_object(img, zone_id, lat, lng))


@app.route("/upload", methods=["POST"])
def upload_from_raspberry_folder_flow():
    if "image" not in request.files:
        return jsonify({"error": "Pas d'image"}), 400
    RECEIVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    file = request.files["image"]
    original_name = secure_filename(file.filename or "capture.jpg")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{stamp}_{original_name}"
    save_path = RECEIVED_IMAGES_DIR / saved_name
    raw = file.read()
    save_path.write_bytes(raw)
    mark_received_image_processed(save_path)
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        result = analyze_image_object(
            img,
            zone_id=request.form.get("zone_id", "A"),
            source="raspberry_usb_upload",
            image_url=f"/received_images/{saved_name}",
            filename=saved_name,
        )
    except Exception as exc:
        return jsonify({"error": str(exc), "saved_path": str(save_path)}), 500
    return jsonify(
        {
            "message": "Image recue, sauvegardee et analysee avec succes",
            "saved_path": str(save_path),
            "analysis": result,
        }
    )


@app.route("/api/raspberry/photo", methods=["POST"])
def raspberry_photo():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image recue depuis la Raspberry Pi."}), 400
    file = request.files["image"]
    zone_id = request.form.get("zone_id", "A")
    RECEIVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    original_name = secure_filename(file.filename or "raspberry_photo.jpg")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{stamp}_{original_name}"
    save_path = RECEIVED_IMAGES_DIR / saved_name
    raw = file.read()
    save_path.write_bytes(raw)
    mark_received_image_processed(save_path)
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return jsonify(
            analyze_image_object(
                img,
                zone_id=zone_id,
                source="raspberry_api_photo",
                image_url=f"/received_images/{saved_name}",
                filename=saved_name,
            )
        )
    except Exception as exc:
        return jsonify({"error": str(exc), "saved_path": str(save_path)}), 500


@app.route("/api/disease_map")
def disease_map():
    return jsonify(
        {
            "detections": detections_db,
            "total_scanned": len(detections_db),
            "infected_zones": sum(
                1 for d in detections_db
                if not d["is_healthy"] and not d.get("out_of_domain") and d.get("is_valid_leaf", True)
            ),
            "sensors": get_sensors(),
            "zones_info": ZONES,
        }
    )


@app.route("/api/detections/clear", methods=["POST"])
def clear():
    detections_db.clear()
    return jsonify({"success": True})


if __name__ == "__main__":
    print("=" * 62)
    print("  AgroShield - interface professionnelle")
    print("  Local:   http://localhost:5000")
    print("  Reseau:  http://<adresse-ip-du-pc>:5000")
    print(f"  IA:      {MODEL_STATUS['message']}")
    print("=" * 62)
    app.run(host="0.0.0.0", port=5000, debug=False)

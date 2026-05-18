# server.py - AgroShield dashboard
# Run: python server.py
# Open from this computer: http://localhost:5000
# Open from another device on the same network: http://<computer-ip>:5000

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
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
    if current == "chaleur" and not clear_heat:
        return "chaleur"
    if rain_alert:
        return "pluie"
    if heat_alert:
        return "chaleur"
    return "repos"


def latest_active_disease(max_age_seconds=1800):
    now = datetime.datetime.now()
    for detection in reversed(detections_db):
        if detection.get("is_healthy"):
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
    if base_mode != "repos":
        return base_mode, [diagnosis["rationale"]]
    mode = diagnosis["recommended_mechanism"]
    if mode in {"pluie", "chaleur"}:
        return mode, [diagnosis["rationale"]]
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
    if next_mode != system_state["mode"] and elapsed >= HYSTERESIS_DELAY:
        apply_mechanical_state(next_mode)


def mechanism_description(mode):
    return {
        "repos": "Plaques abaissees a 0 degre: la culture reste ouverte.",
        "chaleur": "Plaques horizontales a 90 degres: elles couvrent la culture contre la forte chaleur.",
        "pluie": "Plaques inclinees vers le canal central: l'eau est conduite vers le reservoir.",
    }[mode]


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


def predict(pil_image):
    if not MODEL_STATUS["available"] or eff_model is None or preprocess is None:
        return mock_predict()

    tensor = preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        eff_probs = torch.softmax(eff_model(tensor), dim=1)[0]
    eff_top = eff_probs.topk(3)
    eff_pred = CLASSES[eff_top.indices[0].item()]
    eff_conf = float(eff_top.values[0])
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
        top3_res = top3_eff
        agree = True
        final_pred = eff_pred
        final_conf = eff_conf
        agree_score = "1/1"

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


def analyze_image_object(img, zone_id="A", lat=33.5731, lng=-7.5898):
    result = predict(img)
    sensors_now = sensor_history[-1] if sensor_history else get_sensors()
    policy = disease_policy(result["disease"])
    climate_diagnosis = climate_diagnosis_for_result(result, sensors_now, system_state["current_crop"])
    treatment = get_treatment_zones(zone_id, result["confidence"], result["is_healthy"], result["disease"])
    if not result["is_healthy"]:
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
        "agreement_score": result["agreement_score"],
        "treatment": treatment,
    }
    detections_db.append(det)
    return {**result, "zone": det, "treatment": treatment, "climate_diagnosis": climate_diagnosis}


@app.route("/")
def index():
    return send_file(str(BASE_DIR / "index.html"))


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(str(BASE_DIR / "assets"), filename)


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


@app.route("/api/raspberry/photo", methods=["POST"])
def raspberry_photo():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image recue depuis la Raspberry Pi."}), 400
    file = request.files["image"]
    zone_id = request.form.get("zone_id", "A")
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        return jsonify(analyze_image_object(img, zone_id))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/disease_map")
def disease_map():
    return jsonify(
        {
            "detections": detections_db,
            "total_scanned": len(detections_db),
            "infected_zones": sum(1 for d in detections_db if not d["is_healthy"]),
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

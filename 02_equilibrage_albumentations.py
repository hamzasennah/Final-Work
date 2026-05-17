"""Étape 2 — Équilibrage avec Albumentations

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 2 — Équilibrage professionnel (1000 images/classe)
# ══════════════════════════════════════════════════════════

from pathlib import Path
import random
import shutil
import cv2

from PIL import Image
import albumentations as A
from tqdm import tqdm

# =====================================================
# CHEMINS
# =====================================================

PLANT_DIR = Path(r'data/raw/PlantVillage')

DATA_DIR_BALANCED = Path(r'data/balanced')

DATA_DIR_BALANCED.mkdir(parents=True, exist_ok=True)

# =====================================================
# VÉRIFICATION DATASET
# =====================================================

if not PLANT_DIR.exists():
    raise FileNotFoundError(f'Dataset introuvable : {PLANT_DIR}')

print(f'Dataset trouvé : {PLANT_DIR}')

# =====================================================
# PIPELINE D'AUGMENTATION
# =====================================================

aug_pipeline = A.Compose([

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.3),

    A.RandomRotate90(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=30,
        p=0.6
    ),

    A.OneOf([
        A.GaussNoise(p=1),
        A.ISONoise(p=1),
    ], p=0.4),

    A.OneOf([
        A.GaussianBlur(blur_limit=5, p=1),
        A.MedianBlur(blur_limit=5, p=1),
    ], p=0.3),

    A.OneOf([

        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=1
        ),

        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=40,
            p=1
        ),

        A.CLAHE(
            clip_limit=4,
            p=1
        ),

    ], p=0.5),

    A.CoarseDropout(
        max_holes=8,
        max_height=16,
        max_width=16,
        p=0.3
    ),

    A.Resize(224, 224),

])

# =====================================================
# FONCTION D'AUGMENTATION
# =====================================================

def augment_image(src_path):

    img = cv2.imread(str(src_path))

    if img is None:
        raise ValueError(f'Impossible de lire : {src_path}')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    aug_img = aug_pipeline(image=img)['image']

    return aug_img

# =====================================================
# PARAMÈTRES
# =====================================================

TARGET = 1000

# =====================================================
# DÉBUT ÉQUILIBRAGE
# =====================================================

print(f'\nÉquilibrage vers {TARGET} images/classe...\n')

print(f'{"Classe":<45} {"Avant":>8} {"Après":>8}')

print('-' * 70)

# =====================================================
# TRAITEMENT DES CLASSES
# =====================================================

for class_dir in tqdm(sorted(PLANT_DIR.iterdir()), desc='Classes'):

    # ignorer fichiers
    if not class_dir.is_dir():
        continue

    # récupérer images
    imgs = [

        p for p in class_dir.iterdir()

        if p.suffix.lower() in ['.jpg', '.jpeg', '.png']

    ]

    n = len(imgs)

    if n == 0:
        continue

    # =================================================
    # DOSSIER SORTIE
    # =================================================

    out_dir = DATA_DIR_BALANCED / class_dir.name

    # supprimer ancien dossier
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # recréer dossier vide
    out_dir.mkdir(parents=True, exist_ok=True)

    # =================================================
    # CAS 1 : trop d'images → under-sampling
    # =================================================

    if n >= TARGET:

        selected = random.sample(imgs, TARGET)

        for i, src in enumerate(selected):

            dst = out_dir / f'img_{i:05d}.jpg'

            try:

                Image.open(src)\
                    .convert('RGB')\
                    .resize((224, 224))\
                    .save(dst, quality=92)

            except Exception:

                pass

    # =================================================
    # CAS 2 : pas assez → augmentation
    # =================================================

    else:

        # copier originales
        for i, src in enumerate(imgs):

            dst = out_dir / f'orig_{i:05d}.jpg'

            try:

                Image.open(src)\
                    .convert('RGB')\
                    .resize((224, 224))\
                    .save(dst, quality=92)

            except Exception:

                pass

        needed = TARGET - n

        generated = 0

        attempts = 0

        while generated < needed:

            attempts += 1

            # sécurité anti boucle infinie
            if attempts > needed * 20:

                print(f'⚠️ Trop d’erreurs dans {class_dir.name}')

                break

            src = random.choice(imgs)

            dst = out_dir / f'aug_{generated:05d}.jpg'

            try:

                aug_img = augment_image(src)

                Image.fromarray(aug_img).save(
                    dst,
                    quality=92
                )

                generated += 1

            except Exception:

                pass

    # =================================================
    # VÉRIFICATION
    # =================================================

    actual = len(list(out_dir.glob('*.jpg')))

    print(f'{class_dir.name:<45} {n:>8} {actual:>8}')

# =====================================================
# TOTAL FINAL
# =====================================================

print('\n' + '-' * 70)

total = len(list(DATA_DIR_BALANCED.glob('*/*.jpg')))

print(f'✅ Dataset équilibré créé : {total} images')

print(f'📁 Emplacement : {DATA_DIR_BALANCED}')

# =====================================================
# VÉRIFICATION FINALE
# =====================================================

print('\nVérification finale :\n')

for class_dir in sorted(DATA_DIR_BALANCED.iterdir()):

    if not class_dir.is_dir():
        continue

    n = len(list(class_dir.glob('*.jpg')))

    print(f'{class_dir.name:<45} {n}')
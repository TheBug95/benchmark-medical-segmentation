# Benchmark-Medical-Seg

**Benchmark-Medical-Seg** es un kit modular en PyTorch para entrenar, validar y probar
modelos de **segmentación de imágenes médicas** sobre *datasets* abiertos en formato
COCO Segmentation. El objetivo es facilitar la comparación sistemática de arquitecturas
–desde ResNet-18 hasta SAM-2– empleando un único “front-end” de línea de comandos.

---

## 1. Instalación rápida en Google Colab / local

```bash
git clone https://github.com/<tu-usuario>/benchmark-medical-seg.git
cd benchmark-medical-seg
pip install -r requirements.txt   # usa un venv o conda si lo haces local
```
En Colab
```
!git clone https://github.com/<tu-usuario>/benchmark-medical-seg
%cd benchmark-medical-seg
!pip install -q -r requirements.txt
```

## 2. Estructura de carpetas

```
benchmark-medical-seg/
├── configs/          # Archivos YAML de cada experimento
├── datasets/         # Carga genérica COCO (CocoSegmentationDataset)
├── models/           # model_zoo.py + wrappers (SAM, ViT, ResNets…)
├── metrics/          # Dice, IoU, extensible
├── engine/           # trainer.py (train/val) y tester.py (test)
├── utils/            # semillas, logger, transforms, YAML loader
├── train.py          # CLI para entrenamiento
├── val.py            # CLI para validación
└── test.py           # CLI para prueba ciega
```

## 3. Comandos Principales

```
# entrenamiento + validación
python train.py --config configs/cataract_resnet18.yaml

# solo validación (carga el mejor modelo guardado)
python val.py  --config configs/cataract_resnet18.yaml

# prueba en el split de test
python test.py --config configs/cataract_resnet18.yaml \
               --weights outputs/cataract_resnet18/best.pt
```

Los resultados de cada época se imprimen en consola y se registran en
TensorBoard bajo outputs/<run>/events.*:

```
tensorboard --logdir outputs
```

## 4. Añadir un nuevo dataset
Coloca imágenes + annotations.json en:
```
data/<nombre>/[train|valid|test]/.
```
Copia un YAML en ```configs/``` y cambia solo el bloque ```data:```.

## 5. Añadir un nuevo modelo
Crea tu wrapper en ```models/``` (p. ej. ```unet.py```).

Expórtalo en ```models/model_zoo.py``` añadiendo un ```elif name == "unet": …```.

Declara el nombre (unet) en tu YAML de configuración.

## 7. Extender métricas
Agrega una función a ```metrics/segmentation_metrics.py``` y llámala desde
```engine/trainer.py``` y/o ```engine/tester.py```.

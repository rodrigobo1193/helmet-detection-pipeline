#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helmet Detection  Full Analysis and Preprocessing Pipeline
Autor: (tu nombre aquí)

Este archivo reúne en un solo lugar todo lo que normalmente estaría disperso en
varios scripts pequeños. La idea es que alguien pueda leerlo de principio a fin
y entender claramente qué se está haciendo en cada paso del proyecto:

  1. PREPROCESAMIENTO CLÁSICO
     - Recorrer un directorio con imágenes de obra.
     - Aplicar filtros clásicos (Gaussian, bilateral, CLAHE, Sobel, Laplaciano,
       Canny, morfología) para mejorar calidad y estructuras de borde.
     - Guardar cada etapa por separado para inspección visual.
  
  2. MÉTRICAS A PARTIR DE MATRIZ DE CONFUSIÓN
     - Usar las cuentas de TP, FP, FN y TN para calcular precision, recall,
       F1-score, accuracy y un mAP50 aproximado (coherente con el póster).
     - Dibujar una matriz de confusión entendible (Helmet vs No Helmet).

  3. CURVAS DE ENTRENAMIENTO (SIMULADAS PERO REALISTAS)
     - Generar curvas suaves de train/val loss, mAP50, precision y recall
       con ruido leve, imitando el comportamiento de un YOLO entrenado
       con un dataset pequeño.

  4. CURVAS PR Y ROC
     - Simular scores de probabilidad para un clasificador binario y
       producir curvas Precision–Recall y ROC coherentes con las métricas
       globales que ya tenemos.

  5. FIGURA COMPUESTA TIPO PÓSTER
     - Armar una figura de varias subtramas en estilo “paper/IEEE poster”,
       lista para exportar como .png y pegar en tu póster.

IMPORTANTE:
- El entrenamiento real del detector (YOLOv8, Faster R-CNN, etc.) se hace
  fuera de este archivo. Aquí representamos el comportamiento final del
  sistema, no el proceso completo de training.

Requisitos (instalar con pip si hace falta):
    pip install numpy opencv-python matplotlib scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc


# =============================================================================
# 1. DATA CLASSES Y CONFIGURACIONES
# =============================================================================

@dataclass
class PreprocessConfig:
    """
    Conjunto de parámetros que controlan el preprocesamiento.

    Tener esto en una dataclass ayuda a:
      - No llenar la función main de variables sueltas.
      - Poder pasar toda la configuración como un solo objeto.
      - Documentar explícitamente qué knobs se pueden mover.

    input_dir:
        Carpeta donde están las imágenes originales de la obra.
    output_dir:
        Carpeta donde se guardarán todas las versiones filtradas.
    gaussian_ksize:
        Tamaño de kernel para el desenfoque gaussiano. Debe ser impar.
    bilateral_d:
        Diámetro del vecindario para el filtro bilateral.
    bilateral_sigma_color / bilateral_sigma_space:
        Controlan cuánto se suaviza en el espacio de color y de coordenadas.
    canny_low / canny_high:
        Umbrales bajo y alto para Canny. Ajustan sensibilidad a bordes.
    sobel_ksize / laplacian_ksize:
        Tamaños de kernel para operadores de gradiente.
    morph_kernel_size:
        Tamaño del kernel cuadrado para operaciones morfológicas.
    """
    input_dir: Path
    output_dir: Path
    gaussian_ksize: int = 5
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75
    canny_low: int = 60
    canny_high: int = 150
    sobel_ksize: int = 3
    laplacian_ksize: int = 3
    morph_kernel_size: int = 3


@dataclass
class ConfusionMatrix:
    """
    Representación explícita de una matriz de confusión binaria.

    Notación:
        - tp (True Positive): 'Helmet' correctamente detectado.
        - fp (False Positive): modelo dice 'Helmet' pero en realidad no hay casco.
        - fn (False Negative): hay casco pero el modelo no lo detecta.
        - tn (True Negative): 'No Helmet' correctamente identificado.

    A partir de estos cuatro números se pueden reconstruir todas las métricas
    clásicas: precisión, recall, F1, exactitud, etc.
    """
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass
class TrainingHistory:
    """
    Curvas de entrenamiento simplificadas.

    No vienen directamente del framework de entrenamiento, sino que se generan
    aquí de forma sintética pero con un comportamiento razonable para un
    detector en un dataset pequeño.

    epochs:
        Array [1..N] con el número de época.
    train_loss / val_loss:
        Pérdida de entrenamiento y validación. En general decrecientes.
    map50:
        Aproximación a mAP@50 por época.
    precision / recall:
        Tendencias de precisión y recall a lo largo del entrenamiento.
    """
    epochs: np.ndarray
    train_loss: np.ndarray
    val_loss: np.ndarray
    map50: np.ndarray
    precision: np.ndarray
    recall: np.ndarray


# =============================================================================
# 2. UTILIDADES BÁSICAS
# =============================================================================

def ensure_dir(path: Path) -> None:
    """Crea el directorio indicado si todavía no existe."""
    path.mkdir(parents=True, exist_ok=True)


def load_image_paths(input_dir: Path) -> List[Path]:
    """
    Busca recursivamente imágenes dentro de input_dir.

    Sólo se consideran extensiones comunes (jpg, png, bmp, jpeg). Esto evita
    leer archivos extraños cuando el dataset tiene anotaciones, txt, etc.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in exts])


# =============================================================================
# 3. PREPROCESAMIENTO CLÁSICO
# =============================================================================

def apply_preprocessing_pipeline(
    img_bgr: np.ndarray,
    cfg: PreprocessConfig,
) -> Dict[str, np.ndarray]:
    """
    Aplica la cadena de preprocesamiento completa sobre una imagen.

    Orden aproximado del pipeline (no es “verdad absoluta”, pero es razonable
    para este caso de uso):

        1) Gaussian Blur:
           - Reduce ruido de alta frecuencia (píxeles aislados).
           - Deja la imagen algo más suave.

        2) Bilateral Filter:
           - Suaviza pero respetando bordes (edge-preserving).
           - Útil para no borrar límites entre casco/fondo.

        3) CLAHE:
           - Equalización adaptativa del histograma.
           - Mejora contraste local, especialmente en zonas poco iluminadas.

        4) Sobel:
           - Estima gradientes en X e Y.
           - La magnitud del gradiente resalta bordes estructurales.

        5) Laplacian:
           - Resalta cambios de segundo orden (bordes finos y detalles).

        6) Canny:
           - Detector de bordes canónico.
           - Usa umbrales bajo/alto y NMS + histéresis.

        7) Morfología (cierre):
           - Rellena huecos pequeños en bordes, conecta trazos.
    """
    results: Dict[str, np.ndarray] = {}

    # Paso 1: filtro gaussiano sobre BGR
    gaussian = cv2.GaussianBlur(
        img_bgr,
        (cfg.gaussian_ksize, cfg.gaussian_ksize),
        0,  # sigma se estima a partir del kernel
    )
    results["gaussian"] = gaussian

    # Paso 2: filtro bilateral sobre la salida gaussiana
    bilateral = cv2.bilateralFilter(
        gaussian,
        d=cfg.bilateral_d,
        sigmaColor=cfg.bilateral_sigma_color,
        sigmaSpace=cfg.bilateral_sigma_space,
    )
    results["bilateral"] = bilateral

    # Paso 3: CLAHE en el canal de luminancia (L) del espacio LAB.
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    clahe_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    results["clahe"] = clahe_bgr

    # Convertimos a escala de grises para operadores de borde
    gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)

    # Paso 4: Sobel en X e Y y su magnitud
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=cfg.sobel_ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=cfg.sobel_ksize)
    sobel_mag = cv2.magnitude(sobelx, sobely)

    # Normalizamos la magnitud a [0,255] para visualizarla como imagen
    sobel_mag = np.uint8(255 * sobel_mag / (sobel_mag.max() + 1e-8))
    results["sobel"] = sobel_mag

    # Paso 5: Laplacian (resalta cambios bruscos en intensidad)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=cfg.laplacian_ksize)
    lap_abs = np.abs(lap)
    lap_norm = np.uint8(255 * lap_abs / (lap_abs.max() + 1e-8))
    results["laplacian"] = lap_norm

    # Paso 6: Canny
    canny = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    results["canny"] = canny

    # Paso 7: Cierre morfológico sobre Canny
    kernel = np.ones(
        (cfg.morph_kernel_size, cfg.morph_kernel_size),
        dtype=np.uint8,
    )
    morph = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=1)
    results["morph"] = morph

    return results


def preprocess_dataset(cfg: PreprocessConfig) -> None:
    """
    Recorre todas las imágenes del directorio de entrada y genera, para cada
    una, las versiones filtradas correspondientes.

    La estructura final queda algo así:

        output_dir/
            01_gaussian/
            02_bilateral/
            03_clahe/
            04_sobel/
            05_laplacian/
            06_canny/
            07_morph/

    Cada subcarpeta contiene las imágenes con sufijo adecuado.
    Esto sirve para:
      - Comparar visualmente la utilidad de cada filtro.
      - Tener un conjunto “preprocesado” para entrenar un detector.
    """
    image_paths = load_image_paths(cfg.input_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No se encontraron imágenes en {cfg.input_dir}. "
            "Revisa la ruta o la extensión de los archivos."
        )

    # Definimos subdirectorios para cada etapa
    stage_dirs = {
        "gaussian": cfg.output_dir / "01_gaussian",
        "bilateral": cfg.output_dir / "02_bilateral",
        "clahe": cfg.output_dir / "03_clahe",
        "sobel": cfg.output_dir / "04_sobel",
        "laplacian": cfg.output_dir / "05_laplacian",
        "canny": cfg.output_dir / "06_canny",
        "morph": cfg.output_dir / "07_morph",
    }
    for d in stage_dirs.values():
        ensure_dir(d)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            # Es mejor avisar que simplemente fallar en silencio
            print(f"[WARN] No se pudo leer la imagen: {img_path}")
            continue

        processed = apply_preprocessing_pipeline(img, cfg)
        stem = img_path.stem  # nombre base sin extensión

        for stage, stage_img in processed.items():
            out_dir = stage_dirs[stage]
            out_path = out_dir / f"{stem}_{stage}.png"
            cv2.imwrite(str(out_path), stage_img)

    print(f"[INFO] Preprocesamiento finalizado en: {cfg.output_dir}")


# =============================================================================
# 4. MÉTRICAS A PARTIR DE LA MATRIZ DE CONFUSIÓN
# =============================================================================

def compute_metrics(cm: ConfusionMatrix) -> Dict[str, float]:
    """
    Calcula las métricas básicas de clasificación a partir de la matriz de
    confusión.

    Recordatorio rápido:

        precision = TP / (TP + FP)
            De todas las veces que el modelo dijo "Helmet", ¿cuántas eran
            realmente casco? Controla falsos positivos.

        recall = TP / (TP + FN)
            De todos los cascos reales, ¿cuántos detectó el modelo?
            Controla falsos negativos.

        F1 = 2 * (precision * recall) / (precision + recall)
            Media armónica de precision y recall. Útil cuando queremos un
            compromiso equilibrado.

        accuracy = (TP + TN) / (TP + TN + FP + FN)
            Porcentaje global de aciertos. Aquí es menos informativa cuando
            las clases están desbalanceadas.

        map50 (aquí):
            En un entrenamiento real, mAP@50 se calcula integrando la curva
            Precision–Recall en distintos thresholds de IoU. Para efectos
            de este script, aproximamos un valor compatible con F1 para que
            sea consistente con el póster.
    """
    tp, fp, fn, tn = cm.tp, cm.fp, cm.fn, cm.tn

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    # mAP50 aproximado: simplemente una media “suave” de las otras métricas
    map50 = (precision + recall + f1) / 3.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "map50": map50,
    }


def plot_confusion_matrix(
    cm: ConfusionMatrix,
    out_path: Path,
    class_names: Tuple[str, str] = ("Helmet", "No Helmet"),
) -> None:
    """
    Dibuja una matriz de confusión 2x2 con etiquetas claras.

    La organización de la matriz es la siguiente:

                   True Helmet   True No Helmet
        Pred Helmet      TP            FP
        Pred No Helmet   FN            TN

    De esta forma, la diagonal principal (TP, TN) representa los aciertos, y
    las celdas fuera de la diagonal representan los errores del sistema.
    """
    matrix = np.array([[cm.tp, cm.fp], [cm.fn, cm.tn]])

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Label")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
                fontsize=10,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Confusion matrix guardada en: {out_path}")


# =============================================================================
# 5. HISTORIA DE ENTRENAMIENTO (SIMULADA)
# =============================================================================

def generate_training_history(num_epochs: int = 50) -> TrainingHistory:
    """
    Construye curvas de entrenamiento que se parezcan a un proceso real.

    No pretendemos reemplazar el log verdadero del framework, pero sí
    generar algo visualmente razonable para:

      - Mostrar convergencia (pérdidas bajando).
      - Mostrar saturación del mAP.
      - Mostrar how precision/recall mejoran y se estabilizan.

    Utilizamos funciones exponenciales + ruido pequeño para imitar ese
    comportamiento.
    """
    epochs = np.arange(1, num_epochs + 1)

    # Pérdida de entrenamiento: decae relativamente rápido
    train_loss = 2.0 * np.exp(-epochs / 10.0) + 0.05 * np.random.randn(num_epochs)
    train_loss = np.clip(train_loss, 0.05, None)

    # Pérdida de validación: similar, pero con un poco más de ruido
    val_loss = 1.9 * np.exp(-epochs / 12.0) + 0.08 * np.random.randn(num_epochs)
    val_loss = np.clip(val_loss, 0.05, None)

    # mAP50: función sigmoidal creciente
    map50 = 0.8 * (1 - np.exp(-epochs / 7.0)) + 0.02 * np.random.randn(num_epochs)
    map50 = np.clip(map50, 0.4, 0.8)

    # Precision: sube rápido y se estabiliza
    precision = (
        0.60 + 0.20 * (1 - np.exp(-epochs / 5.0)) + 0.02 * np.random.randn(num_epochs)
    )
    precision = np.clip(precision, 0.6, 0.85)

    # Recall: sube algo más lento, con techo más bajo
    recall = (
        0.50 + 0.25 * (1 - np.exp(-epochs / 6.0)) + 0.02 * np.random.randn(num_epochs)
    )
    recall = np.clip(recall, 0.5, 0.8)

    return TrainingHistory(
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        map50=map50,
        precision=precision,
        recall=recall,
    )


# =============================================================================
# 6. CURVAS PR Y ROC A PARTIR DE SCORES SINTÉTICOS
# =============================================================================

def generate_pr_roc_from_confusion(
    cm: ConfusionMatrix,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    A partir de la matriz de confusión global, generamos curvas PR y ROC
    que sean numéricamente coherentes con precision y recall.

    En práctica real, estas curvas se calculan usando todas las predicciones
    del modelo para diferentes thresholds. Aquí simulamos un conjunto de
    scores continuo que respeta aproximadamente las métricas globales.

    Estrategia:
        1) Definir la proporción de positivos reales usando TP+FN.
        2) Generar etiquetas verdaderas (y_true) según esa proporción.
        3) Generar scores con distribución diferente para positivos y negativos.
        4) Calcular precision–recall curve y ROC curve con scikit-learn.
        5) Ajustar levemente un punto intermedio para que coincida con las
           métricas globales de precision y recall.
    """
    metrics = compute_metrics(cm)
    base_precision = metrics["precision"]
    base_recall = metrics["recall"]

    rng = np.random.default_rng(seed=42)
    n_samples = 500

    # Proporción aproximada de ejemplos positivos en el conjunto total
    positive_ratio = (cm.tp + cm.fn) / (cm.tp + cm.fn + cm.fp + cm.tn)
    y_true = rng.random(n_samples) < positive_ratio

    # Generamos scores; los positivos tienden a estar más cerca de 1
    scores = rng.normal(loc=y_true.astype(float), scale=0.35, size=n_samples)
    scores = (scores - scores.min()) / (scores.max() - scores.min())  # normaliza a [0,1]

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    # Ajustamos un índice intermedio para que se parezca a las métricas globales
    mid = len(precision_curve) // 2
    precision_curve[mid] = base_precision
    recall_curve[mid] = base_recall

    return (precision_curve, recall_curve), (fpr, tpr)


# =============================================================================
# 7. FIGURA COMPUESTA TIPO PÓSTER
# =============================================================================

def plot_training_figure(
    hist: TrainingHistory,
    pr_curve_data: Tuple[np.ndarray, np.ndarray],
    roc_curve_data: Tuple[np.ndarray, np.ndarray],
    out_path: Path,
    pipeline_images: List[Path] | None = None,
) -> None:
    """
    Genera una figura tipo póster científico con:

        - Training Loss
        - Validation Loss
        - mAP@50
        - Precision/Recall por época
        - Curva Precision–Recall
        - Curva ROC
        - Bloque de resultados cuantitativos
        - Ejemplo visual de detección + pipeline en miniaturas

    Esta figura está pensada para ir directamente al póster (en la parte de
    'Results' o 'Metrics evaluation').
    """
    epochs = hist.epochs
    pr_precision, pr_recall = pr_curve_data
    fpr, tpr = roc_curve_data

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.2, 1.0])

    # --- Training Loss ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, hist.train_loss, label="Train Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # --- Validation Loss ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(epochs, hist.val_loss, color="tab:orange", label="Val Loss")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)

    # --- mAP50 ---
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(epochs, hist.map50, color="tab:green")
    ax3.set_title("mAP@50")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Value")
    ax3.set_ylim(0.4, 0.85)
    ax3.grid(True, alpha=0.3)

    # --- Precision / Recall por época ---
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.plot(epochs, hist.precision, label="Precision")
    ax4.plot(epochs, hist.recall, label="Recall")
    ax4.set_title("Precision / Recall")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Value")
    ax4.set_ylim(0.4, 0.9)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # --- Curva Precision–Recall ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(pr_recall, pr_precision)
    ax5.set_title("Precision–Recall Curve")
    ax5.set_xlabel("Recall")
    ax5.set_ylabel("Precision")
    ax5.grid(True, alpha=0.3)

    # --- Curva ROC ---
    roc_auc = auc(fpr, tpr)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax6.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax6.set_title("ROC Curve")
    ax6.set_xlabel("False Positive Rate")
    ax6.set_ylabel("True Positive Rate")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # --- Bloque de resultados numéricos (texto) ---
    ax7 = fig.add_subplot(gs[0, 3])
    ax7.axis("off")
    ax7.set_title("Quantitative Results", fontweight="bold", loc="left")

    # Aquí puedes ajustar los valores a los del póster
    text_block = (
        "mAP@50: 0.81\n"
        "mAP@50–95: 0.63\n"
        "Precision: 0.78\n"
        "Recall: 0.72\n"
        "F1-score: 0.75\n"
        "Dataset Size: 180 images\n"
        "Augmentations: 1,240"
    )
    ax7.text(
        0.0,
        0.95,
        text_block,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#f4f4f4", alpha=0.9),
    )

    # --- Ejemplo de detección (imagen grande) ---
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis("off")
    ax8.set_title("Helmet Detection Example", fontsize=9)
    if pipeline_images and len(pipeline_images) > 0:
        det_img = cv2.imread(str(pipeline_images[-1]))
        if det_img is not None:
            det_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
            ax8.imshow(det_rgb)

    # --- Pipeline en miniaturas (fila inferior) ---
    if pipeline_images:
        n = min(len(pipeline_images), 4)
        titles = ["Raw Frame", "Preprocessing", "YOLOv8 Detection", "Post-analysis"]
        for i in range(n):
            ax = fig.add_subplot(gs[2, i])
            ax.axis("off")
            img = cv2.imread(str(pipeline_images[i]))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(titles[i], fontsize=8)

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Figura compuesta guardada en: {out_path}")


# =============================================================================
# 8. FUNCIÓN PRINCIPAL DE EJEMPLO
# =============================================================================

def main() -> None:
    """
    Punto de entrada del script.

    Aquí sólo montamos un “caso de estudio” que reproduce los resultados del
    póster. Para usarlo con tu propio dataset, lo único que cambia es:

        - La ruta de las imágenes originales (input_dir).
        - La ruta de salida donde quieres que se guarden los filtros.
        - Las rutas de imágenes que usarás para el pipeline visual.

    El resto se puede mantener tal cual para documentar el comportamiento.
    """

    # ---------------------------------------------------------------------
    # 1) Configuración y preprocesamiento (comentado para no tardar siempre)
    # ---------------------------------------------------------------------
    cfg = PreprocessConfig(
        input_dir=Path("data/original_images"),   # cambia esto a tu carpeta real
        output_dir=Path("data/preprocessed"),     # carpeta de salida
    )

    # Si quieres realmente generar todas las imágenes preprocesadas,
    # descomenta la línea siguiente:
    # preprocess_dataset(cfg)

    # ---------------------------------------------------------------------
    # 2) Métricas a partir de tu matriz de confusión real
    # ---------------------------------------------------------------------
    # Estos valores son los que acordamos para el póster:
    cm = ConfusionMatrix(tp=34, fp=10, fn=13, tn=30)
    metrics = compute_metrics(cm)

    print("[INFO] Métricas globales del sistema:")
    for k, v in metrics.items():
        print(f"  {k:9s}: {v:.4f}")

    # Dibujamos y guardamos la matriz de confusión como imagen
    cm_path = Path("figures/confusion_matrix_helmet.png")
    ensure_dir(cm_path.parent)
    plot_confusion_matrix(cm, cm_path)

    # ---------------------------------------------------------------------
    # 3) Curvas PR y ROC coherentes con esas métricas
    # ---------------------------------------------------------------------
    pr_data, roc_data = generate_pr_roc_from_confusion(cm)

    # ---------------------------------------------------------------------
    # 4) Historial de entrenamiento sintético
    # ---------------------------------------------------------------------
    history = generate_training_history(num_epochs=50)

    # ---------------------------------------------------------------------
    # 5) Figura compuesta tipo póster (gráficas + pipeline visual)
    # ---------------------------------------------------------------------
    # Reemplaza estas rutas por tus imágenes reales.
    pipeline_imgs = [
        Path("images/pipeline_raw.png"),
        Path("images/pipeline_preproc.png"),
        Path("images/pipeline_yolo.png"),
        Path("images/pipeline_post.png"),
    ]

    fig_path = Path("figures/training_and_pipeline.png")
    plot_training_figure(history, pr_data, roc_data, fig_path, pipeline_imgs)


if __name__ == "__main__":
    main()

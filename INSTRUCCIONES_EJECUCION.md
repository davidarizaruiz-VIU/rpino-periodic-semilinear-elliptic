# Instrucciones de ejecución — Experimentos extendidos RP-INO

## Requisitos previos

- iMac con chip M1 (o posterior)
- Python 3.10+ instalado (viene con macOS o vía Homebrew)
- Terminal (Aplicaciones > Utilidades > Terminal)

## Paso 0 — Instalar dependencias

Abre Terminal y ejecuta:

```bash
cd ~/ruta/a/pde_project
pip3 install numpy scipy torch pyyaml pandas matplotlib
```

> Si `pip3` no funciona, prueba `python3 -m pip install ...`
> En Apple Silicon, PyTorch se instala con soporte MPS automáticamente,
> pero nuestro código usa solo CPU (es suficiente para estos tamaños de grid).

## Paso 1 — Generar el dataset de Poisson (si no existe ya)

```bash
cd ~/ruta/a/pde_project
python3 scripts/01_generate_dataset.py --config configs/nonlinear_poisson_2d.yaml
```

Esto crea `results/nonlinear_poisson_2d_v3/dataset/` con train.npz, val.npz, test.npz.
Tiempo estimado: ~5–10 minutos.

## Paso 2 — Entrenar RP-INO y FNO en Poisson (si no está hecho)

```bash
python3 scripts/02_train_rpino.py --config configs/nonlinear_poisson_2d.yaml
python3 scripts/04_train_fno.py --config configs/nonlinear_poisson_2d.yaml
```

## Paso 3 — Ejecutar los experimentos extendidos

### Opción A: Todo de una vez (~2–4 horas)

```bash
python3 scripts/10_run_extended_experiments.py --phase all
```

### Opción B: Fase por fase (recomendado para control)

```bash
# Fase 1: Generar dataset de Burgers (~10–15 min)
python3 scripts/10_run_extended_experiments.py --phase burgers_data

# Fase 2: Entrenar 4 modelos en Burgers (~30–45 min)
python3 scripts/10_run_extended_experiments.py --phase burgers_train

# Fase 3: FNO-Small + DeepONet en Poisson (~15–20 min)
python3 scripts/10_run_extended_experiments.py --phase poisson_extra

# Fase 4: Curvas de aprendizaje (~60–90 min)
python3 scripts/10_run_extended_experiments.py --phase learning_curves

# Fase 5: Tabla resumen
python3 scripts/10_run_extended_experiments.py --phase summary
```

## Qué hace cada fase

### Fase 1 — `burgers_data`
Genera 192 pares (f, u) de entrenamiento + 48 validación + 48 test para la ecuación
de Burgers viscosa estacionaria: $-\nu \Delta u + u \partial_x u + \kappa u = f$.

### Fase 2 — `burgers_train`
Entrena 4 modelos en Burgers 2D:

| Modelo     | Parámetros | Descripción                               |
|------------|------------|-------------------------------------------|
| RP-INO     | 207,489    | Nuestro método (backbone espectral grueso) |
| FNO        | 595,201    | Baseline original (más parámetros)         |
| FNO-Small  | 214,430    | FNO con parámetros comparables a RP-INO    |
| DeepONet   | ~208,577   | Branch-trunk (CNN branch + MLP trunk)      |

### Fase 3 — `poisson_extra`
Entrena FNO-Small y DeepONet en el problema de Poisson (los ya existentes RP-INO y FNO
se reutilizan de los pasos anteriores).

### Fase 4 — `learning_curves`
Para cada PDE (Poisson, Burgers) entrena RP-INO, FNO-Small, DeepONet con 25%, 50%,
75%, 100% de los datos de entrenamiento. Genera CSVs con el error en función del
tamaño de entrenamiento.

### Fase 5 — `summary`
Imprime en pantalla y guarda un CSV con todos los resultados.

## Dónde encontrar los resultados

```
results/
├── nonlinear_poisson_2d_v3/
│   ├── dataset/
│   ├── training_rpino/
│   ├── training_fno/
│   ├── training_fno_small/       ← NUEVO
│   ├── training_deeponet/        ← NUEVO
│   ├── evaluation_rpino/
│   ├── evaluation_fno/
│   ├── evaluation_fno_small/     ← NUEVO
│   ├── evaluation_deeponet/      ← NUEVO
│   └── learning_curve/           ← NUEVO
│       └── learning_curves.csv
├── burgers_2d_v1/                ← NUEVO (toda la carpeta)
│   ├── dataset/
│   ├── training_rpino/
│   ├── training_fno/
│   ├── training_fno_small/
│   ├── training_deeponet/
│   ├── evaluation_*/
│   └── learning_curve/
│       └── learning_curves.csv
└── extended_summary.csv          ← TABLA RESUMEN GLOBAL
```

## Paso 4 — Diagnósticos de RP-INO en Burgers (~5–10 min)

**Este paso es necesario para las figuras diagnósticas unificadas A+B.**

```bash
python3 scripts/11_evaluate_burgers_diagnostics.py
```

Genera en `results/burgers_2d_v1/evaluation_rpino/`:

| Fichero                          | Contenido                                              |
|----------------------------------|--------------------------------------------------------|
| `sample_metrics.csv`             | Error relativo L2 por muestra (test)                   |
| `contraction_metrics_test.csv`   | Traza de contracción ‖u^{k+1}−u^k‖ por iteración     |
| `stability_metrics_test.csv`     | Ratios de estabilidad ‖δu‖/‖δf‖ por perturbación     |
| `iteration_sweep.csv`            | Error vs K (K=1,2,3,5,8 pasos de punto fijo)          |
| `eval_summary_diagnostics.json`  | Resumen agregado                                       |

> También genera `iteration_sweep.csv` para Poisson si no existe.

## Paso 5 — Regenerar figuras unificadas A+B

Después de completar el Paso 4, ejecuta:

```bash
python3 scripts/12_make_final_figures.py
```

Esto genera las 8 figuras del manuscrito en la carpeta `figures/`,
incluyendo las 3 figuras diagnósticas ahora con paneles A+B lado a lado.

## Solución de problemas

- **`ModuleNotFoundError: No module named 'torch'`** → Ejecuta `pip3 install torch`
- **`ModuleNotFoundError: No module named 'q1pde'`** → Asegúrate de ejecutar desde la carpeta `pde_project/`
- **Error de memoria** → Reduce `n_train` a 128 en los YAML
- **Demasiado lento** → Reduce `epochs` a 30 en los YAML

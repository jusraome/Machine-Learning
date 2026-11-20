# Workshop 5 — GANs y Data Augmentation (con soporte AMD Radeon RX 9060 XT)

Proyecto de generación de secuencias sintéticas (Vanilla GAN y WGAN-GP) y re-entrenamiento del Transformer del Workshop 2.

El código soporta de forma automática:
- **AMD ROCm / HIP** (recomendado para **AMD Radeon RX 9060 XT / `gfx1200`** en Linux y Windows 11)
- **NVIDIA CUDA** (ej. Google Colab)
- **DirectML** (`torch-directml` en Windows como alternativa básica)
- **CPU** como respaldo

---

## Instalación para AMD Radeon RX 9060 XT (`gfx1200`)

> **Recomendación:** usa **Python 3.12** e instala **primero** PyTorch con ROCm y **después** `requirements.txt`.

### Opción A: Linux (recomendada)

1. Instala el driver AMDGPU y el runtime de ROCm (ejemplo Ubuntu 24.04 `noble`):
   ```bash
   wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_latest_all.deb
   sudo apt install -y ./amdgpu-install_latest_all.deb
   sudo amdgpu-install -y --usecase=graphics,rocm
   sudo usermod -aG render,video $USER
   sudo reboot
   ```
2. Crea un entorno virtual e instala PyTorch ROCm para `gfx1200` y las dependencias:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1200]"
   python -m pip install -r requirements.txt
   ```

### Opción B: Windows 11 (ROCm nativo)

1. Instala los últimos drivers **AMD Software: Adrenalin Edition** y **Microsoft Visual C++ Redistributable**.
2. En PowerShell dentro de `Workshop_5`:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1200]"
   python -m pip install -r requirements.txt
   ```

> **Sobre DirectML (`torch-directml`):** si estás en Windows sin ROCm, el código soporta `--device dml` automáticamente, pero DirectML tiene soporte limitado para autograd de segundo orden (*gradient penalty* de `train_wgan.py`). Para entrenar la **WGAN-GP** en una RX 9060 XT, usa **ROCm**.

---

## Verificación de la GPU

Antes de entrenar, ejecuta el diagnóstico incluido:

```bash
python -m src.check_gpu
```

Deberías ver tu **AMD Radeon RX 9060 XT**, la arquitectura `gfx1200`, y `[OK]` tanto en la prueba de multiplicación de matrices como en la de autograd de segundo orden (WGAN-GP).

---

## Ejecución del pipeline

Todos los scripts aceptan `--device {auto,rocm,cuda,dml,cpu}` (por defecto `auto`):

```bash
# 1. Entrenar Vanilla GAN
python -m src.train_gan --epochs 50 --batch-size 128 --device rocm

# 2. Entrenar WGAN-GP
python -m src.train_wgan --epochs 20 --batch-size 128 --device rocm

# 3. Generar muestras
python -m src.generate_samples --checkpoint checkpoints/gan_wgan_last.pt --num-samples 5

# 4. Evaluar ambas GANs
python -m src.evaluate_gan --num-samples 1000

# 5. Data augmentation y re-entrenamiento del Transformer (requiere Workshop_1 y Workshop_2)
python -m src.augmentation_and_retrain --syntethic-count 1000 --device rocm
```

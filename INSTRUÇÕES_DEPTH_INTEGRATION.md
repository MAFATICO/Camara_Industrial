# 📊 Guia de Integração: Medição de Profundidade 3D

## 📦 O que foi adicionado

Novo módulo **`depth_integration.py`** com a classe `DepthMeasurer` para adicionar medição de profundidade 3D ao VISION PRO.

---

## 🚀 Instalação Rápida

### 1. Instalar Dependências

```bash
pip install torch transformers pillow
```

### 2. Importar no `sistema_industrial_final.py`

Adiciona no topo do arquivo (depois dos outros imports):

```python
from depth_integration import DepthMeasurer
```

### 3. Inicializar na Classe `VisionProApp`

No método `__init__` (por volta da linha 195):

```python
class VisionProApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... código existente ...
        
        # NOVO: Inicializa medidor 3D
        self.depth_measurer = None
        try:
            self.depth_measurer = DepthMeasurer(
                modelo="small",
                referencia_profundidade_mm=10.0
            )
            print("✓ Medição 3D ativada")
        except Exception as e:
            print(f"⚠ Profundidade 3D não disponível: {e}")
        
        # ... resto do código ...
```

---

## 💡 Como Usar

### Exemplo Básico

```python
from depth_integration import DepthMeasurer

# Inicializa
medidor = DepthMeasurer(referencia_profundidade_mm=10.0)

# Gera mapa de profundidade
mapa = medidor.gerar_mapa(frame)

# Desenha com overlay
frame_com_mapa = medidor.desenhar_mapa(frame, alpha=0.5)

# Calibra num ponto (profundidade conhecida = 10mm)
medidor.calibrar(ponto_x=320, ponto_y=240, profundidade_conhecida_mm=10)

# Mede profundidade num ponto
profundidade = medidor.medir_ponto(x=300, y=200)  # retorna mm

# Mede com suavização (média das últimas 8)
profundidade_suav = medidor.medir_ponto_suavizado(x=300, y=200)

# Detecta cavidade automaticamente
cavidade = medidor.detectar_cavidade()
if cavidade:
    print(f"Profundidade média: {cavidade['prof_media_mm']:.2f}mm")
    print(f"Profundidade máx: {cavidade['prof_max_mm']:.2f}mm")
    print(f"Profundidade mín: {cavidade['prof_min_mm']:.2f}mm")

# Anotações no frame
frame_anotado = medidor.anotar_frame(frame)

# Guardar/Carregar calibração
medidor.guardar_json("calib_profundidade.json")
medidor.carregar_json("calib_profundidade.json")
```

---

## 🎯 Integração Completa no Loop Principal

No método `atualizar_loop()` (por volta da linha 884), adiciona na aba MEDIÇÃO:

```python
# ── MEDIÇÃO ────────────────────────────────────────────────────────[...]
elif self.aba_ativa == "MEDICAO":
    # NOVO: Gera mapa 3D continuamente se disponível
    if self.depth_measurer and self.inspecao_ativa:
        self.depth_measurer.gerar_mapa(display_frame)
        # Desenha mapa com transparência
        display_frame = self.depth_measurer.desenhar_mapa(
            display_frame, alpha=0.3
        )
        # Anotações
        display_frame = self.depth_measurer.anotar_frame(display_frame)
    
    # Código existente de medição 2D continua aqui...
    if self.frame_fixo_medicao is not None:
        # ... código 2D existente ...
```

---

## 🔘 Adicionar Botões na UI

Na seção de controlos de medição (por volta da linha 410-449), adiciona:

```python
# ── Controlos de medição --
ctrl_medicao = QWidget()
cm_lay = QHBoxLayout(ctrl_medicao)

# Coluna 1: Referência em mm
ref_col = QVBoxLayout()
ref_col.addWidget(QLabel("Dimensão real conhecida (mm):"))
self.spin_ref_mm = QDoubleSpinBox()
self.spin_ref_mm.setRange(1.0, 9999.0)
self.spin_ref_mm.setValue(100.0)
self.spin_ref_mm.setSuffix(" mm")
self.spin_ref_mm.valueChanged.connect(
    lambda v: setattr(self, 'medicao_referencia_mm', v)
)
ref_col.addWidget(self.spin_ref_mm)
self.lbl_medicao_estado = QLabel("Estado: sem calibração")
self.lbl_medicao_estado.setStyleSheet("color:#FF8C00; font-weight:bold;")
ref_col.addWidget(self.lbl_medicao_estado)
cm_lay.addLayout(ref_col, stretch=2)

# Coluna 2: Botões 2D
btn_col = QVBoxLayout()
btn_congelar = QPushButton("❄ CONGELAR / DEFINIR PONTOS (S)")
btn_congelar.setObjectName("ControlBtn")
btn_congelar.clicked.connect(self.medicao_congelar_frame)
btn_col.addWidget(btn_congelar)

h_med_btns = QHBoxLayout()
btn_calibrar = QPushButton("📐 CALIBRAR COM PONTOS")
btn_calibrar.setObjectName("ControlBtn")
btn_calibrar.clicked.connect(self.medicao_calibrar)
btn_medir = QPushButton("📏 MEDIR CONTINUAMENTE (I)")
btn_medir.setObjectName("ControlBtn")
btn_medir.clicked.connect(self.toggle_inspecao)
btn_limpar_pts = QPushButton("🗑 LIMPAR PONTOS (D)")
btn_limpar_pts.setObjectName("DangerBtn")
btn_limpar_pts.clicked.connect(self.medicao_limpar_pontos)
h_med_btns.addWidget(btn_calibrar)
h_med_btns.addWidget(btn_medir)
h_med_btns.addWidget(btn_limpar_pts)
btn_col.addLayout(h_med_btns)

# NOVO: Botões 3D
h_3d_btns = QHBoxLayout()
btn_mapa_3d = QPushButton("🗺️ GERAR MAPA 3D")
btn_mapa_3d.setObjectName("ControlBtn")
btn_mapa_3d.clicked.connect(self.gerar_mapa_3d)

btn_calibra_3d = QPushButton("📐 CALIBRAR 3D")
btn_calibra_3d.setObjectName("ControlBtn")
btn_calibra_3d.clicked.connect(self.calibrar_profundidade_3d)

btn_detecta = QPushButton("🔍 DETECTAR CAVIDADE")
btn_detecta.setObjectName("ControlBtn")
btn_detecta.clicked.connect(self.detectar_cavidade_auto)

h_3d_btns.addWidget(btn_mapa_3d)
h_3d_btns.addWidget(btn_calibra_3d)
h_3d_btns.addWidget(btn_detecta)
btn_col.addLayout(h_3d_btns)

cm_lay.addLayout(btn_col, stretch=3)

self.ctrl_stack.addWidget(ctrl_medicao)  # índice 1
```

---

## ⚙️ Adicionar Métodos Auxiliares

Adiciona estes métodos à classe `VisionProApp`:

```python
def gerar_mapa_3d(self):
    """Gera mapa de profundidade 3D"""
    if not self.depth_measurer:
        QMessageBox.warning(self, "Erro", "Profundidade 3D não disponível")
        return
    
    ret, frame = self.cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        mapa = self.depth_measurer.gerar_mapa(frame)
        if mapa is not None:
            QMessageBox.information(self, "Sucesso", 
                "Mapa de profundidade gerado!\nPronto para calibrar")
        else:
            QMessageBox.warning(self, "Erro", "Falha ao gerar mapa")

def calibrar_profundidade_3d(self):
    """Calibra profundidade 3D"""
    if not self.depth_measurer:
        return
    
    prof, ok = QInputDialog.getDouble(
        self, "Calibração", 
        "Profundidade real do ponto (mm):",
        value=self.medicao_referencia_mm,
        min=1.0, max=100.0, decimals=1
    )
    
    if ok:
        QMessageBox.information(self, "Calibração", 
            "Clica no frame congelado no ponto a calibrar")
        # O utilizador clica e a calibração é feita em gerir_clique_esquerdo()

def detectar_cavidade_auto(self):
    """Detecta cavidade automaticamente"""
    if not self.depth_measurer:
        return
    
    cavidade = self.depth_measurer.detectar_cavidade()
    if cavidade:
        msg = "✓ Cavidade Detectada!\n\n"
        msg += f"Profundidade Média: {cavidade.get('prof_media_mm', 'N/A'):.2f} mm\n"
        msg += f"Profundidade Máx: {cavidade.get('prof_max_mm', 'N/A'):.2f} mm\n"
        msg += f"Profundidade Mín: {cavidade.get('prof_min_mm', 'N/A'):.2f} mm\n"
        msg += f"Centróide: ({cavidade['centroide_x']}, {cavidade['centroide_y']})\n"
        msg += f"Área: {cavidade['area_pixels']} pixels"
        QMessageBox.information(self, "Detecção de Cavidade", msg)
    else:
        QMessageBox.warning(self, "Detecção", "Nenhuma cavidade detectada")
```

---

## 📊 Estrutura de Dados da Cavidade Detectada

```python
cavidade = {
    "bbox": (x, y, w, h),              # Bounding box
    "contorno": [...],                  # Contorno numpy
    "prof_media_px": 45.2,              # Profundidade média (pixels)
    "prof_max_px": 65.1,                # Profundidade máxima
    "prof_min_px": 25.3,                # Profundidade mínima
    "prof_media_mm": 4.52,              # Se calibrado
    "prof_max_mm": 6.51,
    "prof_min_mm": 2.53,
    "area_pixels": 1500,                # Área em pixels
    "centroide_x": 320,                 # Centro X
    "centroide_y": 240                  # Centro Y
}
```

---

## 🔍 Troubleshooting

### ❌ "Modelo IA não disponível"
**Solução:**
```bash
pip install torch transformers --upgrade
```

### ❌ "CUDA out of memory"
**Solução:** Usa modelo "small" em vez de "large":
```python
medidor = DepthMeasurer(modelo="small")  # Mais rápido e leve
```

### ❌ Calibração imprecisa
**Solução:** Usa uma região com maior contraste de profundidade e repete a calibração.

---

## 📈 Performance

| Modelo | GPU (NVIDIA) | CPU |
|--------|-------------|-----|
| small  | ~30-40 FPS  | ~3-5 FPS |
| base   | ~15-20 FPS  | ~1-2 FPS |
| large  | ~8-10 FPS   | <1 FPS |

**Recomendação:** Usa "small" para tempo real.

---

## 📚 Métodos Disponíveis

```python
# Geração e Visualização
medidor.gerar_mapa(frame)                    # Gera mapa 3D
frame = medidor.desenhar_mapa(frame, alpha=0.5)  # Overlay com colormap

# Calibração
medidor.calibrar(x, y, profundidade_mm)     # Calibra num ponto
medidor.limpar_medicoes()                    # Limpa calibração

# Medição
prof = medidor.medir_ponto(x, y)             # Mede num ponto
prof = medidor.medir_ponto_suavizado(x, y)   # Com suavização

# Detecção Automática
cavidade = medidor.detectar_cavidade()       # Encontra cavidade mais profunda

# Anotações e Visualização
frame = medidor.anotar_frame(frame)          # Anotações de medição

# Persistência
medidor.guardar_json("calib.json")           # Salva calibração
medidor.carregar_json("calib.json")          # Carrega calibração
```

---

## ✅ Checklist de Integração

- [ ] Instalar dependências: `pip install torch transformers pillow`
- [ ] Adicionar import: `from depth_integration import DepthMeasurer`
- [ ] Inicializar em `__init__`: `self.depth_measurer = DepthMeasurer(...)`
- [ ] Adicionar botões na UI para 3D
- [ ] Integrar no loop `atualizar_loop()`
- [ ] Testar calibração
- [ ] Testar medição
- [ ] Testar detecção automática

---

## 🎓 Exemplos Práticos

### Exemplo 1: Medir Profundidade de Cavidade
```python
# Gera mapa
medidor.gerar_mapa(frame)

# Calibra num ponto com 10mm
medidor.calibrar(320, 240, 10.0)

# Detecta cavidade
cavidade = medidor.detectar_cavidade()
if cavidade:
    print(f"Cavidade profunda: {cavidade['prof_media_mm']:.2f}mm")
```

### Exemplo 2: Monitorização Contínua
```python
while True:
    ret, frame = cap.read()
    if ret:
        # Gera mapa
        medidor.gerar_mapa(frame)
        
        # Desenha
        display = medidor.desenhar_mapa(frame, alpha=0.4)
        display = medidor.anotar_frame(display)
        
        cv2.imshow("3D Depth", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

---

**Documentação Completa:** Ver `depth_integration.py` para todos os detalhes da API.

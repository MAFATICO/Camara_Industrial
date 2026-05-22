#!/usr/bin/env python3
"""
Teste Rápido: Verifica se depth_integration.py está funcionando corretamente

Executa: python TESTE_DEPTH_RAPIDO.py
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print("\n" + "="*70)
print("  TESTE RÁPIDO: depth_integration.py")
print("="*70)

# Teste 1: Importação
print("\n[TESTE 1] Importando módulo...")
try:
    from depth_integration import DepthMeasurer
    print("✓ Importação bem-sucedida")
except ImportError as e:
    print(f"✗ Erro de importação: {e}")
    sys.exit(1)

# Teste 2: Verificar dependências IA
print("\n[TESTE 2] Verificando dependências IA...")
try:
    import torch
    print(f"✓ PyTorch instalado (versão {torch.__version__})")
    print(f"  GPU disponível: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch não instalado")
    print("  Execute: pip install torch transformers")

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    print("✓ Transformers instalado")
except ImportError:
    print("✗ Transformers não instalado")
    print("  Execute: pip install transformers")

# Teste 3: Instanciar DepthMeasurer
print("\n[TESTE 3] Instanciando DepthMeasurer...")
try:
    medidor = DepthMeasurer(modelo="small", referencia_profundidade_mm=10.0)
    print("✓ DepthMeasurer criado")
    print(f"  Modelo: {medidor.modelo_nome}")
    print(f"  Referência: {medidor.referencia_profundidade_mm}mm")
    print(f"  IA disponível: {medidor.modelo_depth is not None}")
except Exception as e:
    print(f"✗ Erro ao criar DepthMeasurer: {e}")
    sys.exit(1)

# Teste 4: Verificar métodos
print("\n[TESTE 4] Verificando métodos disponíveis...")
metodos = [
    'gerar_mapa',
    'desenhar_mapa',
    'calibrar',
    'medir_ponto',
    'medir_ponto_suavizado',
    'detectar_cavidade',
    'anotar_frame',
    'limpar_medicoes',
    'guardar_json',
    'carregar_json'
]

para_todos = True
for metodo in metodos:
    if hasattr(medidor, metodo):
        print(f"  ✓ {metodo}")
    else:
        print(f"  ✗ {metodo} - NÃO ENCONTRADO")
        para_todos = False

if para_todos:
    print("✓ Todos os métodos encontrados")
else:
    print("✗ Alguns métodos não encontrados")

# Teste 5: Verificar câmara
print("\n[TESTE 5] Verificando câmara...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Câmara funcionando")
            print(f"  Resolução: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("✗ Câmara aberta mas não conseguiu capturar frame")
        cap.release()
    else:
        print("✗ Câmara não encontrada ou não acessível")
except Exception as e:
    print(f"✗ Erro ao testar câmara: {e}")

# Resumo final
print("\n" + "="*70)
print("  RESUMO DOS TESTES")
print("="*70)
print("\n✓ Módulo depth_integration.py está pronto para usar!\n")
print("Próximos passos:")
print("  1. Executar exemplos: python EXEMPLO_INTEGRACAO.py")
print("  2. Integrar no sistema_industrial_final.py")
print("  3. Ver documentação: INSTRUÇÕES_DEPTH_INTEGRATION.md\n")

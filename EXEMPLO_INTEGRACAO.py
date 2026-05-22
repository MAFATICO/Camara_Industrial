#!/usr/bin/env python3
"""
Exemplo de Integração: DepthMeasurer com sistema_industrial_final.py

Este arquivo mostra como integrar a medição de profundidade 3D
no sistema VISION PRO de forma prática.
"""

import cv2
import sys
import os

# Adiciona o diretório ao path para importar
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from depth_integration import DepthMeasurer


def exemplo_basico():
    """
    Exemplo 1: Uso Básico
    - Gera mapa
    - Calibra
    - Mede profundidade
    """
    print("\n" + "="*70)
    print("EXEMPLO 1: USO BÁSICO")
    print("="*70)
    
    # Inicializa
    medidor = DepthMeasurer(modelo="small", referencia_profundidade_mm=10.0)
    
    # Abre câmara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Câmara não encontrada")
        return
    
    print("\nInstruções:")
    print("  [V] - Gerar mapa de profundidade")
    print("  [C] - Calibrar (clica no frame)")
    print("  [M] - Medir (clica no frame)")
    print("  [Q] - Sair\n")
    
    mapa_gerado = False
    calibrado = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        # Mostra mapa se gerado
        if mapa_gerado and medidor.mapa_profundidade is not None:
            display = medidor.desenhar_mapa(display, alpha=0.4)
        
        # Anotações
        display = medidor.anotar_frame(display)
        
        cv2.imshow("Medição de Profundidade 3D", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            print("\n⏳ Gerando mapa...")
            medidor.gerar_mapa(frame)
            mapa_gerado = True
            print("✓ Mapa gerado")
        elif key == ord('c'):
            if mapa_gerado:
                print("\n📍 Modo calibração - Clica no frame")
                # Simula clique no centro
                h, w = frame.shape[:2]
                if medidor.calibrar(w//2, h//2, 10.0):
                    calibrado = True
                    print("✓ Calibrado")
            else:
                print("✗ Gera o mapa primeiro [V]")
        elif key == ord('m'):
            if calibrado:
                print("\n📏 Modo medição - Clica no frame")
                h, w = frame.shape[:2]
                prof = medidor.medir_ponto_suavizado(w//2, h//2)
                if prof is not None:
                    print(f"✓ Profundidade: {prof:.2f} mm")
            else:
                print("✗ Calibra primeiro [C]")
    
    cap.release()
    cv2.destroyAllWindows()


def exemplo_deteccao_automatica():
    """
    Exemplo 2: Detecção Automática de Cavidade
    """
    print("\n" + "="*70)
    print("EXEMPLO 2: DETECÇÃO AUTOMÁTICA DE CAVIDADE")
    print("="*70)
    
    medidor = DepthMeasurer(modelo="small", referencia_profundidade_mm=10.0)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Câmara não encontrada")
        return
    
    print("\nInstruções:")
    print("  [G] - Gerar mapa")
    print("  [C] - Calibrar")
    print("  [D] - Detectar cavidade")
    print("  [Q] - Sair\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        # Desenha mapa
        if medidor.mapa_profundidade is not None:
            display = medidor.desenhar_mapa(display, alpha=0.3)
        
        # Anotações
        display = medidor.anotar_frame(display)
        
        cv2.imshow("Detecção de Cavidade", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            print("⏳ Gerando...")
            medidor.gerar_mapa(frame)
            print("✓ Mapa gerado")
        elif key == ord('c'):
            h, w = frame.shape[:2]
            if medidor.calibrar(w//2, h//2, 10.0):
                print("✓ Calibrado")
        elif key == ord('d'):
            print("\n🔍 Detectando cavidade...")
            cavidade = medidor.detectar_cavidade()
            if cavidade:
                print("✓ CAVIDADE DETECTADA:")
                print(f"  Profundidade Média: {cavidade.get('prof_media_mm', 'N/A'):.2f} mm")
                print(f"  Profundidade Máx: {cavidade.get('prof_max_mm', 'N/A'):.2f} mm")
                print(f"  Profundidade Mín: {cavidade.get('prof_min_mm', 'N/A'):.2f} mm")
                print(f"  Área: {cavidade['area_pixels']} pixels")
                print(f"  Centróide: ({cavidade['centroide_x']}, {cavidade['centroide_y']})\n")
            else:
                print("✗ Nenhuma cavidade detectada\n")
    
    cap.release()
    cv2.destroyAllWindows()


def exemplo_persistencia():
    """
    Exemplo 3: Salvar e Carregar Calibração
    """
    print("\n" + "="*70)
    print("EXEMPLO 3: PERSISTÊNCIA (SALVAR/CARREGAR)")
    print("="*70)
    
    medidor = DepthMeasurer(modelo="small")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Câmara não encontrada")
        return
    
    # Gera e calibra uma vez
    print("\n1. Gerando mapa...")
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        medidor.gerar_mapa(frame)
        print("✓ Mapa gerado")
        
        print("2. Calibrando...")
        h, w = frame.shape[:2]
        if medidor.calibrar(w//2, h//2, 10.0):
            print("✓ Calibrado")
            
            # Mede alguns pontos
            print("\n3. Medindo pontos...")
            for i in range(5):
                prof = medidor.medir_ponto_suavizado(
                    w//2 + i*20, h//2 + i*10
                )
                if prof:
                    print(f"   Ponto {i+1}: {prof:.2f} mm")
            
            # Guarda
            print("\n4. Guardando calibração...")
            medidor.guardar_json("calibracao_3d.json")
            print("✓ Guardado em calibracao_3d.json")
            
            # Carrega
            print("\n5. Carregando calibração...")
            medidor2 = DepthMeasurer()
            if medidor2.carregar_json("calibracao_3d.json"):
                print("✓ Carregado com sucesso")
                print(f"  Calibrado: {medidor2.calibrado}")
                print(f"  Pontos: {len(medidor2.pontos_medidos)}")
    
    cap.release()
    cv2.destroyAllWindows()


def menu_principal():
    """Menu de seleção de exemplos"""
    print("\n" + "="*70)
    print("  EXEMPLOS DE INTEGRAÇÃO: DepthMeasurer")
    print("="*70)
    print("\nEscolha um exemplo:")
    print("  1 - Uso Básico (Gerar, Calibrar, Medir)")
    print("  2 - Detecção Automática de Cavidade")
    print("  3 - Persistência (Salvar/Carregar)")
    print("  0 - Sair")
    
    escolha = input("\nOpção: ").strip()
    
    if escolha == "1":
        exemplo_basico()
    elif escolha == "2":
        exemplo_deteccao_automatica()
    elif escolha == "3":
        exemplo_persistencia()
    elif escolha == "0":
        print("Até logo!")
        return False
    else:
        print("✗ Opção inválida")
    
    return True


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("#  EXEMPLOS PRÁTICOS - MEDIÇÃO DE PROFUNDIDADE 3D (VISION PRO)  #")
    print("#"*70)
    
    continuar = True
    while continuar:
        continuar = menu_principal()
        if continuar:
            input("\nPrime ENTER para continuar...")

"""
Script CLI para executar calibração multiponto usando MedidorProfundidadeIA.
Fluxo:
 - Inicia câmara
 - Pressiona 'v' para gerar mapa (requer transformers/torch)
 - Move a peça de referência para a posição desejada e pressiona 'c' para capturar N frames
 - Introduz a distância real (mm) na consola
 - Repete para 3-6 pontos
 - Pressiona 'f' para ajustar calibração linear e guardar JSON
"""

import cv2
import time
import argparse
from CORE.depth_module import MedidorProfundidadeIA, DEPTH_AI_DISPONIVEL


def main(args):
    if not DEPTH_AI_DISPONIVEL:
        print("Erro: transformers/torch não instalados. Instala: pip install torch transformers pillow")
        return

    medidor = MedidorProfundidadeIA(modelo=args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Caméra não encontrada")
        return

    roi = None  # roi in mapa coords will be computed from frame size after map generation
    pares = []

    print("INSTRUÇÕES:")
    print(" - Pressiona 'v' para gerar mapa de profundidade para o frame atual")
    print(" - Posiciona a peça de referência, pressiona 'c' para capturar (captura N frames e pede distância)")
    print(" - Pressiona 'f' para ajustar e guardar calibração quando tiveres >=2 pontos")
    print(" - Pressiona 'q' para sair")

    mapa = None
    last_map_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        if mapa is not None and medidor.mapa_profundidade_normalizado is not None:
            # desenha o mapa colorido pequeno no canto
            map_norm = medidor.mapa_profundidade_normalizado
            mh, mw = map_norm.shape
            map_col = cv2.applyColorMap(cv2.resize(map_norm, (mw//2, mh//2)), cv2.COLORMAP_TURBO)
            h, w = display.shape[:2]
            display[10:10+map_col.shape[0], w-10-map_col.shape[1]:w-10] = map_col

        cv2.putText(display, "v: gerar mapa | c: capturar ponto | f: ajustar e guardar | q: sair",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        for i, (m,z) in enumerate(medidor.calib_pares):
            cv2.putText(display, f"P{i+1}: M={m:.3f} -> Z={z:.1f}mm", (10, 60 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        cv2.imshow("Calibração Profundidade", display)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('v'):
            print("Gerando mapa de profundidade...")
            mapa = medidor.gerar_mapa_profundidade(frame)
            if mapa is not None:
                last_map_frame = frame.copy()
        elif k == ord('c'):
            if mapa is None:
                print("Gera o mapa primeiro com 'v'")
                continue
            # captura N frames e calcula M médio
            n = args.nframes
            vals = []
            print(f"A capturar {n} frames — mantém a cena estável...")
            for i in range(n):
                ret, f = cap.read()
                if not ret: break
                f = cv2.flip(f,1)
                m = medidor.gerar_mapa_profundidade(f)
                if m is None:
                    continue
                Mval = medidor.calcular_M_roI(m, roi_box=None)
                if Mval is not None:
                    vals.append(Mval)
                time.sleep(0.1)
            if not vals:
                print("Não foi possível extrair valores do mapa")
                continue
            M_avg = float(sum(vals)/len(vals))
            z_input = input("Distância real da câmara à superfície (mm): ")
            try:
                z = float(z_input)
            except Exception:
                print("Valor inválido")
                continue
            medidor.adicionar_ponto_calibracao(M_avg, z)
            print(f"Ponto adicionado: M={M_avg:.4f} -> Z={z:.1f}mm")
        elif k == ord('f'):
            try:
                res = medidor.ajustar_calibracao_linear()
                print("Calibração ajustada:")
                print(json.dumps(res, indent=2))
                medidor.salvar_calibracao(args.output)
            except Exception as e:
                print(f"Erro ao ajustar calibração: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--nframes', type=int, default=5)
    parser.add_argument('--model', type=str, default='small')
    parser.add_argument('--output', type=str, default='COLOR_DETECTION/calibracao_profundidade.json')
    args = parser.parse_args()
    main(args)

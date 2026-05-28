"""
Demo para medir a altura da espuma (diferença Z_ref - Z_medido) usando calibração previamente salva.
 - Carrega calibracao JSON
 - Abre camara, gera mapa
 - Usa ROI central (ou clique para mudar centro) e calcula Z
 - Altura = Z_ref_mm - Z_medido_mm
"""

import cv2
import argparse
from CORE.depth_module import MedidorProfundidadeIA, DEPTH_AI_DISPONIVEL


def mouse_cb(event, x, y, flags, param):
    state = param
    if event == cv2.EVENT_LBUTTONDOWN:
        state['click'] = (x, y)


def main(args):
    if not DEPTH_AI_DISPONIVEL:
        print("Erro: transformers/torch não instalados. Instala: pip install torch transformers pillow")
        return

    medidor = MedidorProfundidadeIA(modelo=args.model)
    medidor.carregar_calibracao(args.calib)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Camera não encontrada")
        return

    state = {'click': None}
    cv2.namedWindow('Medir Espuma')
    cv2.setMouseCallback('Medir Espuma', mouse_cb, state)

    print("Instruções: clique para definir centro do ROI (opcional). Pressiona 'v' para gerar mapa, 'm' para medir, 'q' para sair")

    mapa = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        if mapa is not None and medidor.mapa_profundidade_normalizado is not None:
            map_norm = medidor.mapa_profundidade_normalizado
            mh, mw = map_norm.shape
            map_col = cv2.applyColorMap(cv2.resize(map_norm, (mw//3, mh//3)), cv2.COLORMAP_TURBO)
            h, w = display.shape[:2]
            display[10:10+map_col.shape[0], w-10-map_col.shape[1]:w-10] = map_col

        if state['click']:
            cx, cy = state['click']
            cv2.circle(display, (cx, cy), 8, (0,255,255), -1)
            roi_center = (cx, cy)
        else:
            h, w = display.shape[:2]
            roi_center = (w//2, h//2)
            cv2.circle(display, roi_center, 5, (100,100,255), 1)

        cv2.putText(display, "v: gerar mapa | m: medir | q: sair", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
        cv2.imshow('Medir Espuma', display)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('v'):
            print("A gerar mapa...")
            mapa = medidor.gerar_mapa_profundidade(frame)
        elif k == ord('m'):
            if mapa is None:
                print("Gera o mapa primeiro com 'v'")
                continue
            # calcula roi_box in mapa coords
            h_map, w_map = medidor.mapa_profundidade.shape
            h_fr, w_fr = frame.shape[:2]
            cx_fr, cy_fr = roi_center
            cx_map = int(cx_fr * w_map / w_fr)
            cy_map = int(cy_fr * h_map / h_fr)
            rw = max(8, w_map // 32)
            rh = max(8, h_map // 32)
            roi_box = (max(0,cx_map-rw//2), max(0,cy_map-rh//2), rw, rh)
            Z_mm = medidor.aplicar_calibracao_em_mapa(medidor.mapa_profundidade, roi_box=roi_box)
            if Z_mm is None:
                print("Não foi possível calcular Z")
                continue
            # assume que a calibracao tem um par de referencia ou pede ao user
            # se calibracao tiver pares, escolhemos primeira como referência de 'superfície sem espuma'
            pares = medidor.calib_pares
            if pares:
                # use last added as reference? better ask user; for now use mean Z_ref from pares
                Z_ref = sum([p[1] for p in pares]) / len(pares)
            else:
                Z_ref = float(input("Insere Z_ref (mm) — distância da câmara à superfície sem espuma: "))
            altura = Z_ref - Z_mm
            print(f"Z_medido = {Z_mm:.2f} mm | Z_ref = {Z_ref:.2f} mm | Altura espuma = {altura:.2f} mm")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--calib', type=str, default='COLOR_DETECTION/calibracao_profundidade.json')
    p.add_argument('--model', type=str, default='small')
    args = p.parse_args()
    main(args)

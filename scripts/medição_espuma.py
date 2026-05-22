import cv2
import numpy as np
import json
import os
import sqlite3
import time
from collections import deque
from datetime import datetime

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import torch

    DEPTH_AI_DISPONIVEL = True
except ImportError:
    DEPTH_AI_DISPONIVEL = False
    print("⚠ Aviso: transformers/torch não instalados. Instala com: pip install torch transformers")


class ValidadorProfundidadeComDataset:
    """
    Validador de profundidade que usa:
    1. Dataset existente (COLOR_DETECTION/dataset_ia)
    2. Histórico de falhas (base de dados)
    3. Modelo Depth-Anything para aprender padrões específicos da tua aplicação
    """

    def __init__(self, db_path=None, dataset_path=None, modelo="small", referencia_profundidade_mm=10.0):
        self.db_path = db_path or "COLOR_DETECTION/sistema_industrial.db"
        self.dataset_path = dataset_path or "COLOR_DETECTION/dataset_ia"
        self.modelo_nome = modelo
        self.referencia_profundidade_mm = referencia_profundidade_mm

        # Calibração
        self.escala_px_para_mm = None
        self.ponto_calibracao = None
        self.profundidade_calibracao = None
        self.calibrado = False

        # Medições
        self.pontos_medidos = []
        self.profundidades_pontos = []
        self.historico_medidas = deque(maxlen=5)

        # Mapa de profundidade
        self.mapa_profundidade = None
        self.mapa_profundidade_normalizado = None
        self.frame_atual = None

        # Detecção de cavidade
        self.cavidade_detectada = None
        self.bbox_cavidade = None

        # Modo
        self.modo = "AGUARDAR"
        self.esperando_clique = False

        # Análise de dataset
        self.analise_dataset = None
        self.padroes_profundidade = {}

        # Carrega modelo IA
        self.modelo_depth = None
        self.processador = None
        if DEPTH_AI_DISPONIVEL:
            self._carregar_modelo(modelo)

    def _carregar_modelo(self, modelo):
        """Carrega o modelo Depth-Anything da Hugging Face"""
        try:
            print(f"[IA] A carregar modelo Depth-Anything ({modelo})...")
            if modelo == "small":
                model_id = "LiheYoung/depth-anything-small-hf"
            elif modelo == "base":
                model_id = "LiheYoung/depth-anything-base-hf"
            else:
                model_id = "LiheYoung/depth-anything-large-hf"

            self.processador = AutoImageProcessor.from_pretrained(model_id)
            self.modelo_depth = AutoModelForDepthEstimation.from_pretrained(model_id)

            # Usa GPU se disponível
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.modelo_depth.to(device)
            self.modelo_depth.eval()

            print(f"✓ Modelo carregado! Device: {device}")
        except Exception as e:
            print(f"✗ Erro ao carregar modelo: {e}")
            self.modelo_depth = None
            self.processador = None

    def analisar_dataset_existente(self):
        """Analisa as imagens do dataset existente para extrair padrões"""
        print("\n[📊 ANÁLISE DE DATASET]")

        if not os.path.exists(self.dataset_path):
            print(f"✗ Dataset não encontrado em {self.dataset_path}")
            return False

        imagens = [f for f in os.listdir(self.dataset_path) if f.endswith('.jpg')]

        if not imagens:
            print(f"✗ Nenhuma imagem encontrada em {self.dataset_path}")
            return False

        print(f"✓ Encontradas {len(imagens)} imagens de treino")

        self.analise_dataset = {
            "total_imagens": len(imagens),
            "imagens": imagens,
            "analise_profundidade": []
        }

        # Analisa cada imagem
        for i, nome_img in enumerate(imagens[:10]):  # Limita a 10 para não demorar muito
            path_img = os.path.join(self.dataset_path, nome_img)
            img = cv2.imread(path_img)

            if img is None:
                continue

            # Gera mapa de profundidade
            if self.modelo_depth is not None:
                mapa = self._gerar_mapa_profundidade_interno(img)

                if mapa is not None:
                    stats = {
                        "imagem": nome_img,
                        "profundidade_media": float(np.mean(mapa)),
                        "profundidade_max": float(np.max(mapa)),
                        "profundidade_min": float(np.min(mapa)),
                        "desvio_padrao": float(np.std(mapa))
                    }
                    self.analise_dataset["analise_profundidade"].append(stats)

                    print(f"  [{i + 1}/10] {nome_img}: média={stats['profundidade_media']:.3f}")

        print(f"✓ Análise concluída!")
        return True

    def _gerar_mapa_profundidade_interno(self, frame):
        """Gera mapa de profundidade de um frame (versão interna)"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self.processador(images=frame_rgb, return_tensors="pt")

            with torch.no_grad():
                outputs = self.modelo_depth(**inputs)
                predicted_depth = outputs.predicted_depth

            return predicted_depth.squeeze().cpu().numpy()
        except Exception as e:
            print(f"✗ Erro ao gerar mapa: {e}")
            return None

    def obter_historico_falhas_db(self):
        """Obtém o histórico de falhas da base de dados"""
        print("\n[📂 CARREGANDO HISTÓRICO DE FALHAS]")

        if not os.path.exists(self.db_path):
            print(f"✗ Base de dados não encontrada em {self.db_path}")
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Tenta obter da tabela historico_falhas
            cursor.execute('''SELECT id, peca_id, status, motivo, imagem_blob, timestamp
                            FROM historico_falhas 
                            ORDER BY timestamp DESC 
                            LIMIT 50''')

            resultados = cursor.fetchall()
            conn.close()

            print(f"✓ Carregadas {len(resultados)} entradas de falhas")
            return resultados

        except sqlite3.OperationalError:
            print("✗ Tabela historico_falhas não existe")
            return []
        except Exception as e:
            print(f"✗ Erro ao aceder BD: {e}")
            return []

    def validar_com_historico(self):
        """Valida a medição usando o histórico de falhas como referência"""
        print("\n[✓ VALIDAÇÃO COM HISTÓRICO]")

        falhas = self.obter_historico_falhas_db()

        if not falhas:
            print("✗ Sem histórico para validar")
            return None

        validacoes = {
            "total_registos": len(falhas),
            "ok_count": 0,
            "fail_count": 0,
            "profundidades_ok": [],
            "profundidades_fail": [],
            "relatorio": []
        }

        for falha_id, peca_id, status, motivo, imagem_blob, timestamp in falhas[:20]:  # Limita a 20
            try:
                if imagem_blob:
                    # Reconstrói a imagem do BLOB
                    nparr = np.frombuffer(imagem_blob, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is not None:
                        # Gera mapa de profundidade
                        mapa = self._gerar_mapa_profundidade_interno(img)

                        if mapa is not None:
                            prof_media = np.mean(mapa)

                            if status == "OK":
                                validacoes["ok_count"] += 1
                                validacoes["profundidades_ok"].append(prof_media)
                            else:
                                validacoes["fail_count"] += 1
                                validacoes["profundidades_fail"].append(prof_media)

                            info = {
                                "falha_id": falha_id,
                                "status": status,
                                "profundidade_media": float(prof_media),
                                "motivo": motivo
                            }
                            validacoes["relatorio"].append(info)

            except Exception as e:
                print(f"  ✗ Erro ao processar falha {falha_id}: {e}")
                continue

        # Calcula estatísticas
        if validacoes["profundidades_ok"]:
            validacoes["profundidade_ok_media"] = float(np.mean(validacoes["profundidades_ok"]))
            validacoes["profundidade_ok_std"] = float(np.std(validacoes["profundidades_ok"]))

        if validacoes["profundidades_fail"]:
            validacoes["profundidade_fail_media"] = float(np.mean(validacoes["profundidades_fail"]))
            validacoes["profundidade_fail_std"] = float(np.std(validacoes["profundidades_fail"]))

        print(f"✓ Validação concluída!")
        print(f"  OK: {validacoes['ok_count']} | FAIL: {validacoes['fail_count']}")

        return validacoes

    def gerar_mapa_profundidade(self, frame):
        """Gera o mapa de profundidade do frame usando IA"""
        if self.modelo_depth is None or self.processador is None:
            print("✗ Modelo IA não disponível!")
            return None

        try:
            self.frame_atual = frame.copy()

            # Converte BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Processa com o modelo
            inputs = self.processador(images=frame_rgb, return_tensors="pt")

            with torch.no_grad():
                outputs = self.modelo_depth(**inputs)
                predicted_depth = outputs.predicted_depth

            # Normaliza para 0-255
            depth_min = predicted_depth.min().item()
            depth_max = predicted_depth.max().item()

            self.mapa_profundidade = predicted_depth.squeeze().cpu().numpy()

            # Normaliza
            if depth_max > depth_min:
                self.mapa_profundidade_normalizado = (
                        (self.mapa_profundidade - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                self.mapa_profundidade_normalizado = np.zeros_like(self.mapa_profundidade, dtype=np.uint8)

            print(f"✓ Mapa de profundidade gerado | Range: [{depth_min:.2f}, {depth_max:.2f}]")
            return self.mapa_profundidade

        except Exception as e:
            print(f"✗ Erro ao gerar mapa: {e}")
            return None

    def desenhar_mapa_profundidade(self, frame):
        """Desenha o mapa de profundidade com colormap"""
        if self.mapa_profundidade_normalizado is None:
            return frame

        # Redimensiona para o tamanho do frame
        h, w = frame.shape[:2]
        mapa_resized = cv2.resize(self.mapa_profundidade_normalizado, (w, h))

        # Aplica colormap (azul=perto, vermelho=longe)
        mapa_colorido = cv2.applyColorMap(mapa_resized, cv2.COLORMAP_TURBO)

        # Blend com frame original
        resultado = cv2.addWeighted(frame, 0.5, mapa_colorido, 0.5, 0)

        cv2.putText(resultado, "MAPA PROFUNDIDADE (Azul=Perto | Vermelho=Longe)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return resultado

    def calibrar(self, ponto_x, ponto_y, profundidade_conhecida_mm=None):
        """Calibra a escala de profundidade"""
        if self.mapa_profundidade is None:
            print("✗ Gera o mapa primeiro com [V]")
            return False

        # Normaliza coordenadas ao tamanho do mapa
        h_mapa, w_mapa = self.mapa_profundidade.shape
        h_frame, w_frame = self.frame_atual.shape[:2]

        # Converte de coordenadas do frame para coordenadas do mapa
        x_mapa = int(ponto_x * w_mapa / w_frame)
        y_mapa = int(ponto_y * h_mapa / h_frame)

        # Valida
        if 0 <= x_mapa < w_mapa and 0 <= y_mapa < h_mapa:
            self.profundidade_calibracao = self.mapa_profundidade[y_mapa, x_mapa]
            self.ponto_calibracao = (ponto_x, ponto_y)

            # Escala: pixels de profundidade → mm
            if profundidade_conhecida_mm is None:
                profundidade_conhecida_mm = self.referencia_profundidade_mm

            if self.profundidade_calibracao > 0:
                self.escala_px_para_mm = profundidade_conhecida_mm / self.profundidade_calibracao
                self.calibrado = True
                print(f"✓ Calibração OK! Escala: {self.escala_px_para_mm:.4f} mm/pixel de profundidade")
                self.esperando_clique = False
                return True

        print("✗ Ponto fora do mapa")
        return False

    def medir_ponto(self, ponto_x, ponto_y):
        """Mede a profundidade num ponto específico"""
        if self.mapa_profundidade is None:
            print("✗ Gera o mapa primeiro com [V]")
            return None

        if not self.calibrado:
            print("✗ Calibra primeiro com [C]")
            return None

        # Normaliza coordenadas
        h_mapa, w_mapa = self.mapa_profundidade.shape
        h_frame, w_frame = self.frame_atual.shape[:2]

        x_mapa = int(ponto_x * w_mapa / w_frame)
        y_mapa = int(ponto_y * h_mapa / h_frame)

        if 0 <= x_mapa < w_mapa and 0 <= y_mapa < h_mapa:
            profundidade_pixel = self.mapa_profundidade[y_mapa, x_mapa]

            # CORREÇÃO: Verifica se escala_px_para_mm é válida
            if self.escala_px_para_mm is None or self.escala_px_para_mm == 0:
                print("✗ Escala de calibração inválida! Calibra novamente.")
                return None

            profundidade_mm = profundidade_pixel * self.escala_px_para_mm

            self.pontos_medidos.append((ponto_x, ponto_y))
            self.profundidades_pontos.append(profundidade_mm)
            self.historico_medidas.append(profundidade_mm)

            print(f"✓ Ponto medido: {profundidade_mm:.2f} mm")
            return profundidade_mm

        print("✗ Ponto fora do mapa")
        return None

    def detectar_cavidade(self, limiar_profundidade=0.3):
        """Detecta automaticamente a cavidade (região mais profunda)"""
        if self.mapa_profundidade is None:
            print("✗ Gera o mapa primeiro com [V]")
            return None

        # Normaliza mapa
        mapa_norm = cv2.normalize(self.mapa_profundidade, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Threshold para encontrar regiões profundas
        _, thresh = cv2.threshold(mapa_norm, int(255 * limiar_profundidade), 255, cv2.THRESH_BINARY_INV)

        # Encontra contornos
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            # Maior contorno = cavidade principal
            maior_contorno = max(contornos, key=cv2.contourArea)
            self.bbox_cavidade = cv2.boundingRect(maior_contorno)
            self.cavidade_detectada = maior_contorno

            # Calcula profundidade média da cavidade
            x, y, w, h = self.bbox_cavidade
            regiao_cavidade = self.mapa_profundidade[y:y + h, x:x + w]

            # CORREÇÃO: Verifica se escala é válida antes de usar
            if self.escala_px_para_mm is not None and self.escala_px_para_mm > 0:
                prof_media_mm = np.mean(regiao_cavidade) * self.escala_px_para_mm
                prof_max_mm = np.max(regiao_cavidade) * self.escala_px_para_mm
                prof_min_mm = np.min(regiao_cavidade) * self.escala_px_para_mm

                print(f"✓ Cavidade detectada!")
                print(f"  Média: {prof_media_mm:.2f} mm | Máx: {prof_max_mm:.2f} mm | Mín: {prof_min_mm:.2f} mm")

                return {
                    "bbox": self.bbox_cavidade,
                    "contorno": maior_contorno,
                    "prof_media_mm": prof_media_mm,
                    "prof_max_mm": prof_max_mm,
                    "prof_min_mm": prof_min_mm
                }
            else:
                print("⚠ Cavidade detectada mas não calibrada para profundidade em mm")
                return {
                    "bbox": self.bbox_cavidade,
                    "contorno": maior_contorno,
                    "prof_media_mm": None
                }

        print("✗ Nenhuma cavidade detectada")
        return None

    def anotar_frame(self, frame):
        """Anota frame com informações de medição"""
        frame_out = frame.copy()
        h, w = frame_out.shape[:2]

        # Painel de estado
        estado_txt = f"Modo: {self.modo} | Calibrado: {'✓' if self.calibrado else '✗'}"
        cv2.rectangle(frame_out, (10, 10), (500, 120), (20, 20, 20), -1)
        cv2.rectangle(frame_out, (10, 10), (500, 120), (60, 60, 60), 1)
        cv2.putText(frame_out, estado_txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if self.calibrado and self.escala_px_para_mm:
            cv2.putText(frame_out, f"Escala: {self.escala_px_para_mm:.4f} mm/px",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if self.esperando_clique:
            cv2.putText(frame_out, "⏳ Á ESPERA DE CLIQUE...",
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if len(self.pontos_medidos) > 0:
            cv2.putText(frame_out, f"Pontos medidos: {len(self.pontos_medidos)}",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        # Desenha pontos medidos
        for i, (px, py) in enumerate(self.pontos_medidos):
            if i < len(self.profundidades_pontos):
                prof = self.profundidades_pontos[i]
                cv2.circle(frame_out, (px, py), 8, (0, 220, 255), -1)
                cv2.putText(frame_out, f"P{i + 1}: {prof:.1f}mm",
                            (px + 12, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        # Desenha cavidade detectada
        if self.cavidade_detectada is not None:
            cv2.drawContours(frame_out, [self.cavidade_detectada], 0, (0, 255, 0), 2)

        # Instruções
        instrucoes = [
            "[V] Gerar mapa | [C] Calibrar | [M] Medir | [D] Detectar cavidade",
            "[A] Analisar dataset | [H] Histórico | [S] Guardar | [Q] Sair"
        ]
        for i, txt in enumerate(instrucoes):
            cv2.putText(frame_out, txt, (10, h - 40 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame_out

    def guardar_medicoes(self, path="profundidades_validadas.json"):
        """Guarda as medições em JSON"""
        dados = {
            "timestamp": datetime.now().isoformat(),
            "calibrado": self.calibrado,
            "escala_px_para_mm": self.escala_px_para_mm,
            "referencia_mm": self.referencia_profundidade_mm,
            "pontos": [
                {"x": px, "y": py, "profundidade_mm": prof}
                for px, py, prof in zip(self.pontos_medidos, self.profundidades_pontos)
            ],
            "analise_dataset": self.analise_dataset,
            "validacao_historico": None  # Será preenchido se executado
        }
        with open(path, 'w') as f:
            json.dump(dados, f, indent=2)
        print(f"✓ Medições guardadas em {path}")

    def carregar_medicoes(self, path="profundidades_validadas.json"):
        """Carrega as medições de JSON"""
        if not os.path.exists(path):
            print(f"✗ Ficheiro {path} não encontrado")
            return False

        with open(path) as f:
            dados = json.load(f)

        self.calibrado = dados.get("calibrado", False)
        self.escala_px_para_mm = dados.get("escala_px_para_mm")
        self.referencia_profundidade_mm = dados.get("referencia_mm", 10.0)

        self.pontos_medidos = []
        self.profundidades_pontos = []

        for ponto in dados.get("pontos", []):
            self.pontos_medidos.append((ponto["x"], ponto["y"]))
            self.profundidades_pontos.append(ponto["profundidade_mm"])

        print(f"✓ Medições carregadas de {path}")
        return True


# ==============================================================================
# TESTE STANDALONE
# ==============================================================================
def mouse_callback(event, x, y, flags, param):
    """Callback para clicks do rato"""
    medidor, ref_mm = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if medidor.modo == "CALIBRAR":
            if medidor.calibrar(x, y, ref_mm):
                medidor.modo = "AGUARDAR"
                print("✓ Calibração concluída! Agora podes medir com [M]")
            medidor.esperando_clique = False

        elif medidor.modo == "MEDIR":
            medidor.medir_ponto(x, y)


def main():
    print("=" * 80)
    print("  VALIDADOR DE PROFUNDIDADE COM IA + DATASET + HISTÓRICO")
    print("  Para cavidades em peças plásticas com câmara RGB")
    print("=" * 80)

    if not DEPTH_AI_DISPONIVEL:
        print("\n✗ ERRO: Instala as dependências com:")
        print("  pip install torch transformers pillow")
        return

    ref_mm = input("\nProfundidade de referência conhecida em mm (ex: 10): ").strip()
    try:
        ref_mm = float(ref_mm)
    except ValueError:
        ref_mm = 10.0

    modelo_escolha = input("Escolhe o modelo (small/base/large) [small]: ").strip().lower()
    if modelo_escolha not in ["small", "base", "large"]:
        modelo_escolha = "small"

    medidor = ValidadorProfundidadeComDataset(modelo=modelo_escolha, referencia_profundidade_mm=ref_mm)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Câmara não encontrada!")
        return

    print("\n✓ Câmara iniciada.")
    print("  [V] Gerar mapa de profundidade")
    print("  [C] Calibrar (clica no fundo da cavidade)")
    print("  [M] Medir pontos (clica nos pontos)")
    print("  [D] Detectar cavidade automaticamente")
    print("  [A] Analisar dataset existente")
    print("  [H] Validar com histórico de falhas")
    print("  [S] Guardar medições validadas")
    print("  [Q] Sair")

    cv2.namedWindow("Validador Profundidade | VISION PRO")
    cv2.setMouseCallback("Validador Profundidade | VISION PRO", mouse_callback, (medidor, ref_mm))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # Modo VER MAPA
        if medidor.modo == "VER_MAPA" and medidor.mapa_profundidade is not None:
            display_frame = medidor.desenhar_mapa_profundidade(display_frame)

        # Anota frame
        display_frame = medidor.anotar_frame(display_frame)

        cv2.imshow("Validador Profundidade | VISION PRO", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        elif key == ord('v'):
            print("\n⏳ A gerar mapa de profundidade...")
            medidor.gerar_mapa_profundidade(frame)
            medidor.modo = "VER_MAPA"

        elif key == ord('c'):
            if medidor.mapa_profundidade is None:
                print("⚠ Primeiro gera o mapa com [V]")
            else:
                print("\n📍 Modo CALIBRAÇÃO ativado. Clica no fundo da cavidade.")
                medidor.modo = "CALIBRAR"
                medidor.esperando_clique = True

        elif key == ord('m'):
            if not medidor.calibrado:
                print("⚠ Primeiro calibra com [C]")
            else:
                print("\n📍 Modo MEDIÇÃO ativado. Clica nos pontos a medir.")
                medidor.modo = "MEDIR"
                medidor.esperando_clique = True

        elif key == ord('d'):
            print("\n🔍 A detectar cavidade...")
            medidor.detectar_cavidade()

        elif key == ord('a'):
            print("\n📊 A analisar dataset...")
            medidor.analisar_dataset_existente()

        elif key == ord('h'):
            print("\n📂 A validar com histórico...")
            validacao = medidor.validar_com_historico()
            if validacao:
                print("\n[RELATÓRIO DE VALIDAÇÃO]")
                print(f"Total de registos: {validacao['total_registos']}")
                print(f"OK: {validacao['ok_count']} | FAIL: {validacao['fail_count']}")
                if validacao.get('profundidade_ok_media'):
                    print(
                        f"Profundidade OK (média): {validacao['profundidade_ok_media']:.3f} ± {validacao['profundidade_ok_std']:.3f}")
                if validacao.get('profundidade_fail_media'):
                    print(
                        f"Profundidade FAIL (média): {validacao['profundidade_fail_media']:.3f} ± {validacao['profundidade_fail_std']:.3f}")

        elif key == ord('s'):
            medidor.guardar_medicoes()

        elif key == ord('r'):
            medidor.pontos_medidos = []
            medidor.profundidades_pontos = []
            print("✓ Pontos limpos")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

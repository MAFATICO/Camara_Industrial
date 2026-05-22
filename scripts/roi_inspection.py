"""
ROI INSPECTION MODULE
Aplica Regions of Interest (ROI) na inspeção de peças com base nos polígonos
desenhados durante a fase de treino no sistema_industrial_final.py
"""

import cv2
import numpy as np
import json
import sqlite3

DB_NAME = "COLOR_DETECTION/sistema_industrial.db"


class ROIInspection:
    """Classe para gestão e aplicação de ROIs na inspeção de peças"""
    
    def __init__(self, db_path=DB_NAME):
        """
        Inicializa o módulo de inspeção com ROI
        
        Args:
            db_path: Caminho da base de dados SQLite
        """
        self.db_path = db_path
        self.poligonos = []
        self.peca_id = None
        self.peca_nome = None
        self._cached_masks = {}
        self._cached_frame_shape = None
        self.regras_roi = {
            "presenca_area_min_rel": 0.01,
            "ok_area_min_rel": 0.02,
            "ok_fill_ratio_min": 0.01,
            "ok_fill_ratio_max": 0.95,
            "contorno_min_area_rel": 0.003,
        }

    def _limpar_estado_peca(self):
        """Limpa estado da peça ativa e caches de ROI."""
        self.poligonos = []
        self.peca_id = None
        self.peca_nome = None
        self._cached_masks = {}
        self._cached_frame_shape = None

    def _construir_cache_mascaras(self, frame_shape):
        """
        Pré-calcula máscaras para todos os ROIs da peça ativa.

        Args:
            frame_shape: Shape do frame (altura, largura, canais)
        """
        altura, largura = frame_shape[:2]
        self._cached_masks = {}

        for idx, poligono in enumerate(self.poligonos):
            if len(poligono) < 3:
                continue
            mascara = np.zeros((altura, largura), dtype=np.uint8)
            pts = np.array(poligono, dtype=np.int32)
            cv2.fillPoly(mascara, [pts], 255)
            self._cached_masks[idx] = mascara

        self._cached_frame_shape = frame_shape[:2]
        
    def carregar_poligonos_peca(self, peca_id):
        """
        Carrega os polígonos (ROI) desenhados para uma peça específica
        
        Args:
            peca_id: ID da peça na base de dados
            
        Returns:
            Lista de polígonos ou lista vazia se não encontrar
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT nome, poligonos FROM pecas WHERE id=?", (peca_id,))
                resultado = cursor.fetchone()

            if not resultado:
                self._limpar_estado_peca()
                return []

            self.peca_id = peca_id
            self.peca_nome = resultado[0]
            self.poligonos = json.loads(resultado[1]) if resultado[1] else []
            self._cached_masks = {}
            self._cached_frame_shape = None
            return self.poligonos

        except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as e:
            self._limpar_estado_peca()
            print(f"Erro ao carregar polígonos: {e}")
            return []
    
    def listar_pecas_disponiveis(self):
        """
        Lista todas as peças disponíveis na base de dados
        
        Returns:
            Lista de tuplos (id, nome)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, nome FROM pecas")
                pecas = cursor.fetchall()
                return pecas
        except sqlite3.Error as e:
            print(f"Erro ao listar peças: {e}")
            return []
    
    def aplicar_roi(self, frame, poligono, roi_idx=None):
        """
        Aplica uma máscara ROI baseada num polígono
        
        Args:
            frame: Imagem (frame) OpenCV
            poligono: Lista de pontos definindo o polígono
            
        Returns:
            Máscara binária do ROI e imagem mascarada
        """
        if len(poligono) < 3:
            return None, frame

        frame_shape = frame.shape[:2]
        if self._cached_frame_shape != frame_shape:
            self._construir_cache_mascaras(frame.shape)

        mascara = self._cached_masks.get(roi_idx) if roi_idx is not None else None
        if mascara is None:
            altura, largura = frame_shape
            mascara = np.zeros((altura, largura), dtype=np.uint8)
            pts = np.array(poligono, dtype=np.int32)
            cv2.fillPoly(mascara, [pts], 255)
            if roi_idx is not None:
                self._cached_masks[roi_idx] = mascara
        
        # Aplicar máscara à imagem
        imagem_roi = cv2.bitwise_and(frame, frame, mask=mascara)
        
        return mascara, imagem_roi
    
    def aplicar_todos_rois(self, frame):
        """
        Aplica todos os ROIs carregados a um frame
        
        Args:
            frame: Imagem OpenCV
            
        Returns:
            Dicionário com máscaras e imagens ROI para cada polígono
        """
        resultado = {}
        
        for idx, poligono in enumerate(self.poligonos):
            mascara, roi_img = self.aplicar_roi(frame, poligono, roi_idx=idx)
            if mascara is not None:
                resultado[f"roi_{idx}"] = {
                    "mascara": mascara,
                    "imagem": roi_img,
                    "poligono": poligono
                }
        
        return resultado
    
    def detectar_defeitos_em_roi(self, roi_imagem, mascara, threshold=100, usar_otsu=True):
        """
        Detecta possíveis defeitos na área ROI usando threshold adaptativo
        
        Args:
            roi_imagem: Imagem ROI da peça
            threshold: Valor de limiar para deteção
            
        Returns:
            Imagem binária com áreas suspeitas e número de contornos
        """
        # Converter para escala de cinza
        gray = cv2.cvtColor(roi_imagem, cv2.COLOR_BGR2GRAY) if len(roi_imagem.shape) == 3 else roi_imagem
        
        # Aplicar blur gaussiano para reduzir ruído
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold no ROI (fundo fora do ROI permanece ignorado)
        if usar_otsu:
            _, binaria = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binaria = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        binaria_roi = cv2.bitwise_and(binaria, binaria, mask=mascara)

        # Limpeza morfológica para reduzir ruído
        kernel = np.ones((3, 3), np.uint8)
        binaria_roi = cv2.morphologyEx(binaria_roi, cv2.MORPH_OPEN, kernel, iterations=1)
        binaria_roi = cv2.morphologyEx(binaria_roi, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Encontrar contornos apenas dentro do ROI
        res = cv2.findContours(binaria_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = res[0] if len(res) == 2 else res[1]

        area_roi = float(cv2.countNonZero(mascara))
        area_min_contorno = max(1.0, area_roi * self.regras_roi["contorno_min_area_rel"])
        contornos_filtrados = [c for c in contornos if cv2.contourArea(c) >= area_min_contorno]

        area_total_contornos = float(sum(cv2.contourArea(c) for c in contornos_filtrados))
        contorno_principal_area = float(max((cv2.contourArea(c) for c in contornos_filtrados), default=0.0))
        fill_ratio = (area_total_contornos / area_roi) if area_roi > 0 else 0.0
        area_principal_rel = (contorno_principal_area / area_roi) if area_roi > 0 else 0.0
        status_roi = self.classificar_roi(area_principal_rel, fill_ratio, len(contornos_filtrados))

        metricas = {
            "area_roi": int(area_roi),
            "num_contornos": len(contornos_filtrados),
            "area_total_contornos": area_total_contornos,
            "contorno_principal_area": contorno_principal_area,
            "fill_ratio": fill_ratio,
            "area_principal_rel": area_principal_rel,
            "status": status_roi,
        }

        return binaria_roi, contornos_filtrados, metricas
    
    def calcular_area_roi(self, mascara):
        """
        Calcula a área total do ROI
        
        Args:
            mascara: Máscara binária do ROI
            
        Returns:
            Número de pixels brancos na máscara
        """
        return cv2.countNonZero(mascara)

    def classificar_roi(self, area_principal_rel, fill_ratio, num_contornos):
        """Classifica ROI em AGUARDAR, OK ou FAIL."""
        # Se não há contornos ou a área é mínima, aguarda a peça
        if num_contornos == 0 or area_principal_rel < self.regras_roi["presenca_area_min_rel"]:
            return "AGUARDAR"

        # Se a área está na zona de dúvida (entre presença e OK), continuamos a aguardar por estabilidade
        if area_principal_rel < self.regras_roi["ok_area_min_rel"]:
            return "AGUARDAR"

        # Critérios de OK: área suficiente e preenchimento dentro dos limites
        if (self.regras_roi["ok_fill_ratio_min"] <= fill_ratio <= self.regras_roi["ok_fill_ratio_max"]):
            return "OK"

        # Caso contrário, é uma falha (ex.: peça mal posicionada ou defeito de forma)
        return "FAIL"

    def atualizar_regras_roi(self, **novas_regras):
        """Atualiza limites de decisão das ROIs em runtime."""
        for chave, valor in novas_regras.items():
            if chave in self.regras_roi:
                self.regras_roi[chave] = float(valor)
    
    def desenhar_rois_em_frame(self, frame, cores=(0, 255, 0), espessura=2):
        """
        Desenha os polígonos ROI sobre o frame
        
        Args:
            frame: Imagem OpenCV
            cores: Tuplo (B, G, R) para cor do desenho
            espessura: Espessura da linha
            
        Returns:
            Frame com polígonos desenhados
        """
        frame_copia = frame.copy()
        
        for idx, poligono in enumerate(self.poligonos):
            if len(poligono) > 2:
                pts = np.array(poligono, dtype=np.int32)
                cv2.polylines(frame_copia, [pts], True, cores, espessura)
                
                # Desenhar ponto inicial do polígono (vermelho)
                cv2.circle(frame_copia, tuple(pts[0]), 5, (0, 0, 255), -1)
                
                # Adicionar label do ROI
                centroide = np.mean(pts, axis=0).astype(int)
                cv2.putText(frame_copia, f"ROI {idx+1}", 
                           tuple(centroide), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 0), 2)
        
        return frame_copia
    
    def processar_frame_completo(self, frame, exibir_detalhes=False):
        """
        Processa um frame completo com todos os ROIs
        
        Args:
            frame: Imagem OpenCV
            exibir_detalhes: Se True, retorna informações adicionais
            
        Returns:
            Dicionário com resultados de processamento
        """
        resultado = {
            "peca_id": self.peca_id,
            "peca_nome": self.peca_nome,
            "num_rois": len(self.poligonos),
            "rois_dados": {},
            "status_global": "AGUARDAR",
            "motivo": "Sem deteção de peça"
        }
        
        for idx, poligono in enumerate(self.poligonos):
            mascara, roi_img = self.aplicar_roi(frame, poligono, roi_idx=idx)
            
            if mascara is not None:
                area = self.calcular_area_roi(mascara)
                binaria, contornos, metricas = self.detectar_defeitos_em_roi(roi_img, mascara)
                
                roi_info = {
                    "area_pixeis": area,
                    "num_contornos": metricas["num_contornos"],
                    "fill_ratio": metricas["fill_ratio"],
                    "area_contorno_principal": metricas["contorno_principal_area"],
                    "area_principal_rel": metricas["area_principal_rel"],
                    "status": metricas["status"],
                    "mascara": mascara,
                    "imagem_roi": roi_img
                }
                
                if exibir_detalhes:
                    roi_info["binaria"] = binaria
                    roi_info["contornos"] = contornos
                
                resultado["rois_dados"][f"roi_{idx}"] = roi_info

        statuses = [r["status"] for r in resultado["rois_dados"].values()]
        if any(s == "FAIL" for s in statuses):
            resultado["status_global"] = "FAIL"
            resultado["motivo"] = "Uma ou mais zonas ROI falharam"
        elif any(s == "OK" for s in statuses):
            resultado["status_global"] = "OK"
            resultado["motivo"] = "Peça dentro dos critérios das ROIs"
        else:
            resultado["status_global"] = "AGUARDAR"
            resultado["motivo"] = "Peça ausente nas ROIs"

        return resultado


def exemplo_uso():
    """Exemplo de utilização do módulo ROI Inspection"""
    
    # Inicializar inspetor ROI
    inspector = ROIInspection()
    
    # Listar peças disponíveis
    pecas = inspector.listar_pecas_disponiveis()
    print(f"Peças disponíveis: {len(pecas)}")
    for id_p, nome in pecas:
        print(f"  - ID {id_p}: {nome}")
    
    if not pecas:
        print("Nenhuma peça encontrada na base de dados.")
        return
    
    # Carregar primeira peça e seus ROIs
    peca_idx = 0
    peca_id, peca_nome = pecas[peca_idx]
    poligonos = inspector.carregar_poligonos_peca(peca_id)
    
    print(f"\nPeça carregada: {peca_nome} (ID: {peca_id})")
    print(f"Número de ROIs: {len(poligonos)}")
    
    if len(poligonos) == 0:
        print("Esta peça não tem ROIs definidos ainda. Desenhe na zona de treino!")
        return
    
    # Capturar vídeo da câmara
    cap = cv2.VideoCapture(0)
    
    print("\nProcessando vídeo em tempo real...")
    print("Pressione 'n' para próxima peça, 'p' para peça anterior, 'q' para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inverte horizontalmente a imagem da câmara (efeito espelho)
        frame = cv2.flip(frame, 1)
        
        # Processar frame com ROIs
        resultado = inspector.processar_frame_completo(frame)
        
        # Desenhar ROIs no frame
        frame_com_rois = inspector.desenhar_rois_em_frame(frame)
        
        # Exibir informações
        info_text = (
            f"Peça: {inspector.peca_nome} | ROIs: {resultado['num_rois']} | "
            f"Status: {resultado['status_global']}"
        )
        cv2.putText(frame_com_rois, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame_com_rois, "N: proxima | P: anterior | Q: sair", (10, frame_com_rois.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        
        # Exibir estatísticas de cada ROI
        y_offset = 60
        for roi_nome, roi_dados in resultado["rois_dados"].items():
            texto = (
                f"{roi_nome}: Area={roi_dados['area_pixeis']} px | "
                f"Cnt={roi_dados['num_contornos']} | Fill={roi_dados['fill_ratio']:.3f} | "
                f"{roi_dados['status']}"
            )
            cv2.putText(frame_com_rois, texto, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            y_offset += 25
        
        # Mostrar frame
        cv2.imshow("ROI Inspection - Vídeo", frame_com_rois)
        
        # Mostrar primeiro ROI em janela separada (se existir)
        if "roi_0" in resultado["rois_dados"]:
            roi_img = resultado["rois_dados"]["roi_0"]["imagem_roi"]
            cv2.imshow("ROI 1 (Isolado)", roi_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in (ord('n'), ord('N')) and pecas:
            peca_idx = (peca_idx + 1) % len(pecas)
            novo_id, novo_nome = pecas[peca_idx]
            novos_poligonos = inspector.carregar_poligonos_peca(novo_id)
            print(f"Peça ativa -> ID {novo_id}: {novo_nome} | ROIs: {len(novos_poligonos)}")
        if key in (ord('p'), ord('P')) and pecas:
            peca_idx = (peca_idx - 1) % len(pecas)
            novo_id, novo_nome = pecas[peca_idx]
            novos_poligonos = inspector.carregar_poligonos_peca(novo_id)
            print(f"Peça ativa -> ID {novo_id}: {novo_nome} | ROIs: {len(novos_poligonos)}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    exemplo_uso()


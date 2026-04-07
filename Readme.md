# ajustar menu de comandos
# ver o botão de limpar na lateral, limpa tudo do ecra e so funciona de novo quando vou a inspeção e volto
# desenho na aba treinar esta a ficar ao lado

# ==============================================================================
# MANUAL DE OPERAÇÃO PARA MÚLTIPLAS PEÇAS (IA)
# ==============================================================================
#
# 1. PASSO: TREINO GEOMÉTRICO 
#    - Seleciona a PEÇA 01 na barra lateral
#    - Vai à aba TREINAR e desenha o polígono (Tecla S)
#    - Repete isto para a PEÇA 02, 03, etc
#
# 2. PASSO: COLETA DE DADOS PARA IA 
#    - Com a PEÇA 01 selecionada, vai à aba INSPEÇÃO
#    - Coloca a peça na frente da câmara em várias posições e carrega em [ P ]
#      pelo menos 50 vezes. O sistema salva automaticamente o ID da classe (0)
#    - Seleciona a PEÇA 02 na lista lateral. Repete a tecla [ P ]
#      O sistema salva automaticamente o ID da classe (1)
#
# 3. COMANDOS ADICIONADOS:
#    - [ P ] : Captura foto limpa e gera ficheiro .txt (Auto-Labeling)
#    - [ N ] : Renomeia o modelo selecionado para facilitar a organização
#    - Pasta 'dataset_ia': Fica criada na raiz do projeto com todos os teus dados
#
# 4. PRÓXIMO NÍVEL:
#    - Quando tiveres centenas de fotos, usaremos um script de treino para gerar o
#      ficheiro 'best.pt' que fará a deteção automática de qual peça está no ecrã

# ==============================================================================
# MANUAL DOS COMANDOS
# ==============================================================================
#
#"[ S ]: Salvar Imagem/Poligono (Treino)"
#"[ I ]: Iniciar/Parar Inspecao"
#"[ P ]: Capturar Foto para Dataset IA"
#"[ R ]: Reset Contadores"
#"[ N ]: Renomear Peca Atual"
#"[ D ]: Delete Poligonos"
#"Botao Dir Mouse : Cancelar desenho"
#
# ==============================================================================
#
# ERROS CODIGO E COISAS A FAZER
#
# ==============================================================================
#
# - melhor a ispeção esta fraca
# - ver a parte das referncias que esta com problemas
# - corrigir a forma de inspeção
# - corrigir a parte do treino
# - verificar a implmentação de cadencia de produção
# - verificar a deteção de oclusão (camara suja)
# - # -- NOTAS --
# off_x é o valor que mede os pixeis para a esquerda e direita
# off_y é o valor que mede os pixeis  para a parte de cima do clique e para baixo
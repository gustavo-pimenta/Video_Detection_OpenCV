import cv2
import numpy as np
import math

# Definindo as cores dos tons de pele que serão analisados
# Alterar esses valores pode afetar a precisão do detector
lower_skin = np.array([0,20,70], dtype=np.uint8) # cor mínima
upper_skin = np.array([20,255,255], dtype=np.uint8) # cor máxima

# Criando a classe do objeto Vídeo
# Dentro desta classe estarão as funções resposáveis por extrair o vídeo da webcam, 
# atualizar a cada frame e aplicar os processamentos de imagem
class video_input(): 

    # função init utilizada quando o objeto é criado
    def __init__(self): 
        # define a fonte das imagens como sendo a webcam
        self.video = cv2.VideoCapture(0)

        # cria a variavel que sera utilizada para armazenar a quantidade de dedos reconhecidos
        self.fingers = []

        # variavel booleana de controle de segurança
        # para quando nenhuma mão for detectada
        self.NO_HAND = False

    # função update que ira captar a os frames do video
    def update(self):
        global gray_image, color_image, out_image

        # obtem o frame
        conection, self.frame = self.video.read()

        # salva o frame como imagem
        self.color_image = self.frame

        # converte para tons de cinza
        self.gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def hand_detector(self):
        # importa as variaveis de cor de pele que foram criandas
        global lower_skin, upper_skin

        # Define o ROI (region of interest = região de interesse)
        # se trata da area onde a mão será detectada
        # é bom definir uma area para isso, pois processar a imagem inteira 
        # seria pesado e iria afetar a precisão da detecção
        # um quadrado verde será desenhado para marcar a area do ROI
        cv2.rectangle(self.frame, (100, 100), (300, 300), (0, 255, ), 2)
        roi_image = self.frame[100:300, 100:300]
        
        # Aplicando Gaussian Blur na imagem para remover ruidos
        blur = cv2.GaussianBlur(roi_image, (3, 3), 0)
        self.blur = blur

        # Covertendo a imagem RGB para HSV para poder criar a máscara
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Criando a máscara binária
        # onde a mão será branca e o resto preto
        mask2 = cv2.inRange(hsv, lower_skin, upper_skin)

        # Criando o Kernel para realizar as modificações de matriz
        kernel = np.ones((5, 5))

        # Aplica Trnasformação Morfológica para filtrar o ruido do fundo da imagem 
        dilation = cv2.dilate(mask2, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)

        # Aplicar Gaussian Blur novamente 
        filtered = cv2.GaussianBlur(erosion, (3, 3), 0)

        # Aplicar metodo de Threshold para remover ruido
        ret, thresh = cv2.threshold(filtered, 127, 255, 0)

        # Encontrar os contornos da mão
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        try:
            # calcular a área máxima do contorno
            contour = max(contours, key=lambda x: cv2.contourArea(x))

            # desenhar um retangulo azul em volta do contorno
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(roi_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

            # Encontro o Convex Hull do contorno
            hull = cv2.convexHull(contour)

            # Desenhar o contorno em cima da imagem
            contour_draw = np.zeros(roi_image.shape, np.uint8)
            cv2.drawContours(contour_draw, [contour], -1, (0, 255, 0), 0)
            cv2.drawContours(contour_draw, [hull], -1, (0, 0, 255), 0)

            # Buscar por Defeitos de Convecção
            # Essa é a técnica responsável por descobrir a quantidade de dedos sendo mostrandos
            # ela faz isso contando os espaços vazios entre os dedos
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            count_defects = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                # desenha uma linha verde no contorno
                cv2.line(roi_image, start, end, [0, 255, 0], 2) 

                # se o angulo do defeito for maior que 90 graus
                # serão desenhados circulos no inicio, fim e meio do defeito
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(roi_image, far, 1, [0, 0, 255], -1)
                    cv2.circle(roi_image, start, 3, [255, 0, 0], -1)
                    cv2.circle(roi_image, end, 3, [255, 0, 0], -1)

            # Salva o numero de dedos detectados em uma variavel
            if count_defects == 0:
                self.fingers.append(1)
            elif count_defects == 1:
                self.fingers.append(2)
            elif count_defects == 2:
                self.fingers.append(3)
            elif count_defects == 3:
                self.fingers.append(4)
            elif count_defects == 4:
                self.fingers.append(5)
            else: 
                self.fingers.append(0)

            # deixa a variavel "No_hand" falsa pois a mão foi detectada
            self.NO_HAND = False

            # Salva as imagens que serão mostradas na tela
            self.frame = self.frame
            self.roi_image = np.hstack((contour_draw, roi_image))
            self.hand_draw = contour_draw
            self.roi_blackwhite = thresh

        except: 
            # deixa a variavel "No_hand" verdadeira pois a mão NÃO foi detectada
            self.NO_HAND = True
        
    def show_images(self):
        # aqui serão mostradas na tela as imagens do processo 
        # dessa forma o usuário pode acompanhar em tempo real 
        try:
            cv2.imshow('IMAGEM ORIGINAL',self.frame)
            cv2.imshow('IMAGEM EM TONS DE CINZA',self.gray_image)
            cv2.imshow('GAUSSIAN BLUR',self.blur)
            cv2.imshow('ROI REGIÃO DE INTERESSE',self.roi_image)
            cv2.imshow('MASCARA BINARIA',self.roi_blackwhite)
        except: pass
        
# função responsável por pegar os numeros extraidos e calcular a precisão deles
# mostrando para o usuário o quantidade de dedos que foi reconhecida e a precisão 
# desse valor em porcentagem
def fingers_count():

    # importa a variavel gatiklho global
    global gat

    # importa a variavel de detecção da mão do objeto video
    NO_HAND = video.NO_HAND

    # importa a variavel da quantidade de dedos
    fingers = video.fingers 

    if gat >= 10: # ativa o loop apenas se o gatilho tiver o valor necessário

        if NO_HAND == False: # ativa o loop apenas se a mão for detectada
            try:
                sum = 0 # cria a variavel soma
                for i in fingers:
                    sum = sum + i # soma as quantidade de dedos da lista
                mean = int(sum / len(fingers)) # calcula a media da quantidade de dedos  
            except: pass

            perc = 0 # variavel para porcentagem
            for i in fingers:
                if mean == i: # compara a media com a lista de dedos obtidos
                    perc += 1 
            try:
                perc = perc / len(fingers) # divide o valor da comparação pelo numero de termos
                perc = perc*100 # multiplica por cem para obter porcentagem
            except: pass
            video.fingers = [] # limpa a lista da quantidade de dedos

            print("\n\n\n\n\n\n\n\n\n\n") # quebras de linha para limpar o terminal
            print("NUMERO DE DEDOS RECONHECIDOS: ", mean)
            print("PRECISÃO DA INFORMAÇÃO: ", perc)
        else:
            # avisa o usuário caso nenhuma mao seja detectada
            print('NENHUMA MÃO FOI DETECTADA')
        gat = 0
    else:
        # aumenta a variavel gatilho
        gat+=1
    
# cria a variavel gatilho
gat=0
    
# inicia o objeto de video
video = video_input() 

while True:
    video.update()
    video.hand_detector()
    video.show_images()

    fingers_count()

    # break when ESC is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    
# release the video capture object
video.video.release()
cv2.destroyAllWindows()

from controller import Robot
import cv2 as cv
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from numpy.linalg import inv
cv.namedWindow('window_1')
cv.namedWindow('window_2')
## tiempo sampleo de la simulacion
TIME_STEP = 100
robot = Robot()
## parte de los sensores de distancia

##acceso a los nodos de vision del robot
camera_id=[]    
camara=['camera','camera(1)']
for i in range(2):
    camera_id.append(robot.getCamera(camara[i]))
    camera_id[i].enable(TIME_STEP)

k_1=0
es=3
# lectura de los parametros de la matrix de la camara
cv_file = cv.FileStorage("fer_camara_3.yaml", cv.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()
## componentes para el filtro
# componentes en azul a iltrar
fill_val_1=np.array([15,100,20],np.uint8)
fill_val_2=np.array([45,255,255],np.uint8)

def trackbar_callback_1(idx, value):
    fill_val_1[idx] = value
    
cv.createTrackbar('H', 'window_1', 255, 255, lambda v: trackbar_callback_1(2, v))
cv.createTrackbar('S', 'window_1', 255, 255, lambda v: trackbar_callback_1(1, v))
cv.createTrackbar('V', 'window_1', 255, 255, lambda v: trackbar_callback_1(0, v))


def trackbar_callback_2(idx, value):
    fill_val_2[idx] = value
    
cv.createTrackbar('H', 'window_2', 255, 255, lambda v: trackbar_callback_2(2, v))
cv.createTrackbar('S', 'window_2', 255, 255, lambda v: trackbar_callback_2(1, v))
cv.createTrackbar('V', 'window_2', 255, 255, lambda v: trackbar_callback_2(0, v))
# transformacion del sistema de corenadas
T=np.matrix([[0,1,0,-400],[-1,0,0,250],[0,0,1,0],[0,0,0,1]])
## bucle de simulacion de robot
while robot.step(TIME_STEP) != -1:
    if k_1>es:
        k=cv.waitKey(1)
        ## obtencion de la imagen de la camara izquierda
        image_l = camera_id[0].getImageArray()
        left=np.uint8(image_l)
        left=cv.rotate(left,cv.ROTATE_90_CLOCKWISE)
        b_l,g_l,r_l=cv.split(left)
        ## imagen final para imagen lista de la macara izquierda
        procesamiento_l=cv.merge([r_l,g_l,b_l])
        procesamiento_l=cv.flip(procesamiento_l,1)

        frameHSV_l=cv.cvtColor(procesamiento_l,cv.COLOR_BGR2HSV)
        
        ## filtro para el color azul hsv de la izquierda
        maskBlue_l=cv.inRange(frameHSV_l,fill_val_1,fill_val_2)
        maskBluevis_l=cv.bitwise_and(procesamiento_l,procesamiento_l,mask=maskBlue_l)
        
         ## obtencion de la imagen de la camara derecha
        image_r = camera_id[1].getImageArray()
        right=np.uint8(image_r)
        right=cv.rotate(right,cv.ROTATE_90_CLOCKWISE)
        b_r,g_r,r_r=cv.split(right)
        ## imagen final para imagen lista de la macara derecha
        procesamiento_r=cv.merge([r_r,g_r,b_r])

        procesamiento_r=cv.flip(procesamiento_r,1)

        frameHSV_r=cv.cvtColor(procesamiento_r,cv.COLOR_BGR2HSV)
        
        ## filtro para el color azul hsv de la derecha
        maskBlue_r=cv.inRange(frameHSV_r,fill_val_1,fill_val_2)
        maskBluevis_r=cv.bitwise_and(procesamiento_r,procesamiento_r,mask=maskBlue_r)
        ## calculo del area de la seccion de interes left

        contours, hierarchy=cv.findContours(maskBlue_l,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        for c in contours:
            area=cv.contourArea(c)
            if area>1000:
                M=cv.moments(c)
                if (M["m00"]==0): M["m00"]=1
                x=int(M["m10"]/M["m00"])
                y=int(M["m01"]/M["m00"])
                cv.circle(procesamiento_l, (x,y), 7, (0,255,0), -1)
                font = cv.FONT_HERSHEY_SIMPLEX
                contorno_limpio=cv.convexHull(c)
                x_real=y
                y_real=x
                cv.drawContours(procesamiento_l, [c], -1, (255,0,0), 3)
                entrada=np.matrix([[x_real],[y_real],[0],[1]])
                sistema_real=T@entrada
                if area < 4700:
                    cv.putText(procesamiento_l, 'CAMION FUERA DEL AREA',(200,430), font, 0.75,(0,0,0),1,cv.LINE_AA)
                else:
                    cv.putText(procesamiento_l, 'CAMION DENTRO DEL AREA',(200,430), font, 0.75,(0,0,0),1,cv.LINE_AA)

                cv.putText(procesamiento_l, '{}'.format(area),(200,460), font, 0.75,(0,0,0),1,cv.LINE_AA)

        

        ## muestra de datos de las camaras
        cv.imshow('left',procesamiento_l)
        cv.imshow('lef_filtrado',maskBluevis_l)
        
        if cv.waitKey(1) == ord('q'):
            break
    k_1=k_1+1

cap.release()
cv.destroyAllWindows() 
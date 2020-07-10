from controller import Robot
import cv2 as cv
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
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
image_counter=0


## bucle de simulacion de robot
while robot.step(TIME_STEP) != -1:
    if k_1>es:
        
        ## obtencion de la imagen de la camara izquierda
        image_l = camera_id[0].getImageArray()
        left=np.uint8(image_l)
        left=cv.rotate(left,cv.ROTATE_90_CLOCKWISE)
        b_l,g_l,r_l=cv.split(left)
        ## imagen final para imagen lista de la macara izquierda
        procesamiento_l=cv.merge([r_l,g_l,b_l])
        procesamiento_l=cv.flip(procesamiento_l,1)
        
        
         ## obtencion de la imagen de la camara derecha
        image_r = camera_id[1].getImageArray()
        right=np.uint8(image_r)
        right=cv.rotate(right,cv.ROTATE_90_CLOCKWISE)
        b_r,g_r,r_r=cv.split(right)
        ## imagen final para imagen lista de la macara derecha
        procesamiento_r=cv.merge([r_r,g_r,b_r])
        procesamiento_r=cv.flip(procesamiento_r,1)
        
        ## muestra de datos de las camaras
        cv.imshow('left_1',procesamiento_l)
        cv.imshow('right',procesamiento_r)
        k=cv.waitKey(1)
        if k%256==27:
            print("Escape hit, clossing")
            break
        elif k%256==32:
            img_name_l="opencv_frame_l_{}.png".format(image_counter)
            img_name_r="opencv_frame_r_{}.png".format(image_counter)
            cv.imwrite(img_name_l,procesamiento_l)
            cv.imwrite(img_name_r,procesamiento_r)
            print("{} writeen_l!".format(img_name_l))
            print("{} writeen_r!".format(img_name_r))
            image_counter +=1
            
      
   
    k_1=k_1+1
   
    

cv.destroyAllWindows() 
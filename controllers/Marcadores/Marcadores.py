from controller import Robot
import cv2 as cv
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from numpy.linalg import inv
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

marcador_1=0
marcador_2=1
camion=11
marcadores=[]
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
        
        
        ##procesamiento para aruco
        gray_l = cv.cvtColor(procesamiento_l, cv.COLOR_BGR2GRAY)
        
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        
        parameters = aruco.DetectorParameters_create()
        
        parameters.adaptiveThreshConstant = 10
        
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_l, aruco_dict, parameters=parameters)

        # font for displaying text (below)
        font = cv.FONT_HERSHEY_SIMPLEX
        if np.all(ids !=None):
            rvec, tvec ,_=aruco.estimatePoseSingleMarkers(corners, 0.1, mtx, dist)
            for i in range(0,ids.size):
                aruco.drawAxis(procesamiento_l, mtx, dist, rvec[i], tvec[i], 0.05)
            aruco.drawDetectedMarkers(procesamiento_l, corners)
            strg = ''
            for i in range(0,ids.size):
                strg += str(ids[i][0])+', '
            #for k in range(0,ids.size):
                #print(ids[k][0])
            posicion_1=np.array(np.where(ids==0))
            posicion_2=np.array(np.where(ids==11))
            posicion_3=np.array(np.where(ids==1))
            if (posicion_1.size==0 or posicion_2.size==0 or posicion_3.size==0)==True:
                cv.putText(procesamiento_l, "se perdio de vsta un marcador", (0,130), font, 1, (0,0,0),1,cv.LINE_AA)
            else:
                #cv.putText(procesamiento_l, "Id: " + strg, (0,30), font, 1, (0,0,0),1,cv.LINE_AA)
                #cv.putText(procesamiento_l, "T: " + str(tvec), (0,64), font, 1, (0,0,0),1,cv.LINE_AA)
                #cv.putText(procesamiento_l, "R: " + str(rvec), (0,130), font, 1, (0,0,0),1,cv.LINE_AA)

                trasnlacion_0=tvec[posicion_1[0][0]][0]
                trasnlacion_11=tvec[posicion_2[0][0]][0]
                trasnlacion_1=tvec[posicion_3[0][0]][0]

                rotacion_0=rvec[posicion_1[0][0]][0]
                rotacion_11=rvec[posicion_2[0][0]][0]
                rotacion_1=rvec[posicion_3[0][0]][0]
                
                Mz=np.matrix([[np.cos(np.pi/4),-np.sin(np.pi/4),0,0],[np.sin(np.pi/4),np.cos(np.pi/4),0,0],[0,0,1,0],[0,0,0,1]])
                My=np.matrix([[np.cos(rotacion_0[1]),0,np.sin(rotacion_0[1]),0],[0,1,0,0],[-np.sin(rotacion_0[1]),0,np.cos(rotacion_0[1]),0],[0,0,0,1]])
                Mx=np.matrix([[1,0,0,0],[0,np.cos(np.pi/2+np.pi/4),-np.sin(np.pi/2+np.pi/4),0],[0,np.sin(np.pi/2+np.pi/4),np.cos(np.pi/2+np.pi/4),0],[0,0,0,1]])
                tr=np.matrix([[1,0,0,trasnlacion_0[0]],[0,1,0,trasnlacion_0[1]],[0,0,1,trasnlacion_0[2]],[0,0,0,1]])

                cons=np.matrix([[0.7071,-0.7071,0,0],[-0.50,-0.50,-0.7071,0],[0.5,0.5,-0.7071,0],[0,0,0,1]])
                punto=np.matrix([[trasnlacion_11[0]],[trasnlacion_11[1]],[trasnlacion_11[2]],[1]])
                
                relativo=inv(tr@Mx@Mz)@punto
                Tranforma=np.matrix([[1,relativo[0],relativo[0]**2,0,0,0],[0,0,0,1,relativo[1],relativo[1]**2]])
                constantes=np.matrix([[0.075010778358096],[5.361855383129172],[6.705601056321730],[-0.033949252834476],[8.719894370404730],[-2.861599327071812]])
                relativo_real=Tranforma*constantes
                print("------------")
                print(mtx)
                print(relativo_real)
        else:
            cv.putText(procesamiento_l, "No Ids", (0,64), font, 1, (0,255,0),2,cv.LINE_AA)


        ## muestra de datos de las camaras
        cv.imshow('left_1',procesamiento_l)
        cv.imshow('right',procesamiento_r)
        #print(rvec)
        k=cv.waitKey(1)
        if k%256==27:
            print("Escape hit, clossing")
            break
    marcadores=[]
    k_1=k_1+1
cv.destroyAllWindows() 
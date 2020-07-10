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
## aprestura de txt para toma de datos
file=open("DATOS_X.txt",'w')
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
                cv.putText(procesamiento_l, "PERDIDA DE MARCADOR", (0,130), font, 1, (0,0,0),1,cv.LINE_AA)
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
                
                trasnlacion_11_real=np.array([[1,trasnlacion_11[0],trasnlacion_11[0]**2,trasnlacion_11[0]**3,0,0,0,0,0,0,0,0],[0,0,0,0,1,trasnlacion_11[1],trasnlacion_11[1]**2,trasnlacion_11[1]**3,0,0,0,0],[0,0,0,0,0,0,0,0,1,trasnlacion_11[2],trasnlacion_11[2]**2,trasnlacion_11[2]**3]])@np.array([[-0.074264371900440],[8.006181420630554],[-0.055330029643039],[-1.537629542436671],[0.052599792025994],[8.512653515900697],[0.228316718432177],[0.862756653763298],[-0.049619004946565],[8.614990178004565],[-0.303648114229638],[-0.027089944730950]])
                trasnlacion_0_real=np.array([[1,trasnlacion_0[0],trasnlacion_0[0]**2,trasnlacion_0[0]**3,0,0,0,0,0,0,0,0],[0,0,0,0,1,trasnlacion_0[1],trasnlacion_0[1]**2,trasnlacion_0[1]**3,0,0,0,0],[0,0,0,0,0,0,0,0,1,trasnlacion_0[2],trasnlacion_0[2]**2,trasnlacion_0[2]**3]])@np.array([[-0.074264371900440],[8.006181420630554],[-0.055330029643039],[-1.537629542436671],[0.052599792025994],[8.512653515900697],[0.228316718432177],[0.862756653763298],[-0.049619004946565],[8.614990178004565],[-0.303648114229638],[-0.027089944730950]])
                H=np.array([[0.707106781186548,-0.500000000000000,0.500000000000000,-4],[-0.707106781186548,-0.500000000000000,0.500000000000000,-4],[0,-0.707106781186548,-0.707106781186548,7.4]])
                trasnlacion_11_real=np.array([[float(trasnlacion_11_real[0])],[float(trasnlacion_11_real[1])],[float(trasnlacion_11_real[2])],[1]])
                sistema_coordenadas=H@trasnlacion_11_real
                x_d=0
                e_x=x_d-sistema_coordenadas[0]
                if ((abs(e_x)>0.30)):
                    cv.putText(procesamiento_l, "CAMION FUERA DE LA BALANZA", (0,360), font, 1, (0,0,0),1,cv.LINE_AA)
                else:
                    cv.putText(procesamiento_l, "CAMION DENTRO DE LA BALANZA", (0,360), font, 1, (0,0,0),1,cv.LINE_AA)
                print("error_x")
                print(e_x)
                print("--11 real--")
                print(sistema_coordenadas)
                
                ## para tomar datos
                
                if k%256==32:
                    file.write(str(trasnlacion_11[2])+"\n")
                    print("tomando datos")

                
                
        else:
            cv.putText(procesamiento_l, "No Ids", (0,64), font, 1, (0,255,0),2,cv.LINE_AA)


        ## muestra de datos de las camaras
        cv.imshow('left_1',procesamiento_l)
        #cv.imshow('right',procesamiento_r)
        #print(rvec)
        
        if k%256==27:
            print("Escape hit, clossing")
            file.close()
            break
    marcadores=[]
    k_1=k_1+1
cv.destroyAllWindows() 
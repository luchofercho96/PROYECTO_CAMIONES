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


## datos linea recta 1
recta1y=[]
recta1x=[]
recta2y=[]
recta2x=[]
distancia1=[]
distancia2=[]
## mascaras para los filtros
fill_val_1=np.array([15,100,20],np.uint8)
fill_val_2=np.array([45,255,255],np.uint8)
T=np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
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
        ## obtencion del color azul
        frameHSV_l=cv.cvtColor(procesamiento_l,cv.COLOR_BGR2HSV)
        
        ## filtro para el color azul hsv de la izquierda
        maskBlue_l=cv.inRange(frameHSV_l,fill_val_1,fill_val_2)
        maskBluevis_l=cv.bitwise_and(procesamiento_l,procesamiento_l,mask=maskBlue_l)
        ## calculo del area de la seccion de interes left

        contours, hierarchy=cv.findContours(maskBlue_l,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
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
                x_real=x
                y_real=y
                cv.drawContours(procesamiento_l, [contorno_limpio], -1, (255,0,0), 3)
                entrada=np.matrix([[x_real],[y_real],[0],[1]])
                sistema_real=T@entrada
                prueba1=np.array([sistema_real[0][0],sistema_real[1][0]]).reshape(1,2)

        ##procesamiento para aruco
        gray_l = cv.cvtColor(procesamiento_l, cv.COLOR_BGR2GRAY)
        
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        
        parameters = aruco.DetectorParameters_create()
        
        parameters.adaptiveThreshConstant = 10
        
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_l, aruco_dict, parameters=parameters)
        datos_esquinas=np.array(corners)
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
            ## poscion de los marcadres ya que cambian de posicion alteatoriamente    
            posicion_1=np.array(np.where(ids==0))
            posicion_2=np.array(np.where(ids==11))
            posicion_3=np.array(np.where(ids==1))
            posicion_4=np.array(np.where(ids==2))
            

            if (posicion_1.size==0 or posicion_4.size==0 or posicion_3.size==0)==True:
                cv.putText(procesamiento_l, "PERDIDA DE MARCADOR", (0,430), font, 1, (0,0,0),1,cv.LINE_AA)
            else:
                ## elemetos para graficar el marcador en el plano d ela imagen 
                punto_0=datos_esquinas[posicion_1[0][0]].reshape(4,2)
                punto_0[0][1]=punto_0[0][1]-35
                
                punto_11=prueba1
                
                punto_1=datos_esquinas[posicion_3[0][0]].reshape(4,2)
                punto_1[0][0]=punto_1[0][0]-35

                punto_2=datos_esquinas[posicion_4[0][0]].reshape(4,2)
                punto_2[0][0]=punto_2[0][0]+30
                punto_2[0][1]=punto_2[0][1]-10

                cv.circle(procesamiento_l, tuple(punto_0[0][:]), 5, (255,0,0), -1)
                cv.circle(procesamiento_l, tuple(punto_1[0][:]), 5, (255,0,0), -1)
                cv.circle(procesamiento_l, tuple(punto_2[0][:]), 5, (255,0,0), -1)

                #cv.line(procesamiento_l, tuple(punto_0[0][:]), tuple(punto_1[0][:]), (255, 0, 0),3)
                #cv.line(procesamiento_l, tuple(punto_0[0][:]), tuple(punto_2[0][:]), (0, 0, 0),3)
                ## calculo de la pendiente de las lineas
                fit = np.polyfit((punto_0[0][0],punto_2[0][0]), (punto_0[0][1],punto_2[0][1]), 1)
                fit1= np.polyfit((punto_0[0][0],punto_1[0][0]), (punto_0[0][1],punto_1[0][1]), 1)
                ## proyeccions de ambas lineas
                proyeccion_x=185
                proyeccion_y=fit[0]*proyeccion_x+fit[1]
                proyeccion_x_2=435
                proyeccion_y_2=fit1[0]*proyeccion_x_2+fit1[1]

                # GENERACION DELARE DE TRABAJO
                pts1=np.float32([[proyeccion_x,int(proyeccion_y)],[255,140],[punto_0[0][0],punto_0[0][1]],[proyeccion_x_2,int(proyeccion_y_2)]])
                pts2=np.float32([[0,0],[400,0],[0,600],[400,600]])
                MATRIX=cv.getPerspectiveTransform(pts1,pts2)

                

                ## dibujo circulos de ambas lineas
                cv.circle(procesamiento_l, tuple([proyeccion_x,int(proyeccion_y)]), 5, (255,0,0), -1)
                cv.line(procesamiento_l, tuple(punto_0[0][:]), tuple([proyeccion_x,int(proyeccion_y)]), (0, 0, 0),3)

                cv.circle(procesamiento_l, tuple([proyeccion_x_2,int(proyeccion_y_2)]), 5, (255,0,0), -1)
                cv.line(procesamiento_l, tuple(punto_0[0][:]), tuple([proyeccion_x_2,int(proyeccion_y_2)]), (0, 0, 0),3)

                cv.circle(procesamiento_l, (255,140), 5, (255,0,0), -1)
                
                resultado=cv.warpPerspective(procesamiento_l,MATRIX,(400,600))
                resultado_gray=cv.cvtColor(resultado,cv.COLOR_BGR2GRAY)
                resultado_gray=cv.medianBlur(resultado_gray,5)
                edges = cv.Canny(resultado,25,100)
                
                ## almecanamientos de los datos de la recta 1
                for j in range(int(proyeccion_x),int(punto_0[0][0])):
                    recta1y.append(fit[0]*j+fit[1])
                    recta1x.append(j)
                
                recta1_final=np.array([recta1x,recta1y])
                
                for n in range(0,recta1_final.shape[1]):
                    d=np.linalg.norm(np.array([recta1_final[0][n]-punto_11[0][0],recta1_final[1][n]-punto_11[0][1]]))
                    distancia1.append(d)

                distancia1_final=np.array([distancia1])
                result1 = np.where(distancia1_final == np.amin(distancia1_final))
                minimo1= np.amin(distancia1_final)
                cv.line(procesamiento_l, tuple(punto_11[0][:]), tuple([recta1_final[0][result1[1]],recta1_final[1][result1[1]]]), (0, 0, 0),1)
                

                for j in range(int(punto_0[0][0]),int(proyeccion_x_2)):
                    recta2y.append(fit1[0]*j+fit1[1])
                    recta2x.append(j)
                
                recta2_final=np.array([recta2x,recta2y])
                for n in range(0,recta2_final.shape[1]):
                    d1=np.linalg.norm(np.array([recta2_final[0][n]-punto_11[0][0],recta2_final[1][n]-punto_11[0][1]]))
                    distancia2.append(d1)

                distancia2_final=np.array([distancia2])
                result2 = np.where(distancia2_final == np.amin(distancia2_final))
                minimo2=np.amin(distancia2_final)
                cv.line(procesamiento_l, tuple(punto_11[0][:]), tuple([recta2_final[0][result2[1]],recta2_final[1][result2[1]]]), (0, 0, 0),1)
                
                ## VERIFICACION SI EL CAMION SE ENCUENTRA SOBRE LA LINEA DE PARQUEO
                recta2_final=np.trunc(recta2_final)
                recta1_final=np.trunc(recta1_final)
                
                

                err_x_linea1=-(recta1_final[0][result1[1]]-punto_11[0][0])
                err_y_linea1=recta1_final[1][result1[1]]-punto_11[0][1]

                err_x_linea2=(recta2_final[0][result2[1]]-punto_11[0][0])
                err_y_linea2=recta2_final[1][result2[1]]-punto_11[0][1]
                if (err_x_linea1>0 and err_y_linea1>0 and err_x_linea2>0 and err_y_linea2>0):
                    cv.putText(procesamiento_l, "CAMION DENTRO DEL AREA DE PARQUEO", (0,430), font, 1, (0,0,0),1,cv.LINE_AA)
                    if (recta1_final[0][result1[1]]-10<punto_11[0][0]<recta1_final[0][result1[1]]+10) and (recta1_final[1][result1[1]]-10<punto_11[0][1]<recta1_final[1][result1[1]]+10):
                        cv.putText(procesamiento_l, "MAL PARQUEO LATERAL", (0,460), font, 1, (0,0,0),1,cv.LINE_AA)
                    elif (recta2_final[0][result2[1]]-10<punto_11[0][0]<recta2_final[0][result2[1]]+10) and (recta2_final[1][result2[1]]-10<punto_11[0][1]<recta2_final[1][result2[1]]+10):
                        cv.putText(procesamiento_l, "MAL PARQUEO FRONTAL", (0,460), font, 1, (0,0,0),1,cv.LINE_AA)
                    else:
                        cv.putText(procesamiento_l, "BUEN PARQUEO", (0,460), font, 1, (0,0,0),1,cv.LINE_AA)
                else:
                    cv.putText(procesamiento_l, "CAMION FUERA DEL AREA DE PARQUEO", (0,430), font, 1, (0,0,0),1,cv.LINE_AA)

                
                
                almacenamiento_total_1=np.zeros([contorno_limpio.shape[0],recta1_final.shape[1]])

                for n_2 in range(0,contorno_limpio.shape[0]):
                    for n_1 in range(0,recta1_final.shape[1]):
                        dtotal=np.linalg.norm(np.array([recta1_final[0][n_1]-contorno_limpio[n_2][0][0],recta1_final[1][n_1]-contorno_limpio[n_2][0][1]]))
                        almacenamiento_total_1[n_2][n_1]=dtotal
                
                minimo_1_total = np.where(almacenamiento_total_1 == np.amin(almacenamiento_total_1))

                print(minimo_1_total[0])
                #print("controno")
                #print(contorno_limpio[minimo_1_total[0][0]][0])
                #print("recta")
                #print(int(recta1_final[0][minimo_1_total[1][0]]))
               
                almacenamiento_total_2=np.zeros([contorno_limpio.shape[0],recta2_final.shape[1]])
                for n_2 in range(0,contorno_limpio.shape[0]):
                    for n_1 in range(0,recta2_final.shape[1]):
                        dtota2=np.linalg.norm(np.array([recta2_final[0][n_1]-contorno_limpio[n_2][0][0],recta2_final[1][n_1]-contorno_limpio[n_2][0][1]]))
                        almacenamiento_total_2[n_2][n_1]=dtota2
                
                minimo_2_total = np.where(almacenamiento_total_2 == np.amin(almacenamiento_total_2))
                
                for lineas in contorno_limpio:
                    cv.line(procesamiento_l, tuple(lineas[0]),tuple([int(recta2_final[0][minimo_2_total[1][0]]),int(recta2_final[1][minimo_2_total[1][0]])]), (0, 0, 0),1)
                    cv.line(procesamiento_l, tuple(lineas[0]), tuple([int(recta1_final[0][minimo_1_total[1][0]]),int(recta1_final[1][minimo_1_total[1][0]])]), (0, 0, 0),1)
                
                cv.line(procesamiento_l, tuple(contorno_limpio[minimo_1_total[0][0]][0]), tuple([int(recta1_final[0][minimo_1_total[1][0]]),int(recta1_final[1][minimo_1_total[1][0]])]), (0, 255, 0),1)
                cv.line(procesamiento_l, tuple(contorno_limpio[minimo_2_total[0][0]][0]), tuple([int(recta2_final[0][minimo_2_total[1][0]]),int(recta2_final[1][minimo_2_total[1][0]])]), (0, 255, 0),1)
                #print(minimo1)
                #print("error res en x y 1")
                #print(err_x_linea1,err_y_linea1)
                #print("error res en x y 2")
                #print(err_x_linea2,err_y_linea2)
                
        else:
            cv.putText(procesamiento_l, "No Ids", (0,64), font, 1, (0,255,0),2,cv.LINE_AA)


        ##muestra de datos de las camaras
        cv.imshow('left_1',procesamiento_l)
        
        #cv.imshow('area gris',resultado_gray)
        #cv.imshow('bordes',maskBlue_l)
        
        if k%256==27:
            print("Escape hit, clossing")
            file.close()
            break
    recta1y=[]
    recta1x=[]
    distancia1=[]
    recta2y=[]
    recta2x=[]
    distancia2=[]
    marcadores=[]
    k_1=k_1+1
cv.destroyAllWindows() 
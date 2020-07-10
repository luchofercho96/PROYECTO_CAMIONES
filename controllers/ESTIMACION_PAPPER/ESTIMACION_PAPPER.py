from controller import Robot
import cv2 as cv
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
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

# componentes en azul a filtrar
azulbajo=np.array([115,100,20],np.uint8)
azulalto=np.array([140,255,255],np.uint8)
## matri de trnaformacion para el sistema de cordenadas
transpo=np.matrix([[1,0,0,0],[0,-1,0,500],[0,0,-1,0],[0,0,0,1]])

## DATOS DEL SISTEMA en pixels
u_l=[]
v_l=[]
u_r=[]
v_r=[]
x_p=[]
y_p=[]
z_p=[]
k=0
es=10
## lectura de las matrices de las camaras
cv_file = cv.FileStorage("fer_camara_3.yaml", cv.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

## matrices de cada camara

## bucle de simulacion de robot
while robot.step(TIME_STEP) != -1:
    if k>es:
        
        ## obtencion de la imagen de la camara izquierda
        image_l = camera_id[0].getImageArray()
        left=np.uint8(image_l)
        left=cv.rotate(left,cv.ROTATE_90_CLOCKWISE)
        b_l,g_l,r_l=cv.split(left)
        ## imagen final para imagen lista de la macara izquierda
        procesamiento_l=cv.merge([r_l,g_l,b_l])
        procesamiento_l=cv.flip(procesamiento_l,1)
        ## filtro pasar hsv la imagen izquierda
        frameHSV_l=cv.cvtColor(procesamiento_l,cv.COLOR_BGR2HSV)
        toma_izquierda=cv.cvtColor(procesamiento_l,cv.COLOR_BGR2GRAY)
        ## filtro para el color azul hsv de la izquierda
        maskBlue_l=cv.inRange(frameHSV_l,azulbajo,azulalto)
        maskBluevis_l=cv.bitwise_and(procesamiento_l,procesamiento_l,mask=maskBlue_l)
        
            ## obtencion de la imagen de la camara derecha
        image_r = camera_id[1].getImageArray()
        right=np.uint8(image_r)
        right=cv.rotate(right,cv.ROTATE_90_CLOCKWISE)
        b_r,g_r,r_r=cv.split(right)
        ## imagen final para imagen lista de la macara derecha
        procesamiento_r=cv.merge([r_r,g_r,b_r])
        procesamiento_r=cv.flip(procesamiento_r,1)

        toma_derecha=cv.cvtColor(procesamiento_r,cv.COLOR_BGR2GRAY)
        ## filtro pasar hsv la imagen derecha
        frameHSV_r=cv.cvtColor(procesamiento_r,cv.COLOR_BGR2HSV)
        
        ## filtro para el color azul hsv de la derecha
        maskBlue_r=cv.inRange(frameHSV_r,azulbajo,azulalto)
        maskBluevis_r=cv.bitwise_and(procesamiento_r,procesamiento_r,mask=maskBlue_r)
        
        stereo = cv.StereoBM_create(numDisparities=64, blockSize=15)
        disparity = stereo.compute(toma_izquierda,toma_derecha)
        #print(disparity.shape)
        ## contornos del izquierdo
        
        contours_l, hierarchy_l=cv.findContours(maskBlue_l,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        
        for c_l in contours_l:
            area_l=cv.contourArea(c_l)
           
            if area_l>500:
                
                M_l=cv.moments(c_l)
                if (M_l["m00"]==0): M_l["m00"]=1
                x_l=int(M_l["m10"]/M_l["m00"])
                y_l=int(M_l["m01"]/M_l["m00"])
                 ## generacion del rectangulo en la imagen
                rect_l=cv.minAreaRect(c_l)
                box_l=cv.boxPoints(rect_l)
                box_l=np.int0(box_l)
                ## generacion dle centroide
                cv.circle(procesamiento_l, (x_l,y_l), 7, (0,255,0), -1)
                font = cv.FONT_HERSHEY_SIMPLEX
                contorno_limpio_l=cv.convexHull(c_l)
                x_real_l=x_l
                y_real_l=y_l

                cv.drawContours(procesamiento_l, [contorno_limpio_l], 0, (0,0,0), 3)

                cv.drawContours(procesamiento_l,[box_l],0,(0,0,255),2)
                cv.circle(procesamiento_l, (tuple(box_l[0][:])), 5, (255,0,0), -1)
                cv.circle(procesamiento_l, (tuple(box_l[2][:])), 5, (0,255,0), -1)
                #print(box_l)

                entrada_l=np.matrix([[x_real_l],[y_real_l],[0],[1]])
                sistema_real_l=transpo@entrada_l
                
                cv.putText(procesamiento_l, '{},{}'.format(sistema_real_l[0,0],sistema_real_l[1,0]),(x_l+10,y_l+10), font, 0.75,(0,255,0),1,cv.LINE_AA)
        
        ## contornos del derecho
        
        contours_r, hierarchy_r=cv.findContours(maskBlue_r,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        for c_r in contours_r:
            area_r=cv.contourArea(c_r)
           
            if area_r>500:
                
                M_r=cv.moments(c_r)
                if (M_r["m00"]==0): M_r["m00"]=1
                x_r=int(M_r["m10"]/M_r["m00"])
                y_r=int(M_r["m01"]/M_r["m00"])
                ## generacion del rectangulo en la imagen
                rect_r=cv.minAreaRect(c_r)
                box_r=cv.boxPoints(rect_r)
                box_r=np.int0(box_r)
                ## generacion dle centroide
                cv.circle(procesamiento_r, (x_r,y_r), 5, (0,0,0), -1)
                font = cv.FONT_HERSHEY_SIMPLEX
                contorno_limpio_r=cv.convexHull(c_r)
                x_real_r=x_r
                y_real_r=y_r

                cv.drawContours(procesamiento_r, [contorno_limpio_r], 0, (0,0,0), 3)
                cv.drawContours(procesamiento_r,[box_r],0,(0,0,255),2)
                cv.circle(procesamiento_r, (tuple(box_r[0][:])), 5, (255,0,0), -1)
                cv.circle(procesamiento_r, (tuple(box_r[2][:])), 5, (0,255,0), -1)
            

                entrada_r=np.matrix([[x_real_r],[y_real_r],[0],[1]])
                sistema_real_r=transpo@entrada_r
                
                cv.putText(procesamiento_r, '{},{}'.format(sistema_real_r[0,0],sistema_real_r[1,0]),(x_r+10,y_r+10), font, 0.75,(0,255,0),1,cv.LINE_AA)

        
        
        ## reconstruccion 3D a travez de papper
        b=0.2-(-0.2)
        #print(b)
        #print(float(sistema_real_r[0,0])-float(sistema_real_l[0,0]))
        z=(-mtx[1][1]*b)/(float(disparity[y_real_l][x_real_l]))
        x=(-float(sistema_real_r[0,0])*z)/mtx[1][1]
        y=(-float(sistema_real_r[1,0])*z)/mtx[1][1]
        ##almacenamiento de pixeles de la izuqierda
        x_p.append(x)
        y_p.append(y)
        z_p.append(z)

        X=np.array(x_p)
        Y=np.array(y_p)
        Z=np.array(z_p)

        u_l.append(sistema_real_l[0,0])
        v_l.append(sistema_real_l[1,0])
        U_l=np.array(u_l)
        V_l=np.array(v_l)
        #almacenamiento de pixeles camara derecha
        u_r.append(sistema_real_r[0,0])
        v_r.append(sistema_real_r[1,0])
        U_r=np.array(u_r)
        V_r=np.array(v_r)
        print(x,y,z)
        #print(disparity[y_real_l][x_real_l])
        ## conversion del objeto a 3D
        ## muestra de datos de las camaras
        
        cv.imshow('left',procesamiento_l)
        cv.imshow('right',procesamiento_r)
        ## graficas de las posciiones en pixeles
        plt.figure(1)
        plt.clf()
        #plt.plot(U_l[es:k],V_l[es:k],'r-')
        #plt.plot(X[es:k],Y[es:k],'r-')
        plt.imshow(disparity,'gray')
        #plt.show()
        #plt.plot(U_r[es:k],V_r[es:k],'g-')
        plt.title('Movimiento del objeto')
        #plt.axis([0, 800, 0, 500])
        plt.grid()
        plt.ylabel('V PIXELS')
        plt.xlabel('U PIXELES')
        plt.draw()
        plt.pause(0.0000000001)
        if cv.waitKey(1) == ord('q'):
            break
   
    k=k+1
   
    
cap.release()
cv.destroyAllWindows() 
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

# componentes en azul a filtrar
azulbajo=np.array([115,100,20],np.uint8)
azulalto=np.array([140,255,255],np.uint8)
## matri de trnaformacion para el sistema de cordenadas
T=np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

## DATOS DEL SISTEMA en pixels
u_l=[]
v_l=[]
u_r=[]
v_r=[]
k=0
es=3
## lectura de las matrices de las camaras
cv_file = cv.FileStorage("fer_camara_3.yaml", cv.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()
## generate the test camera projection matrices 
P1 = np.eye(3, 4, dtype=np.float32)
P2 = np.array([[0,-1,0,1],[1,0,0,-1],[0,0,1,0]],dtype=np.float32)
print(P1)
print(P2)
## generate the testo points
N = 5
points3d = np.empty((4, N), np.float32)
points3d[:3, :] = np.random.randn(3, N)
points3d[3, :] = 1
print(points3d)
## proyect the 3d points into two view and add noise
points1 = P1 @ points3d
print("proyeccion")
print(points1)
points1 = points1[:2, :] / points1[2, :]
print("proyeccion")
print(points1)
points1[:2, :] += np.random.randn(2, N) * 1e-2
points2 = P2 @ points3d
points2 = points2[:2, :] / points2[2, :]
points2[:2, :] += np.random.randn(2, N) * 1e-2
## reconstruccion de los puntos a  traves de observaciones con ruido
points3d_reconstr = cv.triangulatePoints(P1, P2, points1, points2)
points3d_reconstr /= points3d_reconstr[3, :]
## impresion de la comparacionprint('Original points')
print(points3d[:3].T)
print('Reconstructed points')
print(points3d_reconstr[:3].T)


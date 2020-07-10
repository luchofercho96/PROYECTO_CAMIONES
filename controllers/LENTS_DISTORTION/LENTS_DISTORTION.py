import numpy as np
import cv2
import glob
## CARA DE LOS VALORES DE LA CAMARA
cv_file = cv2.FileStorage("fer_camara_3.yaml", cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

## LECUTRA DE LA IAMGEN A TRATAR
img = cv2.imread('opencv_frame_l_8.png')

pattern_size = (9, 7)
res, corners = cv2.findChessboardCorners(img, pattern_size)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 140, 0.001)
corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),corners, (10, 10), (-1,-1), criteria)
## IMAGEN SIN DISTORSION
h_corners = cv2.undistortPoints(corners, mtx, dist)
h_corners = np.c_[h_corners.squeeze(), np.ones(len(h_corners))]
img_pts, _ = cv2.projectPoints(h_corners, (0, 0, 0), (0, 0, 0),mtx,dist)
for c in corners:
    cv2.circle(img, tuple(c[0]), 10, (0, 255, 0), 1)
for c in img_pts.squeeze().astype(np.float32):
    cv2.circle(img, tuple(c), 5, (0, 0, 255), 1)
cv2.imshow('undistorted corners', img)
cv2.waitKey()
cv2.destroyAllWindows()
img_pts, _ = cv2.projectPoints(h_corners, (0, 0, 0), (0, 0, 0),mtx,dist)
for c in img_pts.squeeze().astype(np.float32):
    cv2.circle(img, tuple(c), 2, (255, 255, 0), 1)
cv2.imshow('reprojected corners', img)
cv2.waitKey()
cv2.destroyAllWindows()
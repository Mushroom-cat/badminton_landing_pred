import cv2
import numpy as np

# ===============================================================================
r'''
'''
# ===============================================================================

__all__ = ['CamCalib']

class CamCalib:
    '''
    相机的2D姿态转换成3D
    '''
    def setIntrinsicFile(self, fname):
        fs = cv2.FileStorage(fname, cv2.FILE_STORAGE_READ)
        self.CM1 = fs.getNode("CM1").mat()
        self.D1 = fs.getNode("D1").mat()
        self.CM2 = fs.getNode("CM2").mat()
        self.D2 = fs.getNode("D2").mat()
        self.R1 = fs.getNode('R1').mat()
        self.R2 = fs.getNode('R2').mat()
        self.P1 = fs.getNode('P1').mat()
        self.P2 = fs.getNode('P2').mat()
        self.Q = fs.getNode('Q').mat()

    def setExtrinsicFile(self, fname):
        fs = cv2.FileStorage(fname, cv2.FILE_STORAGE_READ)
        self.trans_mat = fs.getNode('ThreeDimTransMat').mat()

    def cvtPoint(self, left_point, right_point):
        pts_left = np.float32([[left_point[:]]]).reshape(-1, 1, 2)
        pts_right = np.float32([[right_point[:]]]).reshape(-1, 1, 2)

        # 你好像没有undistort
        rect_left = cv2.undistortPoints(pts_left, self.CM1, self.D1, R=self.R1, P=self.P1)  #
        rect_right = cv2.undistortPoints(pts_right, self.CM2, self.D2, R=self.R2, P=self.P2)  #

        # print('before', left_point, right_point)

        u_left_rect = rect_left[0, 0]
        u_right_rect = rect_right[0, 0]
        # print('after', u_left_rect, u_right_rect)

        disparity = u_left_rect[0] - u_right_rect[0]
        # print(u_left_rect[0], u_right_rect[0], disparity, 1.0)

        # 这里是用left和right的坐标分别算3D点，然后取平均
        point_2d_disp = np.array([u_left_rect[0], u_left_rect[1], disparity, 1.0])
        point_3d_hom = self.Q @ point_2d_disp
        point_3d_l = point_3d_hom[:3] / point_3d_hom[3]
        # print("Left 3D Point (X, Y, Z):", point_3d_l)

        point_2d_disp = np.array([u_right_rect[0], u_right_rect[1], disparity, 1.0])
        point_3d_hom = self.Q @ point_2d_disp
        point_3d_r = point_3d_hom[:3] / point_3d_hom[3]
        # print("Right 3D Point (X, Y, Z):", point_3d_r)

        # 这里取平均
        point_3d = (point_3d_l + point_3d_r) / 2.0
        # print("Averaged 3D Point (X, Y, Z):", point_3d)

        point_homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])

        # 应用外参变换矩阵
        world_point = np.dot(self.trans_mat, point_homogeneous)
        # print("World Coordinates (X, Y, Z):", world_point[:3])  # 只打印X, Y, Z部分
        x, y, z = [v.item() for v in world_point[:3]]
        return x, y, z


if __name__ == "__main__":
    left_point = (1447, 1151)
    right_point = (424, 1133)
    intrinsic_file = "20250710_intrinsic.yml"
    extrinsic_file = "20250710_extrinsic.yml"  # Replace with your actual YAML file path

    cam = CamCalib()
    cam.setIntrinsicFile(intrinsic_file)
    cam.setExtrinsicFile(extrinsic_file)
    x, y, z = cam.cvtPoint(left_point, right_point)

    print(f'输入2D left:{left_point} right:{right_point} 输出3D:{x, y, z}')

'''
left_point = (1447, 1151)
right_point = (424, 1133)
intrinsic_file = "20250710_intrinsic.yml"
extrinsic_file = "20250710_extrinsic.yml"  # Replace with your actual YAML file path

fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
CM1 = fs.getNode("CM1").mat()
D1 = fs.getNode("D1").mat()
CM2 = fs.getNode("CM2").mat()
D2 = fs.getNode("D2").mat()
R1= fs.getNode('R1').mat()
R2 = fs.getNode('R2').mat()
P1 = fs.getNode('P1').mat()
P2 = fs.getNode('P2').mat()
Q = fs.getNode('Q').mat()

print("CM1:\n", CM1)
print("D1:\n", D1)  
print("CM2:\n", CM2)
print("D2:\n", D2)

pts_left = np.float32([[left_point]]).reshape(-1, 1, 2)
pts_right = np.float32([[right_point]]).reshape(-1, 1, 2)

# 你好像没有undistort
rect_left = cv2.undistortPoints(pts_left, CM1, D1, R=R1, P=P1) # 
rect_right = cv2.undistortPoints(pts_right, CM2, D2, R=R2, P=P2) # 

print('before', left_point, right_point)

u_left_rect = rect_left[0, 0]
u_right_rect = rect_right[0, 0]
print('after', u_left_rect, u_right_rect)

disparity = u_left_rect[0] - u_right_rect[0]
print(u_left_rect[0], u_right_rect[0], disparity, 1.0)

# 这里是用left和right的坐标分别算3D点，然后取平均
point_2d_disp = np.array([u_left_rect[0], u_left_rect[1], disparity, 1.0])
point_3d_hom = Q @ point_2d_disp
point_3d_l = point_3d_hom[:3] / point_3d_hom[3]
print("Left 3D Point (X, Y, Z):", point_3d_l)

point_2d_disp = np.array([u_right_rect[0], u_right_rect[1], disparity, 1.0])
point_3d_hom = Q @ point_2d_disp
point_3d_r = point_3d_hom[:3] / point_3d_hom[3]
print("Right 3D Point (X, Y, Z):", point_3d_r)

# 这里取平均
point_3d = (point_3d_l + point_3d_r) / 2.0
print("Averaged 3D Point (X, Y, Z):", point_3d)


fs = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
trans_mat = fs.getNode('ThreeDimTransMat').mat()

point_homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])

# 应用外参变换矩阵
world_point = np.dot(trans_mat, point_homogeneous)
print("World Coordinates (X, Y, Z):", world_point[:3])  # 只打印X, Y, Z部分
'''
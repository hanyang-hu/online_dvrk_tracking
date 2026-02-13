import numpy as np
import yaml
import cv2

class StereoCamera():
    def __init__(self, cal_file_path, rectify = True, downscale_factor = 2, scale_baseline=1e-3):
        # Load cal file and get all the parameters
        # scale_baseline scales the baseline (i.e. converts units from mm to m!)
        f = open(cal_file_path)
        cal_data = yaml.load(f, Loader=yaml.FullLoader)
        
        self.K1 = np.array(cal_data['K1']['data']).reshape(3,3)
        self.K2 = np.array(cal_data['K2']['data']).reshape(3,3)
        self.D1 = np.array(cal_data['D1']['data'])
        self.D2 = np.array(cal_data['D2']['data'])
        
        self.rotation    = np.array(cal_data['R']['data']).reshape(3,3)
        self.translation = np.array(cal_data['T'])*scale_baseline
        self.T = np.eye(4)
        
        # Downscale stuff
        self.downscale_factor = downscale_factor
        self.img_size = np.array(cal_data['ImageSize'])/self.downscale_factor
        self.img_size = ( int(self.img_size[1]), int(self.img_size[0]) )
        self.K1 = self.K1/self.downscale_factor
        self.K2 = self.K2/self.downscale_factor
        
        self.K1[-1, -1] = 1
        self.K2[-1, -1] = 1
        
        # Prepare undistort and rectification (if desired) here
        if rectify:
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.K1, self.D1, self.K2, self.D2, 
                                                              self.img_size, self.rotation, self.translation)
            self.left_map1,  self.left_map2   = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1[:,:-1], 
                                                                            self.img_size, cv2.CV_32FC1)
            self.right_map1, self.right_map2  = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2[:,:-1], 
                                                                            self.img_size, cv2.CV_32FC1)
            self.K1 = P1[:,:-1]
            self.K2 = P2[:,:-1]
            
            self.rotation = np.eye(3)
            self.translation = np.linalg.norm(self.translation)*P2[:, -1]/np.linalg.norm(P2[:, -1])

        else:
            self.left_map1, self.left_map2   = cv2.initUndistortRectifyMap(self.K1, self.D1, np.eye(3), self.K1, 
                                                                           self.img_size, cv2.CV_32FC1)
            self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(self.K2, self.D2, np.eye(3), self.K2, 
                                                                           self.img_size, cv2.CV_32FC1)
        self.T[:3, :3] = self.rotation
        self.T[:3, -1] = self.translation
        
    def processImage(self, left_image, right_image):
        left_image  = cv2.resize(left_image,  self.img_size)
        right_image = cv2.resize(right_image, self.img_size)
        left_image  = cv2.remap(left_image,  self.left_map1,  self.left_map2,  interpolation=cv2.INTER_LINEAR)
        right_image = cv2.remap(right_image, self.right_map1, self.right_map2, interpolation=cv2.INTER_LINEAR)
        
        return left_image, right_image

    def projectPoints(self, points):
        # points is Nx3 np array
        points = np.transpose(points)
        projected_point_l = np.transpose(np.dot(self.K1, points/points[-1,:]))[:,:-1]
        
        points_homogeneous = np.concatenate( (points, np.ones((1, points.shape[1]))) )
        points_r = np.dot(self.T, points_homogeneous)[:-1, :]
        
        projected_point_r = np.transpose(np.dot(self.K2, points_r/points_r[-1,:]))[:,:-1]
        
        return projected_point_l, projected_point_r

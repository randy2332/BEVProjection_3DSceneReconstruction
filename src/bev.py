from hmac import new
from tkinter.tix import COLUMN
from unicodedata import ucd_3_2_0
import cv2
import numpy as np
import argparse

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):


        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        np.set_printoptions(precision=4, suppress=True)#to solve decimal point issue e.g. cos(pi/2) is not really 0 but very close

        bev_pixel=[]
        bev_point =[]
        front_point=[]
        front_pixel=[]
        new_pixels = []

        ### TODO ###
       
        
        '''image Coordinate'''
        principle_point = [self.height/2,self.width/2]      
        convention_matrix = np.eye(3)
        focal = -1 
        convention_matrix =  np.array([[focal, 0   , principle_point[0]],
                                    [0 , focal, principle_point[1]],
                                    [0 ,     0, 1]])

        print("Convention Matrix")
        print(convention_matrix)
  

        '''Convert to principle coordinate '''
        pointsz1 = points
        for point in pointsz1:
            point.append(1) #from (x,y) to (x,y,1)
            bev = np.dot(convention_matrix,np.array(point))
            bev_pixel.append(bev)
        print("bev_pixel")
        print(np.array(bev_pixel))   

        '''2d to 3d'''
        #here is the hardest part 
        w_p = 2.5 # not sure 
        focal_length = (self.height/2)/np.tan(np.deg2rad(fov/2))
        k_matrix = np.eye(3)
        k_matrix =  np.array([[focal_length, 0   , 0],
                            [0 , focal_length, 0],
                            [0 ,     0, 1]])
    
        k_matrix_inverse = np.linalg.inv(k_matrix)

        for bev in bev_pixel:
            bev_point.append(w_p*np.dot(k_matrix_inverse,bev))

        print("bev_point")
        print(np.array(bev_point))

        
        '''3D rotation'''
        # base on the lecture3 p21
        yaw = np.pi*((gamma)/180) # around z axis
        pitch = np.pi*((phi)/180) # around y axis
        roll = np.pi*(theta/180)  # around x axis

        rotation_yaw = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        rotation_pitch = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
        rotation_roll = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
        rotation = np.dot((np.dot(rotation_yaw,rotation_pitch)),rotation_roll)
    
        transformation_matrix = np.eye(4)
        transformation_matrix[:3,:3] = rotation
        transformation_matrix[:3,3] = [dx,dy,dz]
        print("transformation")
        print(transformation_matrix)
        

        '''Do rotate'''
        # we first augment a column of 1
        column_of_ones = np.ones((np.shape(bev_point)[0],1))
        bev_point = np.column_stack((bev_point,column_of_ones))
        
        for bev in bev_point:
            front_point.append(np.dot(transformation_matrix,bev))
        
        front_point = np.delete(front_point,-1, axis=-1)
        print("front_point")
        print(np.array(front_point))

        '''3d to 2d '''
        for front in front_point:
            front_pixel.append(np.dot(k_matrix,front)/front[2])

        print("front_pixel")
        print(np.array(front_pixel))

        '''from principle coordinate to pixel cooridate'''
        convention_matrix_inverse = np.linalg.inv(convention_matrix)
        for front in  front_pixel:
            new_pixel_x,new_pixel_y ,new_pixel_z= np.dot(convention_matrix_inverse,front)
            new_pixels.append([int(new_pixel_x),int(new_pixel_y),int(new_pixel_z)]) # the data type of new_pixels must be int 
        
        new_pixels = np.delete(new_pixels,-1, axis=-1)
        print("new_pixels")
        print(new_pixels)

     
        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """
        
        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image



def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    args = parser.parse_args()
    if args.floor == 1:
        floor =1
    elif args.floor == 2:
        floor =2

    pitch_ang = 90
    dy=1.5

    front_rgb = "bev_data/front"+str(floor)+".png"
    top_rgb = "bev_data/bev"+str(floor)+".png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang,dy=1.5)
    projection.show_image(new_pixels)
    cv2.destroyAllWindows()

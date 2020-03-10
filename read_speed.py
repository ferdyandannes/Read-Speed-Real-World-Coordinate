import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

import os
import numpy as np
import cv2
import xml.etree.cElementTree as ET
import copy
import matplotlib.pyplot as plt
import matplotlib
import math
import h5py
import json
import shutil

cv2_text_font = cv2.FONT_HERSHEY_COMPLEX
line_size = 3
poly_color = (0,0,0)
my_dpi = 100

def check_dir(dir_list):
    for d in dir_list:
        if not os.path.isdir(d):
            print('Create directory :\n' + d)
            os.makedirs(d)

def trajectories_color(object_id):
    # Color map of object trajectories
    # object_id : Object ID which come from tracker
    # return : Color (BGR)
    n = 15
    c = int(255/n)
    i =  object_id % 6
    k =  object_id % n
    
    if i == 1:
        color = (0, k*c, 255)
    elif i == 2:
        color = (k*c, 255, 0)
    elif i == 3:
        color = (255, 0, k*c)
    elif i == 4:
        color = (0, 255, 255-k*c)
    elif i == 5:
        color = (255, 255-k*c, 0)
    else:
        color = (255-k*c, 0, 255)

    return color

def distance_color(distance):
    # Color map of distance.
    # input(disrance) : Value of distance (meter).
    # output(size) : Text size for cv2.putText.
    if distance <= 7.5:
        color = (0,34*distance,255)
        size = 1.2-distance/45
    elif distance <= 15:
        color = (0,255,510-34*distance)
        size = 1.2-distance/45
    elif distance <= 22.5:
        color = (34*distance-510,255,0)
        size = 1.2-distance/45
    elif distance <= 30:
        color = (255,1020-34*distance,0)
        size = 1.2-distance/45
    elif distance <= 37.5:
        color = (255,0,34*distance-1020)
        size = 0.5
    else :
        color = (255,255,255)
        size = 0.4

    return color, size

def hilang(data_dir, full):
    # Save yg image belom ada (copym dari tracking)
    tr_dir = data_dir+'Tracking/'
    tr = os.listdir(tr_dir)
    tr.sort()

    print("len(tr) = ", len(tr))

    tr_path = tr_dir+tr[0]
    hilang = cv2.imread(tr_path)
    cv2.imwrite(full+'0000.png', hilang)

    tr_path = tr_dir+tr[1]
    hilang = cv2.imread(tr_path)
    cv2.imwrite(full+'0001.png', hilang)

    tr_path = tr_dir+tr[-1]
    imgnum = str(len(tr)-1).zfill(4)
    hilang = cv2.imread(tr_path)
    cv2.imwrite(full+imgnum+'.png', hilang)

# Added kalman filter into the system
class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
            (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def calculate_kalman(speed_temp):
    dt = 1.0/200
    # dt = 1.0/60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)


    # all_speed = []
    # for l in range(len(speed_temp)):
    #     all_speed += speed_temp[l]

    #all_speed = all_speed.astype('float64')
    #all_speed = np.array(all_speed, dtype=np.float64)

    predictions = []
    for z in speed_temp:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    return predictions

def filter_speed(data_dir):
    speed_path = os.path.join(data_dir,"speed.txt")

    with open(speed_path) as speed:
        speed_info = speed.readlines()

    # Copy a file
    src_dir = data_dir+"speed.txt"
    dst_dir = data_dir+"speed_filtered.txt"
    shutil.copy(src_dir,dst_dir)

    # ########################
    save = os.path.join(data_dir, 'speed_anjay.txt')
    position = open(save, 'w+')

    # Create raw
    for i in range(len(speed_info)):
        speed_read = speed_info[i].strip().split()
        frame = speed_read[0]
        position.write(frame)
        position.write("\n")
    position.close()

    # Process for the replaced file
    speed2_path = os.path.join(data_dir,"speed_anjay.txt")

    with open(speed2_path, "r+") as speeds:
        speed2_info = speeds.readlines()

    print("len = ", len(speed2_info))

    # Scan oll of the object ID
    list_obj = []
    for i in range(len(speed_info)):
        speed_read = speed_info[i].strip().split()

        each_obj_speed = speed_read[1:]
        object_id = each_obj_speed[::2]
        object_speed = each_obj_speed[1::2]

        for j in range(len(object_id)):
            objek = object_id[j]
            list_obj.append(objek)

        list_obj = list(dict.fromkeys(list_obj))

    print("list_obj = ", list_obj)

    # Save the speed in a list
    # First scan over the object
    for i in range(len(list_obj)):
        speed_temp = []
        frame_temp = []
        query_temp = []

        # Scan over the speed.txt
        for j in range(len(speed_info)):
            speed_read = speed_info[j].strip().split()

            frame = speed_read[0]
            each_obj_speed = speed_read[1:]
            object_id = each_obj_speed[::2]
            object_speed = each_obj_speed[1::2]

            # Scan over the each frame
            for k in range(len(object_id)):
                objek = object_id[k]
                kecepatan = object_speed[k]

                # Kalo sama save di suatu list
                if objek == list_obj[i]:
                    speed_temp.append(float(kecepatan))
                    frame_temp.append(frame)
                    query_temp.append(j)

        predictions = calculate_kalman(speed_temp)
        print("objek_id = ", list_obj[i])
        print("speed_temp = ", len(speed_temp))
        print("predictions = ", len(predictions))
        print("frame_temp = ", frame_temp)
        print("query_temp = ", query_temp)
        print("")

        id_now = list_obj[i]

        speed2_path = os.path.join(data_dir,"speed_anjay.txt")
        # with open(speed2_path, "r+") as speeds:
        #     speed2_info = speeds.readlines()

        speeds = open(speed2_path, 'r+')
        speed2_info = speeds.readlines()


        for x in range(len(speed2_info)):
            timpa = speed2_info[x].strip()
            tag = False
            #print("timpa = ", timpa)

            for y in range(len(query_temp)):
                urutan = query_temp[y]
                speed_now = predictions[y]

                # print("x = ", x, "  y = ", y)
                # print("urutan = ", urutan)

                if x == urutan:
                    tag = True
                    kec = '%.2f '%speed_now[0]
                    baru_tulis = timpa + " " + str(id_now) + " " + str(kec)
                    speeds.write(baru_tulis)
                    
            if tag == False:
                speeds.write(timpa)

            speeds.write("\n")

        speeds.close()

        # Take only the new
        n = len(speed2_info)
        nfirstlines = []

        with open(speed2_path) as f, open(os.path.join(data_dir,"bigfiletmp.txt"), "w") as out:
            for x in range(n):
                nfirstlines.append(next(f))
            for line in f:
                out.write(line)

        # NB : it seems that `os.rename()` complains on some systems
        # if the destination file already exists.
        os.remove(speed2_path)
        os.rename(os.path.join(data_dir,"bigfiletmp.txt"), speed2_path)

def banding(data_dir):
    speed_path = os.path.join(data_dir,"speed.txt")

    with open(speed_path) as speed:
        speed_info = speed.readlines()

    speed_path2 = os.path.join(data_dir,"speed_anjay.txt")

    with open(speed_path2) as speed2:
        speed_info2 = speed2.readlines()

    list_obj = []
    for i in range(len(speed_info)):
        speed_read = speed_info[i].strip().split()

        each_obj_speed = speed_read[1:]
        object_id = each_obj_speed[::2]
        object_speed = each_obj_speed[1::2]

        for j in range(len(object_id)):
            objek = object_id[j]
            list_obj.append(objek)

        list_obj = list(dict.fromkeys(list_obj))

    for i in range(len(list_obj)):
        speed_raw = []
        speed_fil = []
        sumbu_x = []
        for j in range(len(speed_info)):
            speed_read = speed_info[j].strip().split()
            each_obj_speed = speed_read[1:]
            object_id = each_obj_speed[::2]
            object_speed = each_obj_speed[1::2]

            speed_read2 = speed_info2[j].strip().split()
            each_obj_speed2 = speed_read2[1:]
            object_id2 = each_obj_speed2[::2]
            object_speed2 = each_obj_speed2[1::2]

            for k in range(len(object_id)):
                objek = object_id[k]

                kecepatan1 = object_speed[k]
                kecepatan2 = object_speed2[k]

                if objek == list_obj[i]:
                    speed_raw.append(float(kecepatan1))
                    speed_fil.append(float(kecepatan2))

        print("objek_id = ", list_obj[i])
        print("speed_raw = ", speed_raw)
        print("speed_fil = ", speed_fil)

        plt.plot(range(len(speed_raw)), speed_raw, label = 'Raw')
        plt.plot(range(len(speed_fil)), speed_fil, label = 'Filtered')
        plt.title('Speed Comparison')
        plt.xlabel('Frame')
        plt.ylabel('Speed')
        plt.legend()
        plt.show()


def read_speed(data_dir):
    position_path = os.path.join(data_dir,"position_all.txt")

    with open(position_path) as position:
        position_info = position.readlines()

    # Read the starting frame
    info = position_info[0].strip().split()
    starting_frame = int(info[0])

    # Save the speed information
    save_speed = os.path.join(data_dir, 'speed.txt')
    speed_info = open(save_speed, 'w+')

    # Read the tracking position
    track_dir = data_dir+'Tracking_Pos/'
    tracks = os.listdir(track_dir)
    tracks.sort()

    # Read the images for visualization
    image_dir = data_dir+'Images/'
    images = os.listdir(image_dir)
    images.sort()

    front_lane_dir =  data_dir + 'Lane_Lines/Combine/'
    front_lanes = os.listdir(front_lane_dir)
    front_lanes.sort()

    # Save the visualization
    full = data_dir+'Tracking_Full/'
    check_dir([full])

    # starting_frame == 1
    for i in range(starting_frame, len(position_info)):
        print("i = ", i)
        info_now = position_info[i].strip().split()
        info_past = position_info[i-1].strip().split()

        pos_n = {}
        pos_p = {}

        # Now
        object_info_n = info_now[3:]
        object_id_n = object_info_n[::3]
        object_x_n = object_info_n[1::3]
        object_y_n = object_info_n[2::3]

        for j in range(len(object_id_n)):
            pos_n[object_id_n[j]] = object_y_n[j]

        # Past
        object_info_p = info_past[3:]
        object_id_p = object_info_p[::3]
        object_x_p = object_info_p[1::3]
        object_y_p = object_info_p[2::3]

        for j in range(len(object_id_p)):
            pos_p[object_id_p[j]] = object_y_p[j]

        # Check the same object ID
        save_speed = {}
        for x in pos_n:
            # True = ada valuenya di past
            if str(x) in pos_p:
                sekarang = float(pos_n[x])
                sebelum = float(pos_p[x])
                delta_jarak = abs(sekarang - sebelum)
                kecepatan = (delta_jarak/0.033) * 3.6
                save_speed[x] = kecepatan
            else:
                save_speed[x] = 0

        if save_speed:
            # Read the tracking information
            num = info_now[0]
            print("num = ", num)

            with h5py.File(track_dir+num+'.h5','r') as fa:
                xmin_s = fa['xmin'].value
                ymin_s = fa['ymin'].value
                xmax_s = fa['xmax'].value
                ymax_s = fa['ymax'].value
                key_s = fa['key'].value
                dist_s = fa['dist'].value
                rlxd_s = fa['rlxd'].value
                llxd_s = fa['llxd'].value

            # Read the RGB image for the visualization
            img_path = image_dir+num+'.png'
            img = cv2.imread(img_path)
            front_lane_path = front_lane_dir+num+'.png'
            front_lane_img = cv2.imread(front_lane_path)

            print("save_speed = ", save_speed)
            print("key_s = ", key_s)

            for x in range(len(xmin_s)):
                xmin = xmin_s[x]
                ymin = ymin_s[x]
                xmax = xmax_s[x]
                ymax = ymax_s[x]
                key = key_s[x]
                distance = dist_s[x]
                rlxd = rlxd_s[x]
                llxd = llxd_s[x]

                # Draw the bounding box information
                box_color = trajectories_color(key)
                box_color = tuple ([int(x) for x in box_color])
                cv2.rectangle(front_lane_img,(xmin,ymin),(xmax,ymax),box_color,2)

                # Put the distance information
                dcolor, dsize = distance_color(distance)
                cv2.rectangle(front_lane_img, (xmin, ymin-int(dsize*30)), (xmin+int(dsize*80), ymin), (0,0,0), -1)
                cv2.putText(front_lane_img, '%2.1f'%distance, (xmin, ymin-int(dsize*5)), cv2_text_font, dsize, dcolor, 1, cv2.LINE_AA)

                # Put the speed information
                for y in save_speed:
                    if key == int(y):
                        kecepatan_draw = save_speed[y]

                # cv2.rectangle(front_lane_img, (xmax-int(dsize*80), ymin-int(dsize*30)), (xmax, ymin), (0,0,0), -1)
                # cv2.putText(front_lane_img, '%2.1f'%kecepatan_draw, (xmax-int(dsize*80), ymin-int(dsize*5)), cv2_text_font, dsize, dcolor, 1, cv2.LINE_AA)
                cv2.rectangle(front_lane_img, (xmin, ymax), (xmin+int(dsize*130), ymax+int(dsize*30)), (0,0,0), -1)
                cv2.putText(front_lane_img, '%2.1f'%kecepatan_draw+' km/h', (xmin, ymax+int(dsize*22)), cv2_text_font, dsize*0.7, (255,255,255), 1, cv2.LINE_AA)

                # Draw the lane gap information
                if (llxd >= 0) and (rlxd > 0):
                    cv2.putText(front_lane_img, '%.1f'%abs(rlxd), (xmin, ymax-int(dsize*5)), cv2_text_font, dsize*0.8, box_color, 1, cv2.LINE_AA)
                elif (llxd < 0) and (rlxd <= 0):
                    cv2.putText(front_lane_img, '%.1f'%abs(llxd), (xmax-int(dsize*40), ymax-int(dsize*5)), cv2_text_font, dsize*0.8, box_color, 1, cv2.LINE_AA)
                elif (llxd >= 0) and (rlxd <= 0):
                    if rlxd != 0:
                        cv2.putText(front_lane_img, '%.1f'%abs(rlxd), (xmax-int(dsize*40), ymax-int(dsize*5)), cv2_text_font, dsize*0.8, box_color, 1, cv2.LINE_AA)
                    if llxd != 0:
                        cv2.putText(front_lane_img, '%.1f'%abs(llxd), (xmin, ymax-int(dsize*5)), cv2_text_font, dsize*0.8, box_color, 1, cv2.LINE_AA)    

            cv2.imshow("front", front_lane_img)
            cv2.imwrite(full+num+'.png',front_lane_img)
            cv2.waitKey(1)

            # Save the speed information
            speed_info.write(num)
            for x in save_speed:
                kec = '%.2f '%save_speed[x]
                tulis = " " + str(x) + " " + str(kec)
                speed_info.write(tulis)
            speed_info.write("\n")
            print("")
        else:
            # Only draw the bounding box
            num = info_now[0]
            front_lane_path = front_lane_dir+num+'.png'
            front_lane_img = cv2.imread(front_lane_path)
            cv2.imwrite(full+num+'.png',front_lane_img)

    hilang(data_dir, full)

if __name__ == '__main__' :
    data_dir = "/media/ferdyan/LocalDiskE/Hasil/dataset/Final/1/"
    read_speed(data_dir)
    filter_speed(data_dir)
    banding(data_dir)

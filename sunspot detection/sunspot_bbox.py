import cv2
import math
import numpy as np
import csv
from functools import reduce
import operator
import datetime as dt
import JD
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
from sklearn.manifold import TSNE

def imgPrePro(channel):
    global img2a
    img1=channel 
    img2a = cv2.medianBlur(img1,7)
    img2 = cv2.bilateralFilter(img1,9,50,50)
    gamma = 0.985    
    img3 = np.array(255*(img2 / 255) ** (1/gamma), dtype = 'uint8')
    # cv2.imwrite('gamma.png',img3)
    return img3 

def cannyEdge(img2a):
    global center  
    SunRad=0
    SunDia=0
    # SunCircum=0
    img2a = cv2.Canny(img2a,130,255)
    contours,hierarchy = cv2.findContours(img2a,2,1)
    cnt = contours
    radList = []

    for i in range (len(cnt)):
        (x,y),radius = cv2.minEnclosingCircle(cnt[i])
        center = [int(y),int(x)]
        radius = int(radius)
        radList.append (radius)
        SunRad = max(radList)
        SunDia = (SunRad * 2)
        # SunCircum = 2*math.pi*SunRad
    contours = None
    cnt = None
    del radList
    imgCanny=img2a
    return  SunDia,imgCanny

def write_csv(data):    
    with open('YX_Coordinates.csv', mode='a') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
        data_writer.writerow(data)
        data_file.close()
        del data_writer
        del data

e1 = cv2.getTickCount()

# Load image, check image dimensions, remove watermark, convert to grayscale, and find edges
image = cv2.imread('sun19.jpg', 1)
final_image=image
cv2.rectangle(image, (0, 3960), (1340, 4096), (0,0,0), -1)
channel = image[:, :, 1]
img5 = imgPrePro(channel)

ret, thresh = cv2.threshold(img5, 175, 255, cv2.THRESH_BINARY)
ret, thresh1 = cv2.threshold(img5, 175, 255, cv2.THRESH_BINARY_INV)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel, iterations = 1)
# cv2.imwrite('sun7 opening.png',opening)
mask_opening = cv2.morphologyEx(opening,cv2.MORPH_OPEN,kernel, iterations = 2)
# cv2.imwrite('mask opening.png',mask_opening)

closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 1)
kernel = np.ones((3,3),np.uint8)
mask_closing = cv2.morphologyEx(closing,cv2.MORPH_CLOSE,kernel, iterations = 2)

# Find contour and sort by contour area
cnts = cv2.findContours(mask_closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Find bounding box and extract ROI
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ROI = img5[y:y+h, x:x+w]
    break
cropDia = str(ROI.shape[0])

SunDia,imgCanny=cannyEdge(img2a)
print('Canny Edge Sun Diameter :', SunDia)
print('Cropped Sun Diameter :', cropDia)

min_area = 21     #threshold area 
cnts1, hierarchy = cv2.findContours(mask_closing,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)   
n= len(cnts1)-1
cnts1=sorted(cnts1, key=cv2.contourArea, reverse=False)[:n]
sunspots=[]
total=0
Sum=[]
centroids=[]

for cnt in cnts1:        
    if cv2.contourArea(cnt) < min_area:
        continue
    M = cv2.moments(cnt)
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])
    cv2.circle(image, (cX, cY), 1, (0,255,0), -1)
    cv2.drawContours(image, cnt, -1, (0,255,0) ,1) 
    cv2.imwrite('sun7 contours.png',image)      
    resize = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_AREA)
    # cv2.imwrite('sun7 helio.bmp',resize)
    sunspots.append(cnt)
    # print("Sunspot area in pixels: ", (cv2.contourArea(cnt)))
    Sum.append((cv2.contourArea(cnt)))
    centroids.append([cX,cY])
    
print("Number of Sunspots found: " +str(len(sunspots)))
for i in range(0, len(Sum)):
    total = total + Sum[i]
# print("Total No. of Pixels inside all Sunspots: ", total)

# Generate mask
mask = np.zeros(thresh.shape)
cnts2 = sorted(cnts1, key=cv2.contourArea, reverse=False)
mask = cv2.drawContours(mask, cnts2, -1, 1, -1)
# cv2.imwrite("mask.png", np.uint8(255 * mask))

Sunspot_mask = cv2.bitwise_and(final_image, final_image, mask=mask_opening)
# cv2.imwrite('sunspot.png',Sunspot_mask)

count = (cv2.countNonZero(mask))
print("Total No. of Pixels in all Sunspots: ", count)

coord=cv2.findNonZero(mask)
ss_coord = coord.tolist()
ss_cord=reduce(operator.concat, ss_coord)
# ss_color=ss_cord
# for i in ss_color:
#     final_image[i[1],i[0]] = (255,0,0)   
# cv2.imwrite('Colored Sunspots.png',final_image)

# yy,mm,dd = 2019,4,(13+(4/24)+(0/(24*60)))
# jd = JD.date_to_jd(yy,mm,dd)  
# mean_anomaly=(357.528+0.9856003*(jd-2451545))
# mean_anomaly= math.radians(mean_anomaly-(360*(int(mean_anomaly/360))))
# dis_AU = 1.00014-0.01671*(math.cos(mean_anomaly))-0.00014*(math.cos(2*mean_anomaly))
# S = math.radians(0.2666/dis_AU)

# cx = 2048
# cy = 2048
# As = []
# Am = 0
# Bo = math.radians(-5.79)
# Lo = math.radians(302.36)
# B = []
# L = []
# # lst = [[1928,1671]]
# # lst = [[2550,2679]]
# for i in ss_cord:
#     if (i[0]>cx):
#         i[0]=i[0]-cx
#         x=(i[0])**2
#     elif (i[0]<cx):
#         i[0]=i[0]-cx
#         x=(i[0])**2
#     if (i[1]>cy):
#         i[1]=cy-i[1]
#         y=(i[1])**2
#     elif (i[1]<cy):
#         i[1]=cy-i[1]
#         y=(i[1])**2
#     r_ss=math.sqrt(x+y)
#     # if (1750<r_ss<1950):
#     roh=(math.asin(r_ss/(SunDia/2)))-(S*(r_ss/(SunDia/2)))
#     area_ss=1/(math.cos(roh))
#     theta=math.atan2(i[1],i[0])
#     Lat_rad=math.asin(math.cos(roh)*math.sin(Bo)+math.sin(roh)*math.cos(Bo)*math.sin(theta))
#     Helio_Lat_deg=math.degrees(Lat_rad)
#     Long_rad=math.asin((math.cos(theta)*math.sin(roh))/(math.cos(Lat_rad)))
#     Long_deg=math.degrees(Long_rad)
#     Helio_Long_rad=Lo+Long_rad
#     Helio_Long_deg=math.degrees(Helio_Long_rad)
#     B.append(Helio_Lat_deg)
#     L.append(Helio_Long_deg)
#     # print('Helio Latitude of Sunspot :', Helio_Lat_deg)
#     # print('Helio Longitude of Sunspot :', Helio_Long_deg)
#     As.append(area_ss)
    
# for j in range(0,len(As)):
#     Am+=As[j]

# fract_area = (Am*1E6/((SunDia/2)**2*np.pi*2))
# print('Area of Sunspot in mSH:', fract_area)  
# KISL_SS_area = 481
# acc = (fract_area/KISL_SS_area)*1E2
# print('Accuracy :', acc) 

# projection = TSNE().fit_transform(ss_cord)
# cluster = HDBSCAN(min_cluster_size=200, min_samples=150, cluster_selection_epsilon=145, allow_single_cluster=True).fit(ss_cord)
# %% Section for selecting best values
cluster = HDBSCAN(min_cluster_size=50, min_samples=69, cluster_selection_epsilon=128, allow_single_cluster=True).fit(ss_cord)
labels=cluster.labels_
n_clusters_ = len(set(labels))- (1 if -1 in labels else 0)
# print(f"Estimated number of Sunspot groups = {n_clusters_}")

# cluster.single_linkage_tree_.plot()
# plt.savefig('SIngle Tree.jpg')
# plt.close()

# Display the clustering graphically in a plot
# plt.scatter(*projection.T, c=labels, cmap='rainbow')
# plt.title(f"Estimated number of Sunspot groups: {n_clusters_}")
# plt.savefig('Sunspot Groups.jpg')
# plt.close()

labels=labels.tolist()
# find max num in a list
labels_sorted=list(labels)
labels_sorted.sort()
max_label=labels_sorted[-1]

# dummy lists to store index positions & pixel coordinates
color = [(255,0,0), (0,255,0), (0,255,255), (255,0,255), (255,255,0), (0,128,128), (128,0,11), (150, 160, 120), (111, 50, 1), (150, 100, 120), (111, 50, 100)]
noise_color = [(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255)]
indices_list, noise_indices_list = [],[]
cord_list, noise_cord_list = [],[]
for i in range(0,max_label+1):
    clusters_index, noise_index=[],[]
    clusters_coord, noise_coord=[],[]
    for j in range(len(labels)):
        if(labels[j]==-1):
            noise_index.append(j)
            noise_coord.append(ss_cord[j])
        elif(labels[j]==i):
            clusters_index.append(j)
            clusters_coord.append(ss_cord[j])
    if(len(noise_coord)>=i):
        noise_indices_list.insert(i, noise_index)
        noise_cord_list.insert(i,noise_coord)
        noise_index,noise_coord=[],[]
    if(len(clusters_coord)>=i):    
        indices_list.insert(i, clusters_index)
        cord_list.insert(i,clusters_coord)
        clusters_index,clusters_coord=[],[]

for cords in cord_list:
    number_of_pix = len(cords)
    # print(number_of_pix)
    if number_of_pix > 100:
        continue
    else:
        # print(number_of_pix)
        cord_list.remove(cords)
        noise_cord_list.append(cords)



num_clusters = len(cord_list)
print(f"Estimated number of Sunspot groups = {num_clusters}")

k=0
# NEW CHANGES
if noise_cord_list:
    for i in noise_cord_list:
        for j in i:
            final_image[j[1],j[0]] = noise_color[k]
        k+=1

k=0
for i in cord_list:
    for j in i:
        final_image[j[1],j[0]] = color[k]
    k+=1
# cv2.imwrite('Labelled Sunspot Groups.jpg',final_image)
    

    
# for cords in cord_list:
#     number_of_pix = len(cords)
#     print(number_of_pix)
#     if number_of_pix > 80:
#         continue
#     else:
#         print(number_of_pix)
#         cord_list.remove(cords)



# num_clusters = len(cord_list)
# print(f"Estimated number of Sunspot groups = {num_clusters}")

sg=[]
for cord in cord_list:
    avg = cord[int(len(cord)/2)]
    sg.append(avg)
    
    
a='SG-'
for i in range(len(sg)):
    image = cv2.putText(final_image, '{}{}'.format(a,i+1), (sg[i][0]-40,sg[i][1]-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,69,255), 4, cv2.LINE_AA)
# cv2.imwrite('Labelled Sunspot Groups.jpg',final_image)

# %% differentiating cluster based on distance to the right, left, top and bottom.

count=0
cluster_cen = []
for cord in cord_list:
    x,y = 0,0
    for point in cord: 
        x+= point[0]
        y+= point[1]
        count+=1
    x_avg = int(x/count)
    y_avg = int(y/count)
    count=0 
    cluster_cen.append([x_avg, y_avg])

count = 1
for cord, clust_cen in zip(cord_list, cluster_cen):
    cluster_right = []
    cluster_left = []
    cluster_top = []
    cluster_bottom = []
    dist_right = []
    dist_left = []
    dist_top = []
    dist_bottom = []
    distance=0
    
    
    cluster_cen_x = clust_cen[0]
    cluster_cen_y = clust_cen[1]
    
    for point in cord:
        point_x = point[0]
        point_y = point[1]
        
        if point_x > cluster_cen_x:
            cluster_right.append(point)
        elif point_x <= cluster_cen_x:
            cluster_left.append(point)
        
        if point_y > cluster_cen_y:
            cluster_bottom.append(point)
        elif point_y < cluster_cen_y:
            cluster_top.append(point)
    
            
    for points_r in cluster_right:
        point_x = points_r[0]
        point_y = points_r[1]
        
        x_euc_r = np.square(cluster_cen_x - point_x)
        y_euc_r = np.square(cluster_cen_y - point_y)
        y_euc_r = 0
        
        distance_r = np.sqrt(x_euc_r + y_euc_r)
        dist_right.append(int(distance_r))
        
    max_distance_r = max(dist_right)
    
    for points_l in cluster_left:
        point_x = points_l[0]
        point_y = points_l[1]
        
        x_euc_l = np.square(cluster_cen_x - point_x)
        y_euc_l = np.square(cluster_cen_y - point_y)
        # NEW
        y_euc_l = 0
        
        distance = np.sqrt(x_euc_l + y_euc_l)
        dist_left.append(int(distance))
        
    max_distance_l = max(dist_left)
    
    for points_t in cluster_top:
        point_x = points_t[0]
        point_y = points_t[1]
        
        x_euc_t = np.square(point_x - point_x)
        y_euc_t = np.square(point_y - cluster_cen_y)
        
        # NEW
        x_euc_t = 0
        
        distance = np.sqrt(x_euc_t + y_euc_t)
        dist_top.append(int(distance))
        
    max_distance_t = max(dist_top)
    
    for points_b in cluster_bottom:
        point_x = points_b[0]
        point_y = points_b[1]
        
        x_euc_b = np.square(cluster_cen_x - point_x)
        x_euc_b = 0
        # y_euc_b = np.square(cluster_cen_y - point_y)
        y_euc_b = np.square(point_y - cluster_cen_y)
        # print(x_euc_b, y_euc_b)
        

        
        distance = np.sqrt(x_euc_b + y_euc_b)
        dist_bottom.append(np.floor(distance))

        
    max_distance_b = np.max(dist_bottom)
    # print(max_distance_b)
    
    # diff_dist = (max_distance_r+max_distance_l)/(max_distance_t+max_distance_b)
    # print(f'SG {count} has a width to height ratio of {diff_dist}')
    # count+=1
    # # print(cluster_cen_x, cluster_cen_y)
    
    # bottom_right_x = cluster_cen_x + int(max_distance_r)
    # bottom_right_y = cluster_cen_y + int(max_distance_b)
    # # print(bottom_right_x, bottom_right_y)
    
    
    top_left = (cluster_cen_x - int((max_distance_l)),  cluster_cen_y - int((max_distance_t)))
    bottom_right = (cluster_cen_x + int((max_distance_r)), cluster_cen_y + int((max_distance_b)))
    # print(bottom_right)
    
    image = cv2.circle(final_image, (cluster_cen_x, cluster_cen_y), radius=0, color=(0, 0, 255), thickness=1)
    
    image = cv2.circle(final_image, (top_left), radius=0, color=(0, 0, 255), thickness=-1)
    image = cv2.circle(final_image, (bottom_right), radius=0, color=(0, 0, 255), thickness=-1)
    
    image = cv2.rectangle(final_image, top_left, bottom_right, (80, 200, 54), 1)
    cv2.imwrite('rectangle_sunspot_test.png', final_image)
    
        
# %%    

# for center, cords in zip(cluster_cen, cord_list):
#     all_distances = []
#     for pix in cords:
#         sg_point_x = center[0]
#         sg_point_y = center[1]
        
#         pix_point_x = pix[0]
#         pix_point_y = pix[1]
        
#         x_euc = np.square(sg_point_x - pix_point_x)
#         y_euc = np.square(sg_point_y - pix_point_y)
        
#         dist = np.sqrt(x_euc + y_euc)
#         # print(f'{x_euc} + {y_euc} = {dist}')
#         all_distances.append(dist)

#     max_distance = max(all_distances)
#     print(f'{max_distance} for {pix}')    
#     dist_array = np.array(all_distances)
#     maj_axis = int(max_distance)
#     min_axis = int(maj_axis/2)
#     # image = cv2.ellipse(final_image, (sg_point_x, sg_point_y), (maj_axis,min_axis), (0, 0, 255))
    
#     # now to add a rectangle
#     top_left = (sg_point_x - int(max_distance+1),  sg_point_y - int(max_distance+1))
#     bottom_right = (sg_point_x + int(max_distance+1), sg_point_y + int(max_distance+1))
#     image = cv2.rectangle(final_image, top_left, bottom_right, (80, 200, 54), 2)
#     cv2.imwrite('rectangle_sunspot_test.png', final_image)

# %% Calculating Time

e2 = cv2.getTickCount()
time = (e2 - e1)/cv2.getTickFrequency()
print(f"Elapsed time of code in seconds = {time}")   
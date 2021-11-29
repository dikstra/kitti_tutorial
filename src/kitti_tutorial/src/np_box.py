
import pandas as pd
import numpy as np

COLUMN_NAMES = ['frame','track_id','type','truncated','occluded','alpha','bbox_left','bbox_top','bbox_right','bbox_bottom','height','width','length','pos_x','pos_y','pos_z','rot_y']
df = pd.read_csv('/home/dikstra/2011_09_26/2011_09_26_drive_0005_sync/training/label_02/0000.txt',header=None,sep=' ')
df.columns = COLUMN_NAMES
df.head()

df.loc[df.type.isin(['Truck','Van','Tram']), 'type'] = 'Car'
df = df[df.type.isin(['Car','Pedestrian','Cyclist'])]

box = np.array(df.loc[2,['bbox_left','bbox_top','bbox_right','bbox_bottom']])
DETECTION_COLOR_DICT={'Car':(255,255.0),'Pedestrian':(0,226,255),'Cyclist':(141,40.255)}

import cv2

frame_id = 100

image = cv2.imread('/home/dikstra/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/%010d.png'%frame_id)

boxes = np.array(df[df.frame == frame_id ][['bbox_left','bbox_top','bbox_right','bbox_bottom']])
types = np.array(df[df.frame == frame_id ]['type'])
new_zip = zip(types, boxes)

for typ, box in new_zip:
    top_left = int(box[0]),int(box[1])
    bottom_right = int(box[2]),int(box[3])
    cv2.rectangle(image,top_left, bottom_right,DETECTION_COLOR_DICT[typ],2)

# top_left = int(box[0]),int(box[1])
# bottom_right = int(box[2]),int(box[3])
# cv2.rectangle(image,top_left, bottom_right,(255,255,0),2)
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destoryAllwindows()

np.array(df[df.frame == 0][['bbox_left','bbox_top','bbox_right','bbox_bottom']])

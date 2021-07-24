import cv2
import os
import math
import numpy as np
import random

framePath = 'frames'
maskPath  = 'masks'

os.path.exists(framePath) or os.mkdir(framePath)
os.path.exists(maskPath) or os.mkdir(maskPath)

unmaskedFrames = []

for f in os.listdir(framePath):
  if not os.path.exists(os.path.join(maskPath,f+'.npy')):
    unmaskedFrames.append(os.path.join(framePath,f))

random.shuffle(unmaskedFrames)

print(unmaskedFrames)

drawing = False
points  = []
px,py = 0,0

def drawBlock(event,x,y,flags,param):
  global drawing,radius,px,py,points

  if event==cv2.EVENT_LBUTTONDOWN:
    drawing=True
    px=x
    py=y
    points  = []
    points.append((x,y))

  elif event==cv2.EVENT_LBUTTONUP:
    drawing=False
    
  elif event==cv2.EVENT_MOUSEMOVE:     
    px=x
    py=y
    if drawing:
      lx,ly = points[-1]
      distance = math.sqrt( ((px-lx)**2)+((py-ly)**2) )
      if distance > 60:
        if len(points)>0:
          numpoints = 5
          dx = lx-px
          dy = ly-py
          for n in range(1,numpoints):
            points.append((int(lx-((dx/numpoints)*n )),int(ly-((dy/numpoints)*n ))))  
        else:
          points.append((px,py))



cv2.namedWindow('unmaskedFrame')
cv2.setMouseCallback('unmaskedFrame',drawBlock)

for unmaskedFrame in unmaskedFrames:
  im     = cv2.imread(unmaskedFrame)

  factor = 1

  imw,imh,_ = im.shape
  maxdim = max(imw,imh)
  if maxdim > 1024:
    factor = maxdim/1024
  print(factor)

  im = cv2.resize(im,None,fx=1/factor,fy=1/factor,interpolation=cv2.INTER_AREA)


  if im is None:
    continue

  points = []
  while 1:
    temp = im.copy()
    cv2.circle(temp, (px,py), 10, (255,0,255), 1)
    if drawing:
      tempPoints = points[:]+[(px,py)]
    else:
      tempPoints = points[:]
      
    if len(tempPoints)>1:
      tempPoints = np.array(tempPoints,np.int32)
      tempPoints.reshape((-1, 1, 2))
      print(tempPoints)
      cv2.polylines(temp,[tempPoints],True,(255,0,255))

      for x,y in points:
        cv2.circle(temp,(x,y),2,(255,255,255),-1)

    cv2.imshow('unmaskedFrame', temp )
    k = cv2.waitKey(1)
    if k == ord('q'):
      exit()
    elif k == ord('h'):
      break
    elif k == ord('y'):
      fn = os.path.split(unmaskedFrame)[1]
      fn = os.path.join(maskPath,fn)
      tempPoints = np.array(points,np.int32)
      np.save(fn,tempPoints)
      break
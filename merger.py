import cv2
import os
import numpy as np
import random 
framePath = 'frames'
maskPath  = 'masks'

os.path.exists(framePath) or os.mkdir(framePath)
os.path.exists(maskPath) or os.mkdir(maskPath)

maskedFrames = []

for f in os.listdir(framePath):
  if os.path.exists(os.path.join(maskPath,f+'.npy')):
    maskedFrames.append( (os.path.join(maskPath,f+'.npy'),os.path.join(framePath,f)) )


random.shuffle(maskedFrames)
print(maskedFrames)

cw,ch = 1024,1024*2

placedFrames = []
canvas       = np.zeros((cw,ch,3),np.uint8)
px,py = 0,0
placed = False
changezoom=0

def drawBlock(event,x,y,flags,param):
  global placed,px,py,changezoom
  if event==cv2.EVENT_LBUTTONDOWN:
    px=x
    py=y
    placed=True
  elif event==cv2.EVENT_MOUSEMOVE:     
    px=x
    py=y
  elif event==cv2.EVENT_MOUSEWHEEL:
    if flags>0:
      changezoom = 1
    else:
      changezoom = -1

cv2.namedWindow('canvas')
cv2.setMouseCallback('canvas',drawBlock)

qpressed=False
for maskPath,framePath in maskedFrames:
  if qpressed:
    break

  factor = 4

  frameo  = cv2.imread(framePath)
  frame  = cv2.resize(frameo,None,fx=1/factor,fy=1/factor,interpolation=cv2.INTER_AREA)
  maskRect = np.load(maskPath)/factor
  xo,yo = np.mean( maskRect,axis=0)

  xo = int(xo)
  yo = int(yo)

  while 1:
    tempCanvas = canvas.copy()

    if changezoom != 0:
      factor += changezoom/10
      if factor<1:
        factor = 1

      changezoom=0

      frame  = cv2.resize(frameo,None,fx=1/factor,fy=1/factor,interpolation=cv2.INTER_AREA)
      maskRect = (np.load(maskPath)/factor).astype(int)
      xo,yo = np.mean( maskRect,axis=0)
      xo = int(xo)
      yo = int(yo)


    for _,lastmr,x,y,h,w in placedFrames:
      cv2.rectangle(tempCanvas, (x,y), ((x+h,y+w)), (10,10,10), -1)

    for _,lastmr,x,y,h,w in placedFrames:
      tempPoints = np.array(lastmr,np.int32)
      tempPoints.reshape((-1, 1, 2))
      cv2.polylines(tempCanvas,[tempPoints],True,(100,100,100))

    tempPoints = np.array(maskRect,np.int32)+(px-xo,py-yo)
    tempPoints.reshape((-1, 1, 2))

    cv2.rectangle(tempCanvas, (px-xo,py-yo), ((px-xo+frame.shape[1],py-yo+frame.shape[0])), (50,50,50), 2)
    cv2.polylines(tempCanvas,[tempPoints],True,(255,0,255))

    cv2.imshow('canvas',tempCanvas)
    k = cv2.waitKey(1)
    if k == ord('q'):
      qpressed=True
      break
    if placed:
      placedFrames.append( (frame,maskRect+(px-xo,py-yo),px-xo,py-yo,frame.shape[1],frame.shape[0]) )
      placed=False
      break

subdiv = cv2.Subdiv2D((0,0,ch,cw))

for _,lastmr,_,_,_,_ in placedFrames:
  for px,py in lastmr:
    print('point',px,py)
    subdiv.insert((px,py))

facetList,facetCentres = subdiv.getVoronoiFacetList([])

tempCanvas = canvas.copy()

indexToFacets = {}


minx,maxx = float('inf'),float('-inf')
miny,maxy = float('inf'),float('-inf')

coordToIndex = {}

for i,(_,lastmr,_,_,_,_) in enumerate(placedFrames):
  for px,py in lastmr:
    coordToIndex[(px,py)] = i

canvas = np.zeros((cw,ch,3),np.uint8)

for facet,(cx,cy) in zip(facetList,facetCentres):

  minx = min(minx,cx)
  miny = min(miny,cy)

  maxx = max(maxx,cx)
  maxy = max(maxy,cy)

  tempPoints = np.array(facet,np.int32)
  tempPoints.reshape((-1, 1, 2))
  cv2.polylines(canvas,[tempPoints],True,(100,100,100))  

  cv2.imshow('canvas',canvas)
  k = cv2.waitKey(1)

  matchIndex = coordToIndex.get((cx,cy))
  if matchIndex is not None:
    indexToFacets.setdefault(matchIndex,[]).append(facet)
  else:
    print('NOMATCH')

canvas = np.zeros((cw,ch,3),np.uint8)

for i,(frame,_,px,py,_,_) in enumerate(placedFrames):
  facets = indexToFacets[i]
  source = np.zeros((cw,ch,3),np.uint8)
  mask = np.zeros((cw,ch,3),np.uint8)

  maxh = frame.shape[0]
  if maxh > source.shape[0]-py:
    maxh = source.shape[0]-py

  maxw = frame.shape[1]
  if maxw > source.shape[1]-px:
    maxw = source.shape[1]-px

  source = cv2.resize(frame,(source.shape[1],source.shape[0]))
  source = cv2.GaussianBlur(source,(515,515),cv2.BORDER_DEFAULT)

  source[py:py+maxh,px:px+maxw,:]=frame[:maxh,:maxw,:]

  for facet in facets:
    tempPoints = np.array(facet,np.int32)
    tempPoints.reshape((-1, 1, 2))
    cv2.fillPoly(mask,[tempPoints],(255,255,255))

  foreground = source.astype(float)
  background = canvas.astype(float)

  mask = cv2.medianBlur(mask,27,cv2.BORDER_DEFAULT)
  mask = cv2.blur(mask,(37,37),cv2.BORDER_DEFAULT)

  mask = mask.astype(float)/255

  foreground = cv2.multiply(mask, foreground)
  background = cv2.multiply(1.0 - mask, background)

  canvas = cv2.add(foreground, background).astype(np.uint8)

  cv2.imshow('canvas',canvas)
  k = cv2.waitKey(1)


print(minx,maxx,miny,maxy)

minx,maxx,miny,maxy = int(minx),int(maxx),int(miny),int(maxy)
canvas =  canvas[miny:maxy,minx:maxx,:]

cv2.imshow('canvas',canvas)
k = cv2.waitKey(0)


import time
cv2.imwrite(str(time.time())+'.png', canvas)

"""


  cv2.imshow('canvas',tempCanvas)
  k = cv2.waitKey(100)
  if k == ord('q'):
    break
"""
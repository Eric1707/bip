import numpy as np
import math

print("---basic image process tools are imported in---")
##############################################################
## function: const size blur
## argument: img(numpy.ndarray), fold(int from 1 to 9999)
## 从图像中等间隔抽取行/列，然后复制到相邻行/列
##############################################################
def constSizeBlur(img, fold):
    if type(fold)==int and fold>0 and fold<10000:
        img_new=img
        i=fold-1
        while i<img.shape[0]:
            j=i-fold+1
            while j<i:
                img_new[j,:,:]=img[i,:,:]
                j=j+1
            i=i+fold
        i=fold-1
        while i<img.shape[1]:
            j=i-fold+1
            while j<i:
                img_new[:,j,:]=img[:,i,:]
                j=j+1
            i=i+fold
        return img_new
    else:
        print("---constSizeBlur argument:fold out of range---")
        return


#########################################################################
## function: scale
## argument: image(numpy.ndarray), scale(float), interpolation mode(int)
## 将原图尺寸乘以scale，然后用选定的插值方法插值
#########################################################################
NNINTERPOLATION=1 # interpolation mode: nearest neighbor interpolation
BLINTERPOLATION=2 # interpolation mode: bilinear interpolation
def scale(img, scale, interMode):
    img_new=np.zeros([int(img.shape[0]*scale),int(img.shape[1]*scale),3],dtype=int)
    if interMode==NNINTERPOLATION:
        # use nearest neighbor interpolation method
        for i in range(img_new.shape[0]):
            for j in range(img_new.shape[1]):
                x=round(i/scale)
                y=round(j/scale)
                if x==img.shape[0]: # 为了防止下标越界
                    x=x-1
                if y==img.shape[1]:
                    y=y-1
                img_new[i,j,:]=img[x,y,:]
        return img_new
    elif interMode==BLINTERPOLATION:
        # use bilinear interpolation method
        for i in range(img_new.shape[0]):
            for j in range(img_new.shape[1]):
                x=i/scale
                y=j/scale
                if x>img.shape[0]-1: # 为了防止下标越界
                    x=img.shape[0]-1
                if y>img.shape[1]-1:
                    y=img.shape[1]-1
                if isOnPoint(x) and isOnPoint(y):
                    img_new[i,j,:]=img[int(x),int(y),:]
                elif isOnPoint(x):
                    img_new[i,j,:]=img[int(x),math.floor(y),:]*(math.ceil(y)-y)\
                              +img[int(x),math.ceil(y),:]*(y-math.floor(y))
                elif isOnPoint(y):
                    img_new[i,j,:]=img[math.floor(x),int(y),:]*(math.ceil(x)-x)\
                              +img[math.ceil(x),int(y),:]*(x-math.floor(x))
                else:
                    img_new[i,j,:]=img[math.floor(x),math.floor(y),:]*(math.ceil(x)-x)*(math.ceil(y)-y)\
                              +img[math.ceil(x),math.floor(y),:]*(x-math.floor(x))*(math.ceil(y)-y)\
                              +img[math.floor(x),math.ceil(y),:]*(math.ceil(x)-x)*(y-math.floor(y))\
                              +img[math.ceil(x),math.ceil(y),:]*(x-math.floor(x))*(y-math.floor(y))
        return img_new
    else:
        print("---scale argument:interMode invalid---")  
        return


######################################################
## subfunction: is on point?
## used by: funcion scale
## argument: x(float)
## 用来判断一个float是否足够接近整数，如1.0、2.0
######################################################
def isOnPoint(x):
    if (x-math.floor(x))<0.0001:
        return 1
    else:
        return 0


#####################################################
## function: segment image
## argument: img(numpy.ndarray), segMode(int)
## 用来分割图像并且生成掩膜
#####################################################
BINSEG=1
CLASSICSEG=2
MYSEG=3
def seg(img,segMode): 
    if segMode==BINSEG:
        # 分割二值化图像
        imgConnectivity=np.zeros([img.shape[0],img.shape[1]],dtype=bool)
        isConnected(img,imgConnectivity)
        imgSegMat=np.zeros([img.shape[0],img.shape[1]],dtype=int)
        color=1 # 记录当前用来标记的颜色
        colorDict={} # 融合相邻连通域时用此字典
        linearSeg={} # 用来把颜色线性映射到0-255
        # 开始遍历img
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # 仅对连接点操作
                if imgConnectivity[i][j]:
                    if (i-1<0 or imgConnectivity[i-1][j]==0 or imgSegMat[i-1][j]==0)\
                    and (j-1<0 or imgConnectivity[i][j-1]==0 or imgSegMat[i][j-1]==0)\
                    and (i+1>=img.shape[0] or imgConnectivity[i+1][j]==0 or imgSegMat[i+1][j]==0)\
                    and (j+1>=img.shape[1] or imgConnectivity[i][j+1]==0 or imgSegMat[i][j+1]==0)\
                    and (i-1<0 or j-1<0 or imgConnectivity[i-1][j-1]==0 or imgSegMat[i-1][j-1]==0)\
                    and (i-1<0 or j+1>=img.shape[1] or imgConnectivity[i-1][j+1]==0 or imgSegMat[i-1][j+1]==0)\
                    and (i+1>=img.shape[0] or j-1<0 or imgConnectivity[i+1][j-1]==0 or imgSegMat[i+1][j-1]==0)\
                    and (i+1>=img.shape[0] or j+1>=img.shape[1] or imgConnectivity[i+1][j+1]==0 or imgSegMat[i+1][j+1]==0):
                        # 若一个连接点周围所有的连接点都没有上色，那么给他一个新颜色
                        imgSegMat[i][j]=color
                        colorDict[color]=color
                        color=color+1
                    else:
                        # 否则，查看周围所有的连接点的颜色，选择其中序号最小的颜色
                        colorToUse=[]
                        minColor=999
                        for m in range(i-1,i+2):
                            for n in range(j-1,j+2):
                                if m>=0 and n>=0 and m<img.shape[0] and n<img.shape[1]:
                                    if imgConnectivity[m][n] and imgSegMat[m][n]:
                                        colorToUse.append(imgSegMat[m][n])
                        for s in range(0,len(colorToUse)):
                            if colorToUse[s]<minColor:
                                minColor=colorToUse[s]
                        for m in range(i-1,i+2):
                            for n in range(j-1,j+2):
                                if m>=0 and n>=0 and m<img.shape[0] and n<img.shape[1]:
                                    if imgConnectivity[m][n] and imgSegMat[m][n]:
                                        colorDict[imgSegMat[m][n]]=minColor
                        imgSegMat[i][j]=minColor
        # print(colorDict) ############################################################################### 检查点
        # 再次遍历修改部分标签，以融合相邻的连通域
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if imgSegMat[i][j]:
                    old=0
                    while old!=imgSegMat[i][j]:
                        old=imgSegMat[i][j]
                        imgSegMat[i][j]=colorDict[imgSegMat[i][j]]
                    linearSeg[imgSegMat[i][j]]=0
        # 将颜色线性映射到 0-255 之间，颜色数量大于 255 时不生效
        if len(linearSeg)<256:
            step=255//len(linearSeg)
            intensity=0
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if imgSegMat[i][j]:
                        if linearSeg[imgSegMat[i][j]]:
                            imgSegMat[i][j]=linearSeg[imgSegMat[i][j]]
                        else:
                            intensity=intensity+step
                            linearSeg[imgSegMat[i][j]]=intensity
                            imgSegMat[i][j]=intensity
        return imgSegMat
    elif segMode==CLASSICSEG:
        # 分割彩色图像
        # 这里采用课上方法，将图像变成二值化图像，这里用像素点平均亮度作为分界线
        avg=np.sum(img)/(img.shape[0]*img.shape[1]*img.shape[2])
        img_bin=(img[:,:,0]/3+img[:,:,1]/3+img[:,:,2]/3)>avg
        return seg(img_bin,BINSEG)
    elif segMode==MYSEG:
        # 我自己的分割方法
        imgConnectivity=np.zeros([img.shape[0],img.shape[1]],dtype=bool)
        isConnected(img,imgConnectivity)
        imgSegMat=np.zeros([img.shape[0],img.shape[1]],dtype=int)
        color=1
        # 开始遍历img
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # 仅对未访问的连接点操作
                if imgConnectivity[i][j] and imgSegMat[i][j]==0:
                    # 调用paintRec()递归地上色
                    paintRec(imgSegMat,i,j,color,imgConnectivity)
                    color=color+1
        return imgSegMat
    else:
        print("---seg argument:segMode invalid---")
        return


######################################################################
## subfunction: is connected?
## used by: function: segment image
## argument: img(numpy.ndarray), imgConnectivity(numpy.ndarray)
## 填充判断连接性的矩阵，这个函数会改变imgConnectivity
######################################################################
def isConnected(img,imgConnectivity):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 这里采用课上判断方法，只区分前景和背景，所有前景皆可视为连接点
            if img[i][j]:
                imgConnectivity[i][j]=1
    return
                

#######################################################################################################
## subfunction: paint recursively
## used by: funcion: segment image
## argument: imgSegMat(numpy.ndarray), i(int), j(int), color(int), imgConnectivity(numpy.ndarray)
## 为(i,j)上色后，调用自身为周围未访问的相邻连接点上色
#######################################################################################################
def paintRec(imgSegMat,i,j,color,imgConnectivity):
    imgSegMat[i][j]=color
    for m in range(i-1,i+2):
        for n in range(j-1,j+2):
            if m>=0 and n>=0 and m<imgSegMat.shape[0] and n<imgSegMat.shape[1]:
                if imgConnectivity[m][n] and imgSegMat[m][n]==0:
                    paintRec(imgSegMat,m,n,color,imgConnectivity)
    return
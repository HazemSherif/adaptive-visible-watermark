import PIL
from PIL import Image
from cv2 import cv2
from PIL import ImageFilter
from PIL import ImageEnhance
import numpy as np  
import array 
from math import log10, sqrt 
import matplotlib.pyplot as plt


def PSNR(original, compressed): 
    original = original.astype(np.float64) / 255.
    compressed = compressed.astype(np.float64) / 255.
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

# padding the watermark
# read logo
img1 = cv2.imread('logo2.png')
ht, wd, cc= img1.shape
img2 = cv2.imread("lena.jpg")
h_img,w_img,_ = img2.shape
# create new image of desired size and color (white) for padding
ww = w_img
hh = h_img
color = (255,255,255)
result = np.full((hh,ww,cc), color, dtype=np.uint8)
# compute center offset
xx = (ww - wd) // 2
yy = (hh - ht) // 2
# copy img image into center of result image
result[yy:yy+ht, xx:xx+wd] = img1

# view result
#cv2.imshow("result", result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# save result
cv2.imwrite("mylogo_padded.jpg", result)

# watermark image
img = Image.open("lena.jpg")
imgArr = np.array(img)
logo = Image.open("mylogo_padded.jpg")
logoArr = np.array(logo)
gray = img.convert("L")
grayImgArr =np.array(gray)
#contrast and entropy arrays
weber = np.array(gray).astype(float)
H = np.array(gray).astype(float)
# equations variables 
J = np.array(gray).astype(float)
alpha = np.array(gray).astype(float)
beta = np.array(gray).astype(float)
#final image array
final_image = np.array(img)
#calculate pixels contrsat value
def contrastSensitivity(image):
    print("calculating contrast senstivity.......")
    output_image = image.convert("L")
    for x in range(output_image.width):
        for y in range(output_image.height):
           grayImgArr[y][x] = output_image.getpixel((x,y))

           # change J for weber later !!!!
           weber[y][x] = np.float(abs(grayImgArr[y][x]-128) / 128) 
    #print(J)
    
#calculate the entropy of each pixel with 4*4 neighbourhood
def entropy(signal):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig=signal.size
        symset=list(set(signal))
        numsym=len(symset)
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent

def entropy_H():
    print("calculating entropy.......")
    N= 2
    S= grayImgArr.shape
    for row in range(S[0]):
        for col in range(S[1]):
                Lx=np.max([0,col-N])
                Ux=np.min([S[1],col+N])
                Ly=np.max([0,row-N])
                Uy=np.min([S[0],row+N])
                region=grayImgArr[Ly:Uy,Lx:Ux].flatten()
                H[row,col]=entropy(region)
                
# calculate the visual factor from all the previous variables
def visualFactor ():
    print("calculating visual factor.......")
    for x in range(img.width):
        for y in range(img.height):
            J[y][x] = weber[y][x] * H[y][x]
            
def embeddingWatermark(a,b,c,d):
    print("embedding watermark.......")
    for x in range(img.width):
        for y in range(img.height):
            alpha[y][x] = (b-a) * (J[y][x] - np.amin(J)) / (np.amax(J) - np.amin(J)) + a
            beta [y][x] = (d-c) * (J[y][x] - np.amin(J)) / (np.amax(J) - np.amin(J)) + c
            final_image[y][x] = alpha[y][x] * imgArr[y][x] + beta[y][x] * logoArr[y][x]
            if  y == img.height/4 and x == img.width/4 :
                print("25%...")
            if y == img.height/2 and x == img.width/2  :
                print("50%...")
            if y == img.height/2 + img.height/4  and x == img.width/2 + img.width/4 :
                print("75%...")

    final = Image.fromarray(final_image)
    #final.show()
    file_name = ("a; "+str(a) + "b; "+str(b) + "c; "+str(c) + "d; "+ str(d)+ ".jpg")
    final.save(file_name)
    
    for x in range(img.width):
        for y in range(img.height):
            final_image[y][x] = 0.8 * imgArr[y][x] + 0.2 * logoArr[y][x]
    not_adaptive = Image.fromarray(final_image)
    #not_adaptive.show()
    not_adaptive.save("not_adaptive_1.jpg")
    psnr1 =  cv2.imread(file_name)
    psnr2 =  cv2.imread('lena.jpg')
    value = PSNR(psnr1, psnr2) 
    print(value)

contrastSensitivity(img)
entropy_H()
visualFactor()
embeddingWatermark (0.6999999999999998,0.8,0.24999999999999994,0.1160)

def plot_v():
    a = 0.9
    b = 0.8
    c = 0.05
    d = 0.1160
    x1 = []
    #x2 = []
    x3 = []
    #x4 = []
    x5 = []

    for i in range(25):
        embeddingWatermark (a,b,c,d)
        file_name = ("a; "+str(a) + "b; "+str(b) + "c; "+str(c) + "d; "+ str(d)+ ".jpg")
        psnr1 =  cv2.imread(file_name)
        psnr2 =  cv2.imread('lena.jpg')
        value = PSNR(psnr1, psnr2) 
        a -= 0.02
        #b += 0.02
        c += 0.02
        #d += 0.02
        x1.append(a)
        #x2.append(b)
        x3.append(c)
        #x4.append(d)
        x5.append(value)
        print(i)
    plt.plot(x1, x5, label = "image")
    plt.plot(x3, x5, label = "watermark")
    plt.xlabel('embedding factor')
    plt.ylabel('PSNR')
    plt.title('PSNR to embedding factors change')
    plt.legend()
    plt.show()
    a = 0.9
    b = 0.8
    c = 0.05
    d = 0.1160
    for i in range(25):
        embeddingWatermark (a,b,c,d)
        file_name = ("a; "+str(a) + "b; "+str(b) + "c; "+str(c) + "d; "+ str(d)+ ".jpg")
        psnr1 =  cv2.imread(file_name)
        psnr2 =  cv2.imread('lena.jpg')
        value = PSNR(psnr1, psnr2) 
        a += 0.02
        #b += 0.02
        c -= 0.02
        #d += 0.02
        x1.append(a)
        #x2.append(b)
        x3.append(c)
        #x4.append(d)
        x5.append(value)
        print(i)

    plt.plot(x1, x5, label = "image")
    #plt.plot(x2, x5, label = "image")
    plt.plot(x3, x5, label = "watermark")
    #plt.plot(x4, x5, label = "watermark")
    plt.xlabel('embedding factor')
    plt.ylabel('PSNR')
    plt.title('PSNR to embedding factors change')
    plt.legend()
    plt.show()
        
#plot_v()
cv2.waitKey(0)
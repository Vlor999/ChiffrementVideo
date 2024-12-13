#! /usr/bin/python3.8
#test 
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
# correlation et entropie 

image = cv2.imread('../../image/ImageCree/output_chiffré.jpeg')
image_2 = cv2.imread('../../image/ImageCree/output.jpeg')

def creation_T(img):
    Nl, Nc, _ = np.shape(img)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    T = 0.33*R + 0.33*G + 0.34*B
    size = Nl * Nc
    return T, size, Nl, Nc

T, size, Nl, Nc = creation_T(image)
#1 corrélation :

def col_correlation(pixel, C1):
    ex = np.sum(pixel[:,C1]) / Nc # espérence
    ey = np.sum(pixel[:,C1 + 1]) / Nc # idem

    dx = np.sum((pixel[:,C1]-ex) ** 2) / Nc 
    dy = np.sum((pixel[:,C1 + 1]-ex) ** 2) / Nc

    cov = np.sum((pixel[:,C1]-ex) * (pixel[:,C1 + 1]-ey)) / Nc # covariance

    r = 0
    if dx*dy > 0 :
        r = cov / math.sqrt(dx * dy)
    return r

def lig_correlation(pixel, L1):
    ex = np.sum(pixel[L1, :]) / Nl
    ey = np.sum(pixel[L1 + 1, :]) / Nl

    dx = np.sum((pixel[L1, :]-ex) ** 2) / Nl
    dy = np.sum((pixel[L1 + 1, :]-ex) ** 2) / Nl

    cov = np.sum((pixel[L1, :]-ex) * (pixel[L1 + 1, :]-ey)) / Nl

    r = 0
    if dx * dy > 0 :
        r = cov / math.sqrt(dx * dy)
    return r

def cor_image(T):
    M, N = 0, 0

    Nc_1, Nl_1 = Nc - 1, Nl - 1

    for i in range(Nc_1):
        M += col_correlation(T,i)
    for i in range(Nl_1):
        N += lig_correlation(T,i)

    moy_c = M / (Nc_1)
    moy_l = N / (Nl_1)

    cor_col_moyenne = np.round(moy_c, 6) #  correlation moyenne 
    cor_lig_moyenne = np.round(moy_l, 6)
    
    return cor_col_moyenne, cor_lig_moyenne

#2 entropie 

def entropie(img):

    liste = np.zeros(256) 


    for l in range(Nl):
        for c in range(Nc):
            color_lc = img[l, c, 0]
            liste[color_lc] += 1
    entropie = 0

    liste = liste/size
    L1 = np.log2(liste)

    L2 = liste * L1
    L2 = -1 * np.sum(L2)

    entropie = np.round(L2, 6)

    return entropie

#3 NPCR

def compare_pix(P1, P2):
    R1, G1, B1 = P1[0], P1[1], P1[2]
    R2, G2, B2 = P2[0], P2[1], P2[2]
    if R1 == R2 and G1 == G2 and B1 == B2:
        return True 
    else :
        return False


def NPCR_and_UACI(img1, img2):
    NPCR = 0
    for l in range(Nl):
        for c in range(Nc):
            P1, P2 = img1[l,c], img2[l,c]
            if not compare_pix(P1,P2):
                NPCR += 1
    NPCR = (NPCR / size) * 100

    R1, G1, B1 = img1[:,:,0], img1[:,:,1], img1[:,:,2]
    T1 = R1 * 0.33 + G1 * 0.33 + B1 * 0.34
    R2, G2, B2 = img2[:,:,0], img2[:,:,1], img2[:,:,2]
    T2 = R2 * 0.33 + G2 * 0.33 + B2 * 0.34

    UACI = np.abs(np.sum(T2 - T1)/(255*size))*100

    return NPCR, UACI


def histogramme(img, titre) :
    dictionnaire = {i: 0 for i in range(256)}
    
    Nl, Nc, _ = np.shape(img)
    
    for l in range(Nl) :
        for c in range(Nc) :
            R, _, _ = img[l,c]
            dictionnaire[R] += 1
    
    plt.bar(list(dictionnaire.keys()), dictionnaire.values(), color='g')
    plt.xlabel('valeur du pixel')
    plt.ylabel("nombre d'occurence")
    plt.title(titre)
    # plt.savefig(titre)
    plt.show()
    
    

correlation_image_c, correlation_image_l = cor_image(T)   
val_entropie = entropie(image_2)
val_NPCR, val_UACI = NPCR_and_UACI(image,image_2)
histogramme(image, 'image chiffrée')
histogramme(image_2, 'image claire')

print(f"correlation de l'image, colonne : {correlation_image_c} \ncorrelation de l'image, ligne : ", correlation_image_l)
print(f'Entropie: {val_entropie}')
print(f'NPCR : {val_NPCR}')
print(f'UACI : {val_UACI}')


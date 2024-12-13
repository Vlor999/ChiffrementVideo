# Integration fontion

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('/Users/willem/Desktop/test python video /image TIPE/output.jpeg')
Nl, Nc = image.shape[0: 2]
image = image[:, :, 0]
liste_alea = np.random.randint(1, 100, Nc)

def vecteur_alpha(n):
    V = np.array([np.random.randint(0,255, n)])
    V = V.transpose()
    return V

def somme_alpha_beta(vecteur1, vecteur2, b):
    return b + int(vecteur2 @ vecteur1)

def new_beta(vecteur, valeur):
    V1 = vecteur[0,:-1]
    vecteur = np.matrix(np.append([valeur], V1))
    return vecteur

def somme_alpha_b_prime(liste_alpha, bprime, b, n):
    somme = 0
    for i in range(len(liste_alpha)):
        somme = somme + liste_alpha[i] * bprime[n-i]
    bprime = b + somme
    return bprime

def transfo(vect):
    _, n = vect.shape
    vect1 = np.matrix([vect[0, n-i] for i in range(1,n+1)])
    return vect1

def transfo_vigenere(image):
    image_0 = np.ones((Nl,Nc), dtype = 'uint8')
    N = 15
    vect_alpha = vecteur_alpha(N)
    
    vecteur_beta = np.transpose(np.matrix([vect_alpha[i] * liste_alea[i] for i in range(N-1, -1, -1)]))
    
    for l in range(Nl):
        for c in range(Nc):
            b = image[l, c]
            bprime = somme_alpha_beta(vect_alpha, vecteur_beta, b)
            vecteur_beta = new_beta(vecteur_beta, bprime)
            image_0[l, c] = bprime
    return image_0



image_0 = transfo_vigenere(image)

# plt.imshow(image_0)

# cv2.imwrite('/Users/willem/Desktop/Image-final.jpg', image_0)
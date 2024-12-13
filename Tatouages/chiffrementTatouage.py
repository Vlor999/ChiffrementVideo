##cryptage par tatouage par codage linéaire et RSA


#import de module:

import random
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt




#code nécessairepour l'import d'une photo quelconque
#pour utiliser une autre image il faut adapter le lien
image=plt.imread('/Users/willem/Desktop/TIPE/image/cerisier.jpeg')
image=image[:,:,:3]
Nl,Nc=image.shape[0:2]

#fontion nécessaire pour l'affichage d'une image
def affiche(im):
    plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.show()
    plt.pause(0.0001)

#je vais avoir besoin de décomposer un entier 'n' en une somme de carré afin de créer la matrice
#householder
def decompo_carre(n):
    L = []
    while n > 1:
        i = int(sqrt(n))+1 #je sais que n'admet pas de carré plus grands que 'i'
        while i**2 > n:
            i -= 1 # je traite l'ensemble des carrées disponible inférieur à 'n'
        L.append(i)
        n -= i**2 #je retravaille avec un n sans la valeur considérée précédement
    if n == 1:
        L.append(1)
    return L

#je vais avoir besoin de trouver l'inverse de la matrice Householder.j'utilise la fonction
#native de python pour avoir des résultats plus rapide
def inverse_H(H):
    inv_H = np.linalg.inv(H) #fonction native de numpy
    return inv_H

#ici j'ai la focntion du produit de matrice natif de python afin d'obtenir des résultats 
#rapide 
def prod_mat(A,B):
    D = A.dot(B)
    return D

#je vais ici pouvoir initialiser la matrice Householder qui me permettra de crypter 
# la matrice image

#vecteur privée de la matrice H
def vecteur_privee(n):
    V_1 = np.zeros(shape=(n,1)) # sera le vecteur secret de la matrice
    L = decompo_carre(n+1)
    elem = [random.randint(1,i)*n for i in range(1,n-len(L)+1)]
    L += elem
    for i in range(len(L)):
        val=L[i]
        V_1[i] = ((-1)**i)*val
    return V_1

#ici ce sera le vecteur transmis a tout le monde via la clé publique


#la matrice H est ici créée
def matrice_Householder(n):
    id = np.eye(n) #nécessaire pour la matrice Householder
    V_1=vecteur_privee(n)
    V_2=np.transpose(V_1) #correspond à la transposée de V_1
    VV = prod_mat(V_1,V_2)
    matrice_H = id-2*VV #j'obtiens une matrice Householder
    inve_H=inverse_H(matrice_H) #je récupère alors son inverse
    return matrice_H,inve_H


##fonction nécessaire pour l'affichage et la transformation de l'image 


#j'ai remarqué que l'inverse d'une matrice était assez approximatif. ce qui est problématique pour
# la récupération d'image. Je vais alors récupérer la valeur la plus proche de x entre 
# int(x) et int(x)+1 
def val_app(a):
    val_inf=int(a)
    val_sup=int(a)+1
    if val_sup-a>0.5:
        return val_inf
    return val_sup


#via la Matrice householder j'btiens des entier qui dépassent les 255 or l'image n'est codé
#que sur 255 bits. je reconvertie alors cette valeur dans une base de 255 valeurs.
def convert_base_255(M):
    Nl,Nc=M.shape[0:2]
    for l in range(Nl):
        for c in range(Nc):
            terme=M[l,c]%256 #chaque terme reviens dans la base 255 avec le reste de la DE
            terme=val_app(terme) #les valeurs sont cependants parfois approchée 
            M[l,c]=terme  #la matrice est alors dans la base de cryptage
    return M    

#ici je fais à la fois le produit de matrice et la conversion en base 255
def prod_conv(A,B):
    D=prod_mat(A,B)
    D=convert_base_255(D)#utilisation des deux codes précédents
    return D

#j'utilise le produit de la matrice Householder et de l'image (matrice R,G et B)
#l'objet crypte qui est renvoyé est ce qui sera envoyé à un individu ou stocké dans la caméra.
#l'objet im est directement l'image dans la base 255. ceci est plus pratique pour l'afficher.

##tranformation de l'image claire d'image

def transfo_image(im,cle):
    im=np.copy(im) #je garde une copy de l'image initiale
    (H,_)=cle #récupération de H
    R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]
    R1,G1,B1,=prod_mat(H,R),prod_mat(H,G),prod_mat(H,B) #image cryptée
    crypte=R1,G1,B1 #sera envoyé ou conservée par la caméra
    R,G,B=prod_conv(H,R),prod_conv(H,G),prod_conv(H,B)
    im[:,:,0],im[:,:,1],im[:,:,2]=R,G,B #image cryptée avec des non sens total
    return crypte,im



##récupération d'image

#via la donnée cryptée je peux avec la matrice inverse à la matrice Householder retrouver les 
#valeurs initiales de R,G,B
def recup_image(crypte,cle):
    im=np.zeros((Nl,Nc,3),dtype='uint8') # image vierge
    R,G,B=crypte # récupération des valeurs cryptées
    (_,H_inv)=cle # récupération de la matrice H_inv
    R1,G1,B1=prod_conv(H_inv,R),prod_conv(H_inv,G),prod_conv(H_inv,B) #je fais ici le chemin inverse
    im[:,:,0],im[:,:,1],im[:,:,2]=R1,G1,B1 #puis je transpose tout dans l'image initialement vierge
    return im


cle=matrice_Householder(Nl)
crypte,im_1=transfo_image(image,cle)
im_2=recup_image(crypte,cle)

affiche(image) #première image
affiche(im_1) #image cryptée
affiche(im_2) #image décryptée





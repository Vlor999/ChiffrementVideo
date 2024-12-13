#! /usr/bin/env python3.8
import numpy as np
from math import sqrt
import random

###                         fonction pour le code RSA
##indicatrice d'euler
def id_eul(p, q):
    return (p-1) * (q-1) #on a ce calcul car p et q sont premiers

## algorithme d'euclide

#explication: 
#le code utilise le principe suivant : PGCD(a,b)=PGCD(b,r), avec r le reste de la division 
#euclidienne de b par a. On détermine ainsi le PGCD de a,b. Cet algorithme est nécessaire pour 
#trouver l'ensemble de nombres premiers avec a.
def algo_euclide(a, b):
    if b == 0:
        return a
    else:
        return algo_euclide(b, a%b)

##calculs des nombres premiers

#explication:
#on calcul l'ensemble des nombres premiers inférieur (strictement) au nombre étudier.
#ceci est nécessaire pour trouver l'exposant de chiffrement du code RSA.
def premier(a, i=1):
    L = []
    while i < a and len(L) < 10000:
        if algo_euclide(a,i) == 1 :
            L.append(i)  #si le PGCD entre a et i est égale à 1 alors ils sont premiers entre eux  
        i += 1
    return L

##agorithme d'euclide étendu

#explication:
#ce code permet de retrouver des coefficients de la relation de Bézouts.
#on trouve un couple solution. Néanmoins pour la suite des calculs il me faut
#un 'v' positif donc avec la solution particulière je trouve la solution générale.
def div_euclid_etendu(a, b):
    if b == 0:
        return (a, 1, 0)
    else:
        (d,u,v) = div_euclid_etendu(b, a%b)
        return (d, v, u-((a//b)*v))  

def div_eucl_pos(a, b):
    (d,u,v) = div_euclid_etendu(a, b) #PGCD est constant donc n'est pas utilie ici
    while v < 0:
        v += a
        u -= b
    return (d,u,v)


###                                   code RSA
def RSA(p, q):
    n = p * q   #création du module de chiffrement ave p,q premiers
    phi = id_eul(p,q)     #indicatrice d'euler
    expo = random.choice(premier(phi))   #récupère un nombre premier avec phi de façons arbitraire, qui est lexposant de chiffrement
    _,_,dech = div_eucl_pos(phi,expo)  #grâce à ce code, je créer l'exposant de déchiffrement 
    cle_pub = n,expo    #je peux alors créer la clé privée et publique
    cle_priv = n,dech
    return cle_pub,cle_priv

#je vais avoir besoin de décomposer un entier 'n' en une somme de carré afin de créer la matrice
#householder
def decompo_carre(n):
    L = []
    while n > 1:
        i = int(sqrt(n)) + 1 #je sais que n n'admet pas de carré plus grands que 'i'
        while i**2 > n:
            i -= 1 # je traite l'ensemble des carrées disponible inférieur à 'n'
        L.append(i)
        n -= i ** 2 #je retravaille avec un n sans la valeur considérée précédement
    if n == 1:
        L.append(1)
    return L


#je vais avoir besoin de trouver l'inverse de la matrice Householder.j'utilise la fonction
#native de python pour avoir des résultats plus rapide
def inverse_H(H):
    inv_H = np.linalg.inv(H) #fonction native de numpy
    return inv_H


#je vais ici pouvoir initialiser la matrice Householder qui me permettra de crypter 
#la matrice image

#vecteur privée de la matrice H

def convert_base_255(A, B):
    Nl, Nc = A.shape[0:2]
    D = A.dot(B)
    D = np.matrix.round(D)
    C = np.zeros((Nl, Nc), dtype = 'uint8')
    for l in range(Nl):
        C[l,:] = D[l,:] #chaque terme reviens dans la base 255 avec le reste de la DE
    return C, D


#edition du code version 2

def chiffrement(M, cle_publ):
    M = int(M)
    n, expo = cle_publ
    C = (M**expo) % n     #correspond au reste de la div euclidienne de M puissance expo par n
    return C

def chiff_vect(V, cle):
    Nl, Nc = V.shape[0:2]
    for l in range(Nl) :
        for c in range(Nc):
            elem = V[l, c]
            elem = chiffrement(elem, cle)
            V[l,c] = elem
    return V

def vecteur_privee_2(n, cle):
    V_1 = np.zeros((1,n)) # sera le vecteur secret de la matrice
    L = decompo_carre(n+1)
    elem = [random.choice(L) for _ in range(1,n-len(L)+1)]
    L += elem
    for i in range(n):
        val = L[i]
        V_1[0, i] = val
    V_1 = chiff_vect(V_1, cle)
    return V_1

def dechiffrement(C, cle_priv):
    C = int(C)
    n, dech = cle_priv
    val = C ** dech
    return val % n

def dechi_vect(V, cle):
    Nl, Nc = V.shape[0:2]
    V = np.round(V)
    for l in range(Nl) :
        for c in range(Nc):
            elem = V[l, c]
            elem = dechiffrement(elem, cle)
            V[l,c] = elem
    return V

def matrice_Householder_2(V, cle):
    _, n = V.shape[0:2]
    id = np.eye(n) #nécessaire pour la matrice Householder
    V_1 = dechi_vect(V,cle)
    V_2 = np.transpose(V_1) #correspond à la transposée de V_1
    VV = V_2.dot(V_1)
    matrice_H = id-2*VV #j'obtiens une matrice Householder
    inve_H = inverse_H(matrice_H) #je récupère alors son inverse
    return matrice_H,inve_H 

def transfo(im,cle):
    im1 = np.copy(im)
    H, _ = cle
    R, G, B = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2]
    return list(map(lambda x: x.dot(H),[R,G,B]))

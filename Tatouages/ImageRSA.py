##cryptage par tatouage par codage linéaire et RSA

#texte codée par RSA
texte = """cryptage d'un message via le code RSA"""

#RSA
#valeur:
p = 43
q = 47
#import de module:

import random
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import time

##je produit ici l'ensemble des fonctions nécessaire pour empêcher l'analyse des fréquences
#je me base sur le texte d'Hamlet en francais
#il est cependant possible d'utiliser n'importe quel texte
#si vous voulez essayer le code il faut ajuster le chemin
#chemin du texte:
#https://www.dropbox.com/s/ehtw0u2ipwkxmk9/3-1%20-%20TD%20-%20Lecture%20dans%20un%20fichier%20-%20Hamlet%20français.txt?dl=0
titre = '/Users/willem/Desktop/informatique/3 chapitre/python_fichier/texte francais /Hamlet français.txt'

###                     fonction pour le texte auxiliaire
#je récupère l'ensemble des lignes du texte et les convertie en texte
def recup_donnee(titre):
    fichier = open(titre,'r')
    ensemble_ligne = ''
    for lignes in fichier:
        lignes = lignes.strip()
        if lignes != '':
            ensemble_ligne += lignes #je forme un texte qui est le texte étudié
    fichier.close()
    return ensemble_ligne

#je récupère le texte et le transforme en une chaine de chiffre 
def transfo_texte(texte):
    texte = str(texte)
    trans = []
    for terme in texte:
        trans.append(ord(terme))
    return trans

#je prend le texte auxiliaire et je transforme chaque caractère en chiffre
def para_texte(titre):
    texte_aux = recup_donnee(titre)
    par_texte = transfo_texte(texte_aux)
    return par_texte



###                         fonction pour le code RSA
##indicatrice d'euler
def id_eul(p,q):
    return (p-1)*(q-1) #on a ce calcul car p et q sont premiers

## algorithme d'euclide

#explication: 
#le code utilise le principe suivant : PGCD(a,b)=PGCD(b,r), avec r le reste de la division 
#euclidienne de b par a. On détermine ainsi le PGCD de a,b. Cet algorithme est nécessaire pour 
#trouver l'ensemble de nombres premiers avec a.
def algo_euclide(a,b):
    if b == 0:
        return a
    else:
        return algo_euclide(b,a%b)

##calculs des nombres premiers

#explication:
#on calcul l'ensemble des nombres premiers inférieur (strictement) au nombre étudier.
#ceci est nécessaire pour trouver l'exposant de chiffrement du code RSA.
def premier(a,i=1):
    L = []
    while i<a and len(L)<10000:
        if algo_euclide(a,i) == 1 :
            L.append(i)  #si le PGCD entre a et i est égale à 1 alors ils sont premiers entre eux  
        i += 1
    return L

##agorithme d'euclide étendu

#explication:
#ce code permet de retrouver des coefficients de la relation de Bézouts.
#on trouve un couple solution. Néanmoins pour la suite des calculs il me faut
#un 'v' positif donc avec la solution particulière je trouve la solution générale.
def div_euclid_etendu(a,b):
    if b == 0:
        return (a,1,0)
    else:
        (d,u,v) = div_euclid_etendu(b,a%b)
        return (d,v,u-((a//b)*v))  

def div_eucl_pos(a,b):
    (_,u,v) = div_euclid_etendu(a,b) #PGCD est constant donc n'est pas utilie ici
    while v < 0:
        v += a
        u -= b
    return (_,u,v)



###                                   code RSA
def RSA(p,q):
    n = p*q   #création du module de chiffrement ave p,q premiers
    phi = id_eul(p,q)     #indicatrice d'euler
    expo = random.choice(premier(phi))   #récupère un nombre premier avec phi de façons arbitraire
    (_,_,dech) = div_eucl_pos(phi,expo)  #grâce à ce code, je créer l'exposant de déchiffrement 
    cle_pub = (n,expo)    #je peux alors créer la clé privée et publique
    cle_priv = (n,dech)
    return cle_pub,cle_priv

##bilan des fonctions précédentes:
#l'ensemble des fonctions au dessus du code RSA permettent d'avoir un code assez simple.
#j'arrive à avoir la clé publique et privé. je dois maintenant pouvoir les utiliser.
#il est donc nécessaire de pourvoir créer un code qui va crypter le message clair.
#et un autre qui permettra de lire le message.

##1ére étape:
#la première étape consiste a transformer une suite de caractère (chiffres ou lettres) via la clé 
#publique cette clé peut être utilisée par n'importe qui. On crypte les données afin de ne laiser
#que le créateur voir le message

##                                  chiffrement

#explication:
#le chiffrement consiste à trouver la valeur C tel que C est congru à M puissance expo
#(l'exposant de chiffrement) modulo n (le module de chiffrement). Ces valeurs sont trouvables sur
#la clé publique. On trouvera la valeur de C en trouvant le reste de la division euclidienne de 
#M puissance expo par n.
def chiffrement(M,cle_publ):
    M = int(M)
    n,expo = cle_publ
    C = (M**expo)%n     #correspond au reste de la div euclidienne de M puissance expo par n
    return C

#je fais ici le cryptage pouyr une liste de valeurs ce qui sera utilile pour la clé secrète 
#de la matrice Householder

def chiffrement_liste(L,cle_pub):
    crypte = []
    for elem in L:
        crypte.append(chiffrement(elem,cle_pub))
    return crypte

##2nd étape:
#cette seconde étape n'est réalisable que par le créateur de la clé privée. elle lui permet de 
#déchiffrer les caractères. 

##                                     déchiffrement

#explication:
#Grace a la clé privé nous connaissons l'inverse modulaire qui permet de de retrouver l'unique 
#antécédents de C (message crypté). Pour cela, on cherche le reste de la division euclidienne
#de C (caractère crypté) à la puissance dech (exposant de déchiffrement ou inverse modulaire)
#par n.

def dechiffrement(C,cle_priv):
    C = int(C)
    n,dech = cle_priv
    val = C**dech
    return val%n

def dechiffrement_liste(L,cle_priv):
    decrypte = []
    for elem in L:
        decrypte.append(dechiffrement(elem,cle_priv))
    return decrypte

## conclusion sur la création les codes de chiffrements et de déchiffrements:
#les codes sont utilisables et possèdent des temps de calculs assez faible. 
#Pour coder un message, il faut transformer les caractères sous forme de nombres. 
#De plus, on utilisera la méthode de vigenère pour empêcher l'analyse des fréquences. 


##pour empêcher l'analyse des fréquences je vais utiliser la méthodes de césaire/vigenère
#l'idée que Ai = Bi + Ci avec Ai la lettre envoyé Bi la lettre du texte à coder et Ci une 
#lettre quelconque d'un texte auxiliaire.

#je cripte le texte avec un texte auxiliaire 
def chiffrement_texte(texte,cle_restreinte,cle_publ):
    titre,prim_char = cle_restreinte 
    texte = transfo_texte(texte) #je transforme le texte dans une base python (en chiffre)
    par_texte = para_texte(titre) #idem pour le para texte
    texte_chif = ''
    i = prim_char
    for terme in texte:
        val = chiffrement(terme,cle_publ)+par_texte[i]#je chiffre la lettre puis rajoute une lettre
        texte_chif += chr(val)                        #qui correspond a l'indice de la lettre dans 
        i += 1                                        #le texte auxiliaire
    return texte_chif

#je décripte le texte codé via la clé privée et la clée réstreinte 
def dechiffrement_texte(texte_chif,cle_restreinte,cle_priv):
    titre,prim_char = cle_restreinte
    par_texte = para_texte(titre)
    texte = ''
    i = prim_char
    for terme in texte_chif:
        val = ord(terme) - par_texte[i] #ici je fais pareil mais dans le sens inverse 
        terme = dechiffrement(val,cle_priv)#pour le déchiffrement
        texte += chr(terme)
        i += 1
    return texte

##                              cryptage image

#il est possible que j'appelle la matrice Householder la matrice H

#code nécessaire pour l'import d'une photo quelconque
#pour utiliser une autre image il faut adapter le lien
image = plt.imread('/Users/willem/Desktop/informatique/3 chapitre/image/Resultat_3.jpg')
image = image[:,:,:3]
Nl, Nc = image.shape[0:2]

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

def chiff_vect(V,cle):
    Nl,Nc = V.shape[0:2]
    for l in range(Nl) :
        for c in range(Nc):
            elem = V[l,c]
            elem = chiffrement(elem,cle)
            V[l,c] = elem
    return V

#vecteur privée de la matrice H

def vecteur_privee(n,cle):
    V_1 = np.zeros(shape=(1,n)) # sera le vecteur secret de la matrice
    L = decompo_carre(n+1)
    elem = [random.choice(L) for _ in range(1,n-len(L)+1)]
    L += elem
    for i in range(n):
        val = L[i]
        V_1[0,i] = val  #matrice ligne
    V_1 = chiff_vect(V_1,cle)
    return V_1

#ici ce sera le vecteur transmis a tout le monde via la clé publique

#ceci sera le vecteur déchiffré du vecteur envoyé 
def dechi_vect(V,cle):
    Nl, Nc = V.shape[0:2]
    for l in range(Nl) :
        for c in range(Nc):
            elem = V[l,c]
            elem = dechiffrement(elem,cle)
            V[l,c] = elem
    return V

#la matrice H est ici créée

def matrice_Householder(V, cle):
    _,n = V.shape[0:2]
    id = np.eye(n) #nécessaire pour la matrice Householder
    V_1 = dechi_vect(V,cle)
    V_2 = np.transpose(V_1) #correspond à la transposée de V_1
    VV = prod_mat(V_2,V_1)
    matrice_H = id-2*VV #j'obtiens une matrice Householder
    inve_H = inverse_H(matrice_H) #je récupère alors son inverse
    return matrice_H,inve_H 

##fonction nécessaire pour l'affichage et la transformation de l'image 


#j'ai remarqué que l'inverse d'une matrice était assez approximatif. ce qui est problématique pour
# la récupération d'image. Je vais alors récupérer la valeur la plus proche de x entre 
# int(x) et int(x)+1 
def val_app(a):
    val_inf = int(a)
    if a-val_inf < 0.5 :
        return val_inf
    return val_inf+1

def val_l(l):
    l = [val_app(i) for i in l]
    return l

#via la Matrice householder j'obtiens des entier qui dépassent les 255 or l'image n'est codé
#que sur 255 bits. je reconvertie alors cette valeur dans une base de 255 valeurs.
def convert_base_255(M):
    Nl,Nc = M.shape[0:2]
    A = np.zeros((Nl,Nc))
    for l in range(Nl):
        ligne = M[l,:] #chaque terme reviens dans la base 255 avec le reste de la DE
        ligne = ligne % 256
        terme = val_l(ligne) #les valeurs sont cependants parfois approchée 
        A[l,:] = terme  #la matrice est alors dans la base de cryptage
    return A    

def convert_base_255_rapide(M):
    Nl,Nc = M.shape[0:2]
    M = np.matrix.round(M) #rajouté le 11 juin permet d'obtenir un arrondie parfait
    A = np.ones((Nl,Nc),dtype='uint8')
    for l in range(Nl):
        A[l,:] = M[l,:] #chaque terme reviens dans la base 255 avec le reste de la DE
    return A   

#ici je fais à la fois le produit de matrice et la conversion en base 255
def prod_conv(A,B):
    D = prod_mat(A,B)
    D = convert_base_255_rapide(D)#utilisation des deux codes précédents
    return D

def prod_eff(M,N):
    C = prod_mat(M,N)
    A = convert_base_255_rapide(C)
    return A, C

#j'utilise le produit de la matrice Householder et de l'image (matrice R,G et B)
#l'objet crypte qui est renvoyé est ce qui sera envoyé à un individu ou stocké dans la caméra.
#l'objet im est directement l'image dans la base 255. ceci est plus pratique pour l'afficher.

##tranformation de l'image claire d'image

def transfo_image(im,cle):
    im1 = np.copy(im) #je garde une copy de l'image initiale
    H,_ = cle #récupération de H
    R,G,B = im1[:,:,0],im1[:,:,1],im1[:,:,2]
    R,R1 = prod_eff(R,H)
    G,G1 = prod_eff(G,H)
    B,B1 = prod_eff(B,H)
    crypte = R1,G1,B1 #sera envoyé ou conservée par la caméra
    im1[:,:,0], im1[:,:,1], im1[:,:,2] = R, G, B #image cryptée avec des non sens total
    return crypte, im1

##récupération d'image

#via la donnée cryptée je peux avec la matrice inverse à la matrice Householder retrouver les 
#valeurs initiales de R,G,B

def recup_image(crypte,cle):
    im = np.zeros((Nl,Nc,3),dtype='uint8') # image vierge
    R, G, B = crypte # récupération des valeurs cryptées
    _, H_inv = cle # récupération de la matrice H_inv
    R1, G1, B1 = prod_conv(R,H_inv), prod_conv(G,H_inv), prod_conv(B,H_inv) #je fais ici le chemin inverse
    im[:,:,0], im[:,:,1], im[:,:,2] = R1, G1, B1 #puis je transpose tout dans l'image initialement vierge
    return im

#je compare ici l'image initiale et l'image finale
def compare(im1,im2):
    i = 0
    Nl,Nc = im1.shape[0:2]
    image_b = 255 * np.ones((Nl,Nc,3), dtype = 'uint8')
    for l in range(Nl):
        for c in range(Nc):
            R, G, B = im1[l,c][0:3]
            R1, G1, B1 = im2[l,c][0:3]
            #je vais afficher une image blanche qui montre les problemes de pixels
            if R != R1 :
                image_b[l,c][0] = 0
                i += 1
            elif G != G1 :
                image_b[l,c][1] = 0
                i += 1
            elif B != B1:
                image_b[l,c][2] = 0
                i += 1
    return i, image_b

def compare_100(im1,im2):
    Nl, Nc = im1.shape[0:2]
    i, image_noir = compare(im1,im2)
    totale = Nl * Nc * 3
    ecart = 100 * (i/totale)
    return ecart, image_noir

def tout():

    prim_char = 500
    cle_restreinte = (titre,prim_char) #la clé restreinte correspond a un passage particulier d'un livre
    cle_pub,cle_priv = RSA(p,q)


    V=vecteur_privee(Nc,cle_pub)

    cle_2 = matrice_Householder(V,cle_priv)
    tic=time.time()
    crypte, im_1 = transfo_image(image,cle_2)
    tac=time.time()
    print(tac-tic)

    im_2 = recup_image(crypte,cle_2)

    val2, image_b = compare_100(image,im_2)
    
    affiche(image)

    affiche(im_1)

    affiche(im_2)

    affiche(image_b)

    print(val2)

    texte_1 = chiffrement_texte(texte, cle_restreinte, cle_pub) #texte crypté
    texte_2 = dechiffrement_texte(texte_1, cle_restreinte, cle_priv) #retour du texte originale
    print(texte, texte_1, texte_2)

tout()

### jai fais un changement dans le vecteur privé qui me reduit considérable l'erreur voir même
### l'annule je ne connais pas vraiment la raison.
#j'ai changer ma fonction de conversion pour être plus rapide ce qui marche. on remarque
#une amélioration, environ 30 fois plus rapide. Néanmoins on a une légère erreur, les floatants
#sont convertis en l'entier inférieur même ce n'est pas lui le plus proche. on ne voit pas la 
#différence sur l'image.

#le 11 juin j'ai trouvé la fonction matrix.round qui me permet d'obtenir les bons arrondies tout
#en gardant un temps de calcul assez faible.


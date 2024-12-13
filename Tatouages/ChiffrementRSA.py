##cryptographie par codage RSA
#valeur:
p=197
q=199
#import de module:

import random

##je créée ici l'ensemble des fonctions nécessaire pour empêcher l'analyse 
##des fréquences
#ouverture de fichier pour empêcher l'analyse des fréquences
#je me base sur le texte d'Hamlet en francais
#il est cependant possible d'utiliser n'importe quel texte
#si vous voulez essayer le code il faut ajuster le chemin
#lien du texte:
#https://www.dropbox.com/s/ehtw0u2ipwkxmk9/3-1%20-%20TD%20-%20Lecture%20dans%20un%20fichier%20-%20Hamlet%20français.txt?dl=0
titre='/Users/willem/Desktop/informatique/#3 chapitre/python_fichier/texte francais /Hamlet français.txt'

#je récupère l'ensemble des lignes du texte et les convertie en texte
def recup_donnee(titre):
    fichier=open(titre,'r')
    ensemble_ligne=''
    for lignes in fichier:
        lignes=lignes.strip()
        if lignes!='':
            ensemble_ligne+=lignes #je forme un texte qui est le texte étudié
    fichier.close()
    return ensemble_ligne

##ensemble des fonctions nécessaire pour la création des clefs de chiffrements:

##indicatrice d'euler
def id_eul(p,q):
    return (p-1)*(q-1) #on a ce calcul car p et q sont premiers

## algorithme d'euclide

#explication: 
#le code utilise le principe suivant : PGCD(a,b)=PGCD(b,r), avec r le reste de la division 
#euclidienne de b par a. On détermine ainsi le PGCD de a,b. Cet algorithme est nécessaire pour 
#trouver l'ensemble de nombres premiers avec a.
def algo_euclide(a,b):
    if b==0:
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
#ce code permet de retrouver des coefficients de la relation de Bézouts
#on trouve un coiuple solution. Néanmoins pour la suite des calculs il me faut
#un v négative donc avec la solution particulière je trouve le solution générale
def div_euclid_etendu(a,b):
    if b == 0:
        return (a,1,0)
    else:
        (d,u,v) = div_euclid_etendu(b,a%b)
        return (d,v,u-((a//b)*v))  

def div_eucl_pos(a,b):
    (_,u,v) = div_euclid_etendu(a,b) #PGCD est constant donc pas utilie ici
    while v<0:
        v+=a
        u-=b
    return (_,u,v)

#code RSA
def RSA(p,q):
    n = p*q   #création du module de chiffrement ave p,q premiers
    phi = id_eul(p,q)     #indicatrice d'euler
    expo=random.choice(premier(phi))   #récupère un nombre premier avec phi de façons arbitraire
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

##chiffrement
#explication:
#le chiffrement consiste à trouver la valeur C tel que C est congru à M puissance expo
#(l'exposant de chiffrement) modulo n (le module de chiffrement). Ces valeurs sont trouvables sur
#la clé publique.on trouvera la valeur de C en trouvant le reste de la division euclidienne de 
#M puissance expo par n.
def chiffrement(M,cle_publ):
    n,expo=cle_publ
    val=M**expo
    C=val%n     #correspond au reste de la div euclidienne de M puissance expo par n
    return C

##2nd étape:
#cette seconde étape n'est réalisable que par le créateur de la clé privée. elle lui permet de 
#pouvoir déchiffrer les caractères. 

##déchiffrement
#explication:
#Grace a la clé privé nous connaissons l'inverse modulaire qui permet de de retrouver l'unique 
#antécédents de C (message crypté). Pour cela, on cherche le reste de la division euclidienne
#de C (caractère crypté) à la puissance dech (exposant de déchiffrement ou inverse modulaire)
#par n.

def dechiffrement(C,cle_priv):
    n,dech=cle_priv
    val=C**dech
    return val%n

## conclusion sur la création les codes de chiffrements et de déchiffrements:
#les codes sont utilisables et possèdent des temps de calculs assez faible. 
#ils sont donc utilisables. Pour coder un message, il faut cependant transformer les caractères 
#sous forme de nombres. On les ordonnera sous forme de bloque de taille constante inférieur 
#à n. Ceci permet de ne pas pouvoir analyser les fréquences des lettres. 

texte="""assistant tu chausses du combien ?"""

##pour empêcher l'analyse des fréquences je vais utiliser la méthodes de césaire/vigenère
## l'idée que Ai = Bi + Ci avec Ai la lettre envoyé Bi la lettre du texte à coder et Ci une 
## lettre quelconque du texte auxiliaire

#je récupère le texte et le transforme en une chaine de chiffre 
def transfo_texte(texte):
    texte=str(texte)
    trans=[]
    for i in range(len(texte)):
        trans.append(ord(texte[i]))
    return trans

#je prend le texte auxiliaire et je transforme chaque caractère en chiffre
def para_texte(titre):
    texte_aux=recup_donnee(titre)
    par_texte=transfo_texte(texte_aux)
    return par_texte

#je cripte le texte avec un texte auxiliaire qui permet d'annuler l'analyse des fréquences
def cript_texte(texte,cle_restreinte,cle_publ):
    (titre,prim_char)=cle_restreinte 
    texte=transfo_texte(texte) #je transforme le texte dans une base python (en chiffre)
    par_texte=para_texte(titre) #idem pour le para texte
    texte_chif=''
    i=prim_char
    for terme in texte:
        val=chiffrement(terme,cle_publ)+par_texte[i]#je chiffre la lettre puis rajoute une lettre
        texte_chif+=chr(val)                        #qui correspond a l'indice de la lettre dans 
        i+=1                                        #le texte auxiliaire
    return texte_chif

#je décripte le texte codé via la clé privée et la clée réstreinte 
def decript_texte(texte_chif,cle_restreinte,cle_priv):
    (titre,prim_char)=cle_restreinte
    par_texte=para_texte(titre)
    texte=''
    i=prim_char
    for terme in texte_chif:
        val=ord(terme)-par_texte[i] #ici je fais pareil mais dans le sens inverse 
        terme=dechiffrement(val,cle_priv)#pour le déchiffrement
        texte+=chr(terme)
        i+=1
    return texte

prim_char=17
cle_restreinte=(titre,prim_char) #la clé restreinte correspond a un passage particulier d'un livre
cle_pub,cle_priv=RSA(p,q)        #clé nécessaire pour le cryptage et le décryptage
texte_1=cript_texte(texte,cle_restreinte,cle_pub) #texte crypté
texte_2=decript_texte(texte_1,cle_restreinte,cle_priv) #retour du texte originale

print(texte,texte_1,texte_2)
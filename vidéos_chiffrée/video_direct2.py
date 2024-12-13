# /usr/bin/python3.8
# -*- coding: utf-8 -*-

import cv2 as cv
from transfo_image import RSA, vecteur_privee_2, matrice_Householder_2
import numpy as np

CAP = cv.VideoCapture(0)
P, Q = 97, 89
P1, Q1 = 43, 47
NB_COL = int(CAP.get(cv.CAP_PROP_FRAME_WIDTH))
NB_LIN = int(CAP.get(cv.CAP_PROP_FRAME_HEIGHT))
CLE_VERNAM = np.random.randint(0, 255, size=(NB_LIN, NB_COL), dtype='uint8')


def transfoCouleur(im, M) : 
    """
    This methos compute a black and white transformation 
    @param : im is the picture matrix 
    M is the Householder matrix. 
    """
    NL, NC, PROFONDEUR = im.shape
    array = np.zeros((NL, NC, PROFONDEUR), dtype='float32')
    for i in range(PROFONDEUR) : 
        array[:,:,i] = im[:,:,i] @ M
    return array

def putIntoImage(final, current, x, y):
    NL, NC, PROFONDEUR = current.shape
    for i in range(PROFONDEUR):
        final[x:x+NL, y:y+NC, i] = current[:,:,i]
    return final

def XORImage(im, cle, round = True):
    """
    This method compute the XOR operation between the image and the key
    """
    if round:
        im = np.round(im)
    im = np.array(im, dtype='uint8')
    try:
        NL, NC, PROFONDEUR = im.shape
        for i in range(PROFONDEUR):
            im[:,:,i] ^= cle
    except:
        NL, NC = im.shape
        PROFONDEUR = 1
        im ^= cle
    return im

def transfo_video_tout(H, H_inv, CLE):
    """
    Main method 
    print filtered and unfiltered image
    """
    final_video = np.zeros((NB_LIN * 2, NB_COL * 2, 3), dtype='uint8')
    while True:
        ret, frame = CAP.read()
        if not ret:
            break
        frame = np.array(frame, dtype='uint8')
        
        frame_chiffre = transfoCouleur(frame, H)
        frame_chiffre_2 = frame

        frame_chiffre_2 = XORImage(frame_chiffre_2, CLE)

        arrayChiffre = transfoCouleur(frame_chiffre_2, H)
        frame_chiffre_20 = frame_chiffre_2[:,:,0] @ H
        frame_chiffre_21 = frame_chiffre_2[:,:,1] @ H
        frame_chiffre_22 = frame_chiffre_2[:,:,2] @ H

        arrayDechiffre = transfoCouleur(arrayChiffre, H_inv)
        frame_dechiffre0 = frame_chiffre_20 @ H_inv
        frame_dechiffre1 = frame_chiffre_21 @ H_inv
        frame_dechiffre2 = frame_chiffre_22 @ H_inv

        frame_chiffre_2 = np.round(arrayChiffre)
        frame_chiffre_2 = np.array(frame_chiffre_2, dtype = 'uint8')
    
        frame_dechiffre = np.zeros((NB_LIN, NB_COL, 3), dtype='float32')
        frame_dechiffre[:,:,0] = frame_dechiffre0
        frame_dechiffre[:,:,1] = frame_dechiffre1
        frame_dechiffre[:,:,2] = frame_dechiffre2

        frame_dechiffre = XORImage(frame_dechiffre, CLE)
        
        putIntoImage(final_video, frame, 0, 0)
        putIntoImage(final_video, frame_chiffre, 0, NB_COL)
        putIntoImage(final_video, arrayChiffre, NB_LIN, NB_COL)
        putIntoImage(final_video, frame_dechiffre, NB_LIN, 0)

        cv.imshow('all videos', final_video)
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()


def main():
    if not CAP.isOpened():
        exit()

    cle_pub, private_key = RSA(P, Q)
    V = vecteur_privee_2(NB_COL, cle_pub)

    H, H_inv = matrice_Householder_2(V, private_key)
    transfo_video_tout(H, H_inv, CLE_VERNAM)

    CAP.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

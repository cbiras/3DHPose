import os
import numpy as np
import cv2
from cpn_config import cfg
import random
import time


def get_seg(height, width, seg_ann):
    label = np.zeros((height, width, 1))
    if type(seg_ann) == list:
        for s in seg_ann:
            poly = np.array(s, np.int).reshape(len(s) // 2, 2)
            cv2.fillPoly(label, [poly], 1)
    else:
        if type(seg_ann['counts']) == list:
            rle = mask.frPyObjects([seg_ann], label.shape[0], label.shape[1])
        else:
            rle = [seg_ann]
        # we set the ground truth as one-hot
        m = mask.decode(rle) * 1
        label[label == 0] = m[label == 0]
    return label[:, :, 0]


def data_augmentation(trainData, trainLabel, trainValids, segms=None):
    trainSegms = segms
    tremNum = cfg.nr_aug - 1
    gotData = trainData.copy()
    trainData = np.append(trainData, [trainData[0] for i in range(tremNum * len(trainData))], axis=0)
    if trainSegms is not None:
        gotSegm = trainSegms.copy()
        trainSegms = np.append(trainSegms, [trainSegms[0] for i in range(tremNum * len(trainSegms))], axis=0)
    trainLabel = np.append(trainLabel, [trainLabel[0] for i in range(tremNum * len(trainLabel))], axis=0)
    trainValids = np.append(trainValids, [trainValids[0] for i in range(tremNum * len(trainValids))], axis=0)
    counter = len(gotData)
    for lab in range(len(gotData)):
        ori_img = gotData[lab].transpose(1, 2, 0)
        if trainSegms is not None:
            ori_segm = gotSegm[lab].copy()
        annot = trainLabel[lab].copy()
        annot_valid = trainValids[lab].copy()
        height, width = ori_img.shape[0], ori_img.shape[1]
        center = (width / 2., height / 2.)
        n = cfg.nr_skeleton

        # affrat = random.uniform(0.75, 1.25)
        affrat = random.uniform(0.7, 1.35)
        halfl_w = min(width - center[0], (width - center[0]) / 1.25 * affrat)
        halfl_h = min(height - center[1], (height - center[1]) / 1.25 * affrat)
        # img = cv2.resize(ori_img[int(center[0] - halfl_w) : int(center[0] + halfl_w + 1), int(center[1] - halfl_h) : int(center[1] + halfl_h + 1)], (width, height))
        img = cv2.resize(ori_img[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                         int(center[0] - halfl_w): int(center[0] + halfl_w + 1)], (width, height))
        if trainSegms is not None:
            segm = cv2.resize(ori_segm[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                              int(center[0] - halfl_w): int(center[0] + halfl_w + 1)], (width, height))
        for i in range(n):
            annot[i << 1] = (annot[i << 1] - center[0]) / halfl_w * (width - center[0]) + center[0]
            annot[i << 1 | 1] = (annot[i << 1 | 1] - center[1]) / halfl_h * (height - center[1]) + center[1]
            annot_valid[i] *= (
            (annot[i << 1] >= 0) & (annot[i << 1] < width) & (annot[i << 1 | 1] >= 0) & (annot[i << 1 | 1] < height))

        trainData[lab] = img.transpose(2, 0, 1)
        if trainSegms is not None:
            trainSegms[lab] = segm
        trainLabel[lab] = annot
        trainValids[lab] = annot_valid

        # flip augmentation
        newimg = cv2.flip(img, 1)
        if trainSegms is not None:
            newsegm = cv2.flip(segm, 1)
        cod = []
        allc = []
        for i in range(n):
            x, y = annot[i << 1], annot[i << 1 | 1]
            if x >= 0:
                x = width - 1 - x
            cod.append((x, y))
        if trainSegms is not None:
            trainSegms[counter] = newsegm
        trainData[counter] = newimg.transpose(2, 0, 1)

        # **** the joint index depends on the dataset ****
        for (q, w) in cfg.symmetry:
            cod[q], cod[w] = cod[w], cod[q]
        for i in range(n):
            allc.append(cod[i][0])
            allc.append(cod[i][1])
        trainLabel[counter] = np.array(allc)
        allc_valid = annot_valid.copy()
        for (q, w) in cfg.symmetry:
            allc_valid[q], allc_valid[w] = allc_valid[w], allc_valid[q]
        trainValids[counter] = np.array(allc_valid)
        counter += 1

        # rotated augmentation
        for times in range(tremNum - 1):
            angle = random.uniform(0, 45)
            if random.randint(0, 1):
                angle *= -1
            rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
            newimg = cv2.warpAffine(img, rotMat, (width, height))
            if trainSegms is not None:
                newsegm = cv2.warpAffine(segm, rotMat, (width, height))

            allc = []
            allc_valid = []
            for i in range(n):
                x, y = annot[i << 1], annot[i << 1 | 1]
                coor = np.array([x, y])
                if x >= 0 and y >= 0:
                    R = rotMat[:, : 2]
                    W = np.array([rotMat[0][2], rotMat[1][2]])
                    coor = np.dot(R, coor) + W
                allc.append(coor[0])
                allc.append(coor[1])
                allc_valid.append(
                    annot_valid[i] * ((coor[0] >= 0) & (coor[0] < width) & (coor[1] >= 0) & (coor[1] < height)))

            newimg = newimg.transpose(2, 0, 1)
            trainData[counter] = newimg
            if trainSegms is not None:
                trainSegms[counter] = newsegm
            trainLabel[counter] = np.array(allc)
            trainValids[counter] = np.array(allc_valid)
            counter += 1
    if trainSegms is not None:
        return trainData, trainLabel, trainSegms
    else:
        return trainData, trainLabel, trainValids

def joints_heatmap_gen(data, label, tar_size=cfg.output_shape, ori_size=cfg.data_shape, points=cfg.nr_skeleton,
                       return_valid=False, gaussian_kernel=cfg.gaussain_kernel):
    if return_valid:
        valid = np.ones((len(data), points), dtype=np.float32)
    ret = np.zeros((len(data), points, tar_size[0], tar_size[1]), dtype='float32')
    for i in range(len(ret)):
        for j in range(points):
            if label[i][j << 1] < 0 or label[i][j << 1 | 1] < 0:
                continue
            label[i][j << 1 | 1] = min(label[i][j << 1 | 1], ori_size[0] - 1)
            label[i][j << 1] = min(label[i][j << 1], ori_size[1] - 1)
            ret[i][j][int(label[i][j << 1 | 1] * tar_size[0] / ori_size[0])][
                int(label[i][j << 1] * tar_size[1] / ori_size[1])] = 1
    for i in range(len(ret)):
        for j in range(points):
            ret[i, j] = cv2.GaussianBlur(ret[i, j], gaussian_kernel, 0)
    for i in range(len(ret)):
        for j in range(cfg.nr_skeleton):
            am = np.amax(ret[i][j])
            if am <= 1e-8:
                if return_valid:
                    valid[i][j] = 0.
                continue
            ret[i][j] /= am / 255
    if return_valid:
        return ret, valid
    else:
        return ret

#primeste imaginea d, si din demo primeste 'test'
def Preprocessing(d, stage='train'):
    height, width = cfg.data_shape
    imgs = []
    labels = []
    valids = []
    #asta e setat pe false in cfg
    if cfg.use_seg:
        segms = []

    vis = False
    #nu stiu cand se seteaza proprietatea de 'data', s-ar putea sa fie salvata direct din dataset. oricum, din ce arata in comentariu, va fi pathul la o imagine
    img = d['data']#cv2.imread(os.path.join(cfg.img_path, d['imgpath']))
    # hack(multiprocessing data provider)
    #daca nu e setata proprietatea 'data', cum s-ar putea sa nu fie
    while img is None:
        import pdb
        pdb.set_trace()
        print('read none image')
        time.sleep(np.random.rand() * 5)
        #img ia valoarea imaginii, ca practic asta se face cv2.imread
        img = cv2.imread(os.path.join(cfg.img_path, d['imgpath']))
    #shape[0] e height, shape[1] e width si shape[2] e channels
    add = max(img.shape[0], img.shape[1])
    #bimg e o imagine ca si img, doar ca la care adauga bordere de dimensiunea cea mai mare a imaginii, si de ceva culoare
    bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                              value=cfg.pixel_means.reshape(-1))

    bbox = np.array(d['bbox']).reshape(4, ).astype(np.float32)
    #la primele 2 coordonate ale bbox-ului, se adauga cea mai mare dimensiune a imaginii
    bbox[:2] += add

    if 'joints' in d:
        #joints o sa fie o matrice cu 17 linii si 3 coloane. banuiesc ca din cauza ca sunt 17 jointuri luate in considerare
        joints = np.array(d['joints']).reshape(cfg.nr_skeleton, 3).astype(np.float32)
        #la primele 2 coloane se adauga cea mai mare dimensiune a imaginii
        joints[:, :2] += add
        #nu stiu de ce joints au si 3 coloane, dar daca a treia coloana are vreo valoare pe 0, se pun si celelalte 2 linii de la coloana respectiva pe -10000
        #am aflat de ce sunt 3 coloane. primele 2 sunt x si y, si a treia banuiesc ca e z sau ceva depth
        inds = np.where(joints[:, -1] == 0)
        joints[inds, :2] = -1000000

    #ideea e ca aici seteaza dimensiunile la care sa fie cropuita poza
    crop_width = bbox[2] * (1 + cfg.imgExtXBorder * 2)
    crop_height = bbox[3] * (1 + cfg.imgExtYBorder * 2)
    #in asta e salvat centrul imaginii
    objcenter = np.array([bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.])

    if stage == 'train':
        crop_width = crop_width * (1 + 0.25)
        crop_height = crop_height * (1 + 0.25)
    #seteaza dimensiunea cea mai mica
    if crop_height / height > crop_width / width:
        crop_size = crop_height
        min_shape = height
    else:
        crop_size = crop_width
        min_shape = width
    crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
    crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

    min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
    max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
    min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
    max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)

    x_ratio = float(width) / (max_x - min_x)
    y_ratio = float(height) / (max_y - min_y)

    #asta e un soi de normalizare la dimensiunile dorite, si practic asigneaza valorile pt jointuri pt training si validare
    if 'joints' in d:
        joints[:, 0] = joints[:, 0] - min_x
        joints[:, 1] = joints[:, 1] - min_y

        joints[:, 0] *= x_ratio
        joints[:, 1] *= y_ratio
        label = joints[:, :2].copy()
        valid = joints[:, 2].copy()
    #imaginea devine imaginea cu border, dar cu widht si height
    img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))

    if stage != 'train':
        #un aray cu coordonatele care sunt diferenta dintre minimurile obtinute mai sus si cea mai mare dimensiune a imaginii initiale
        details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add])

    if cfg.use_seg is True and 'segmentation' in d:
        seg = get_seg(ori_img.shape[0], ori_img.shape[1], d['segmentation'])
        add = max(seg.shape[0], seg.shape[1])
        bimg = cv2.copyMakeBorder(seg, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        seg = cv2.resize(bimg[min_y:max_y, min_x:max_x], (width, height))
        segms.append(seg)
    # e pus in functia asta pe false
    if vis:
        tmpimg = img.copy()
        from utils.visualize import draw_skeleton
        draw_skeleton(tmpimg, label.astype(int))
        cv2.imwrite('vis.jpg', tmpimg)
        from IPython import embed; embed()

    #creadeam ca in pixel means e culoarea borderului da aparent sunt si ceva dimensiuni
    img = img - cfg.pixel_means
    if cfg.pixel_norm:
        img = img / 255.
    img = img.transpose(2, 0, 1)
    imgs.append(img)
    if 'joints' in d:
        labels.append(label.reshape(-1))
        valids.append(valid.reshape(-1))

    if stage == 'train':
        imgs, labels, valids = data_augmentation(imgs, labels, valids)
        heatmaps15 = joints_heatmap_gen(imgs, labels, cfg.output_shape, cfg.data_shape, return_valid=False,
                                        gaussian_kernel=cfg.gk15)
        heatmaps11 = joints_heatmap_gen(imgs, labels, cfg.output_shape, cfg.data_shape, return_valid=False,
                                        gaussian_kernel=cfg.gk11)
        heatmaps9 = joints_heatmap_gen(imgs, labels, cfg.output_shape, cfg.data_shape, return_valid=False,
                                       gaussian_kernel=cfg.gk9)
        heatmaps7 = joints_heatmap_gen(imgs, labels, cfg.output_shape, cfg.data_shape, return_valid=False,
                                       gaussian_kernel=cfg.gk7)

        return [imgs.astype(np.float32).transpose(0, 2, 3, 1),
                heatmaps15.astype(np.float32).transpose(0, 2, 3, 1),
                heatmaps11.astype(np.float32).transpose(0, 2, 3, 1),
                heatmaps9.astype(np.float32).transpose(0, 2, 3, 1),
                heatmaps7.astype(np.float32).transpose(0, 2, 3, 1),
                valids.astype(np.float32)]
    else:
        #practic intoarce un array cu imaginile normalizate si cu border, si un array cu dimensiunile
        return [np.asarray(imgs).astype(np.float32), details]
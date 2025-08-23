from random import choice, sample
import random
import numpy as np
from PIL import Image
import pandas as pd
# import torchvision.transforms as transforms
from torchvision import transforms
import torch
import torchvision
import torch.utils.data as data
import os
import time

"""
20220509: 增加了对CornellKinFace(不区分relation)、UBKinFace数据集的处理
"""
transform_train = transforms.Compose([
    transforms.Resize(73),
    transforms.RandomCrop(64),
    # transforms.RandomResizedCrop((64, 64), (0.8, 1.0)),
    transforms.RandomHorizontalFlip(0.5),  # added by xpchen
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(73),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
tile_tr = []
tile_random_grayscale = 0.1
tile_tr.append(transforms.RandomGrayscale(tile_random_grayscale))
tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

rel_lookup = {'fd': 'father-dau', 'fs': 'father-son', 'md': 'mother-dau', 'ms': 'mother-son', 'all': 'all',
              'allthree': 'allthree'}


class RotateTransform:
    """
    Torchvision-style transform to apply SESEMI augmentation to image.
    """

    classes = ('0', '90', '180', '270')

    def __call__(self, x):
        tf_type = random.randint(0, len(self.classes) - 1)
        if tf_type == 0:
            x = x
        elif tf_type == 1:
            x = transforms.functional.rotate(x, 90)
        elif tf_type == 2:
            x = transforms.functional.rotate(x, 180)
        elif tf_type == 3:
            x = transforms.functional.rotate(x, 270)
        return x, tf_type


class RotateTransformSix:
    """
    Torchvision-style transform to apply SESEMI augmentation to image.
    """

    classes = ('0', '90', '180', '270', 'hflip', 'vflip')

    def __call__(self, x):
        tf_type = random.randint(0, len(self.classes) - 1)
        if tf_type == 0:
            x = x
        elif tf_type == 1:
            x = transforms.functional.rotate(x, 90)
        elif tf_type == 2:
            x = transforms.functional.rotate(x, 180)
        elif tf_type == 3:
            x = transforms.functional.rotate(x, 270)
        elif tf_type == 4:
            x = transforms.functional.hflip(x)
        elif tf_type == 5:
            x = transforms.functional.rotate(x, 180)
            x = transforms.functional.hflip(x)
        return x, tf_type


class grid:
    def __init__(self, grid_size, padding):
        self.grid_size = grid_size
        self.padding = padding

    def __call__(self, x):
        return torchvision.utils.make_grid(x, self.grid_size, self.padding)


def read_img(path, train_flag=True):
    img = Image.open(path).convert('RGB')
    if train_flag == True:

        img = transform_train(img)
    else:
        img = transform_test(img)
    return img


class test_dataloader(data.Dataset):
    def __init__(self, relat="fs", k=1, data_root='./data/KinFaceW-I/'):
        super(test_dataloader, self).__init__()
        self.isTrain = False
        self.k = k
        self.data_root = data_root
        self.relat = relat
        self.data_name = data_root.split("/")[-2]
        if self.data_name == 'UBKinFace':
            self.images_root = data_root + "images/"
        elif self.data_name=='FIW':
            self.images_root = data_root + "images/";
        else:
            temp_relation = rel_lookup[relat]
            self.images_root = data_root + "images/" + temp_relation + '/'

        fold_root = self.data_root + "labels/" + relat + ".csv"
        csv_data = pd.read_csv(fold_root)
        csv_data = csv_data[csv_data['fold'] == k]

        csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                    'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values)}

        self.data = pd.DataFrame(csv_data)

    def __getitem__(self, i):
        if self.data_name == 'UBKinFace':
            if self.relat == 'set1':
                # old parent and child
                images_root1 = self.images_root + '03' + '/'
                images_root2 = self.images_root + '01' + '/'
            elif self.relat == 'set2':
                # young paren and child
                images_root1 = self.images_root + '02' + '/'
                images_root2 = self.images_root + '01' + '/'
        else:
            images_root1 = self.images_root
            images_root2 = self.images_root
        img1_path = os.path.join(images_root1 + self.data.loc[i]['p1'].replace("'", ""))
        img2_path = os.path.join(images_root2 + self.data.loc[i]['p2'].replace("'", ""))

        img1 = read_img(img1_path, self.isTrain)
        img2 = read_img(img2_path, self.isTrain)

        label = self.data.loc[i]['label']

        return img1, img2, label

    def __len__(self):
        return len(self.data)


def meta_data_num(batch_size, relat="fs", k=1, data_root='./data/KinFaceW-I/'):
    # generate balanced samples
    temp_relation = rel_lookup[relat]

    fold_root = data_root + "labels/" + relat + ".csv"
    csv_data = pd.read_csv(fold_root)
    csv_data = csv_data[csv_data['fold'] == k]
    csv_data = csv_data[csv_data['label'] == 1]
    data_name = data_root.split("/")[-2]
    if data_name == 'UBKinFace':
        images_root = data_root + "images/"
    elif data_name == 'KinFaceW-I' or data_name == 'KinFaceW-II' or data_name == 'TSKinFace':
        images_root = data_root + "images/" + temp_relation + '/'
    pid1 = []
    pid2 = []
    for batch_idx, (p11, p12) in enumerate(zip(csv_data['p1'].values, csv_data['p2'].values)):
        p11 = p11.replace("'", "")
        p12 = p12.replace("'", "")
        if data_name == 'KinFaceW-I' or data_name == 'KinFaceW-II' or data_name == 'KinFaceW-I-II':
            label1 = int(p11.split('_')[1]) - 1
            label2 = int(p12.split('_')[1]) - 1

            pid1.append(label1)
            pid2.append(label2)
        elif data_name == 'CornellKinFace':
            label1 = int(p11.split('_')[1].split('.')[0]) - 1
            label2 = int(p12.split('_')[1].split('.')[0]) - 1
            pid1.append(label1)
            pid2.append(label2)
        elif data_name == 'TSKinFace':
            label1 = int(p11.split('-')[1]) - 1
            label2 = int(p12.split('-')[1]) - 1
            pid1.append(label1)
            pid2.append(label2)
        elif data_name == 'UBKinFace':
            label1 = int(p11.split('.')[0]) - 1
            label2 = int(p12.split('.')[0]) - 1
            pid1.append(label1)
            pid2.append(label2)
    return len(np.unique(pid1))


def rotate_image(imgpath, bias_whole_image, ssl_transform):
    img = Image.open(imgpath).convert('RGB')
    order = -1
    if bias_whole_image:  # float,是否倾向于原始的图像
        if bias_whole_image > random.random():
            order = 0
    if order == 0:
        data = transform_train(img)
    else:
        data, order = ssl_transform(img)
        data = transform_train(data)
    return data, order


def meta_data_loader(batch_size, relat, k, test_fold, data_root='./data/KinFaceW-I/', jig_classes=100,
                     tile_transformer=None, patches=True,
                     bias_whole_image=None):
    fold_root = data_root + "labels/" + relat + ".csv"
    ngData = pd.read_csv(fold_root)
    ngData = ngData[ngData['fold'] != test_fold]
    ngData = ngData[ngData['label'] == 1]
    cc = len(ngData)
    each_fold = int(cc / 5)

    # 将csv文件内数据读出
    csv_data = pd.read_csv(fold_root)

    #   添加新的foldn
    ngList = []  # 准备一个列表，把新列的数据存入其中
    index = 0
    for i, row in csv_data.iterrows():  # 遍历数据表，计算每一位名字的长度
        if row['fold'] != test_fold:
            if index < 4 * each_fold:
                ngList.append(int(index / each_fold + 1))
            else:
                ngList.append(5)
            index = index + 1
        else:
            ngList.append(0)
    csv_data['foldn'] = ngList  # 注明列名，就可以直接添加新列
    k.append(0)
    for t in k:
        csv_data = csv_data[csv_data['foldn'] != t]
    csv_data = csv_data[csv_data['label'] == 1]

    data_name = data_root.split("/")[-2]
    if data_name == 'UBKinFace':
        images_root = data_root + "images/"
    elif data_name == 'KinFaceW-I' or data_name == 'KinFaceW-II' or data_name == 'TSKinFace' or data_name == 'CornellKinFace':
        temp_relation = rel_lookup[relat]
        images_root = data_root + "images/" + temp_relation + '/'
    elif data_name=='FIW':
        temp_relation = rel_lookup[relat]
        images_root = data_root + "images/"
    pid1 = []
    pid2 = []
    if data_name != 'FIW':
        for batch_idx, (p11, p12) in enumerate(zip(csv_data['p1'].values, csv_data['p2'].values)):
            p11 = p11.replace("'", "")
            p12 = p12.replace("'", "")
            if data_name == 'KinFaceW-I' or data_name == 'KinFaceW-II' or data_name == 'KinFaceW-I-II':
                label1 = int(p11.split('_')[1]) - 1
                label2 = int(p12.split('_')[1]) - 1

                pid1.append(label1)
                pid2.append(label2)
            elif data_name == 'CornellKinFace':
                label1 = int(p11.split('_')[1].split('.')[0]) - 1
                label2 = int(p12.split('_')[1].split('.')[0]) - 1
                pid1.append(label1)
                pid2.append(label2)
            elif data_name == 'TSKinFace':
                label1 = int(p11.split('-')[1]) - 1
                label2 = int(p12.split('-')[1]) - 1
                pid1.append(label1)
                pid2.append(label2)
            elif data_name == 'UBKinFace':
                label1 = int(p11.split('.')[0]) - 1
                label2 = int(p12.split('.')[0]) - 1
                pid1.append(label1)
                pid2.append(label2)
        if data_name == 'KinFaceW-I-II':
            csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                        'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values),
                        'pid1': pd.Series(pid1), 'pid2': pd.Series(pid2), 'rel': pd.Series(csv_data['rel'].values),
                        'dname': pd.Series(csv_data['dname'].values)}
        else:
            csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                        'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values),
                        'pid1': pd.Series(pid1), 'pid2': pd.Series(pid2)}
    else:
        # for FIW
        pid_container = set()
        for batch_idx, (p11, p12) in enumerate(zip(csv_data['p1'].values, csv_data['p2'].values)):
            p11 = p11.replace("'", "")
            p12 = p12.replace("'", "")
            label1 = p11.split('/')[2]
            label2 = p12.split('/')[2]
            pid_container.add(label1)
            pid_container.add(label2)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for batch_idx, (p11, p12) in enumerate(zip(csv_data['p1'].values, csv_data['p2'].values)):
            p11 = p11.replace("'", "")
            p12 = p12.replace("'", "")
            label1 = pid2label[p11.split('/')[2]]
            label2 = pid2label[p12.split('/')[2]]
            pid1.append(label1)
            pid2.append(label2)
        csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                    'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values),
                    'pid1': pd.Series(pid1), 'pid2': pd.Series(pid2)}

    data = pd.DataFrame(csv_data)
    data_size = len(data)
    images1 = []
    images2 = []
    if data_name == 'UBKinFace':
        if relat == 'set1':
            # old parent and child
            images_root1 = images_root + '03' + '/'
            images_root2 = images_root + '01' + '/'
        elif relat == 'set2':
            # young paren and child
            images_root1 = images_root + '02' + '/'
            images_root2 = images_root + '01' + '/'
    elif data_name == 'KinFaceW-I' or data_name == 'KinFaceW-II' or data_name == 'TSKinFace' or data_name == 'CornellKinFace'\
            or data_name=='FIW':
        images_root1 = images_root
        images_root2 = images_root


    for i in range(len(data)):
        if data_name == 'KinFaceW-I-II':
            tem_rel = data.loc[i]['rel'].replace("'", "")
            tem_rel = rel_lookup[tem_rel]
            img1_path = os.path.join(
                './data/' + data.loc[i]['dname'].replace("'", "") + "/images/" + tem_rel + '/' +
                data.loc[i]['p1'].replace("'", ""))
            img2_path = os.path.join(
                './data/' + data.loc[i]['dname'].replace("'", "") + "/images/" + tem_rel + '/' +
                data.loc[i]['p2'].replace("'", ""))
        else:
            img1_path = os.path.join(images_root1 + data.loc[i]['p1'].replace("'", ""))  # 主要是csv数据问题，替换里面的单引号
            img2_path = os.path.join(images_root2 + data.loc[i]['p2'].replace("'", ""))
        images1.append(img1_path)
        images2.append(img2_path)
    # self-supervised
    returnFunc = RotateTransform()

    while True:
        batch_pos_images1 = []
        batch_pos_images2 = []
        batch_pos_images1_0 = []
        batch_pos_images2_0 = []
        batch_pos_orders1 = []
        batch_pos_orders2 = []
        batch_neg_images1 = []
        batch_neg_images2 = []
        batch_neg_images1_0 = []
        batch_neg_images2_0 = []
        batch_neg_orders1 = []
        batch_neg_orders2 = []
        pos_sample = sample(range(data_size), batch_size // 2)
        neg_sample = [sample(range(data_size), 2) for i in range(batch_size // 2)]
        neg_sample = np.array(neg_sample)
        # positive
        for i in pos_sample:
            img1 = read_img(images1[i], True)
            img1_0, order1 = rotate_image(images1[i], bias_whole_image, returnFunc)
            batch_pos_images1_0.append(torch.as_tensor(img1_0))
            batch_pos_images1.append(torch.as_tensor(img1))
            batch_pos_orders1.append(torch.as_tensor(order1, dtype=torch.int))
        # batch_pos_images1 = [read_img(images1[i], True) for i in pos_sample]
        batch_pos_pids1 = [torch.tensor(data.loc[i]['pid1'], dtype=torch.int) for i in pos_sample]
        batch_pos_pids1 = torch.stack(batch_pos_pids1, dim=0)

        batch_pos_images1 = torch.stack(batch_pos_images1, dim=0)
        batch_pos_images1_0 = torch.stack(batch_pos_images1_0, dim=0)
        batch_pos_orders1 = torch.stack(batch_pos_orders1, dim=0)

        for i in pos_sample:
            img2 = read_img(images2[i], True)
            img2_0, order2 = rotate_image(images2[i], bias_whole_image, returnFunc)
            batch_pos_images2_0.append(torch.as_tensor(img2_0))
            batch_pos_images2.append(torch.as_tensor(img2))
            batch_pos_orders2.append(torch.as_tensor(order2, dtype=torch.int))
        # batch_pos_images2 = [read_img(images2[i], True) for i in pos_sample]
        batch_pos_images2 = torch.stack(batch_pos_images2, dim=0)
        batch_pos_images2_0 = torch.stack(batch_pos_images2_0, dim=0)
        batch_pos_orders2 = torch.stack(batch_pos_orders2, dim=0)
        batch_pos_pids2 = [torch.tensor(data.loc[i]['pid2'], dtype=torch.int) for i in pos_sample]
        batch_pos_pids2 = torch.stack(batch_pos_pids2, dim=0)
        # negative
        for i in neg_sample[:, 0]:
            img1 = read_img(images1[i], True)
            img1_0, order1 = rotate_image(images1[i], bias_whole_image, returnFunc)
            batch_neg_images1.append(torch.as_tensor(img1))
            batch_neg_images1_0.append(torch.as_tensor(img1_0))
            batch_neg_orders1.append(torch.as_tensor(order1, dtype=torch.int))

        # batch_neg_images1 = [read_img(images1[i], True) for i in neg_sample[:, 0]]
        batch_neg_images1 = torch.stack(batch_neg_images1, dim=0)
        batch_neg_images1_0 = torch.stack(batch_neg_images1_0, dim=0)
        batch_neg_orders1 = torch.stack(batch_neg_orders1, dim=0)
        batch_neg_pids1 = [torch.tensor(data.loc[i]['pid1'], dtype=torch.int) for i in neg_sample[:, 0]]
        batch_neg_pids1 = torch.stack(batch_neg_pids1, dim=0)

        for i in neg_sample[:, 1]:
            img2 = read_img(images2[i], True)
            img2_0, order2 = rotate_image(images2[i], bias_whole_image, returnFunc)
            batch_neg_images2.append(torch.as_tensor(img2))
            batch_neg_images2_0.append(torch.as_tensor(img2_0))
            batch_neg_orders2.append(torch.as_tensor(order2, dtype=torch.int))

        # batch_neg_images2 = [read_img(images2[i], True) for i in neg_sample[:, 1]]
        batch_neg_images2 = torch.stack(batch_neg_images2, dim=0)
        batch_neg_images2_0 = torch.stack(batch_neg_images2_0, dim=0)
        batch_neg_orders2 = torch.stack(batch_neg_orders2, dim=0)
        batch_neg_pids2 = [torch.tensor(data.loc[i]['pid2'], dtype=torch.int) for i in neg_sample[:, 1]]
        batch_neg_pids2 = torch.stack(batch_neg_pids2, dim=0)
        labels = [1 if i < batch_size // 2 else 0 for i in range((batch_size // 2) * 2)]

        yield torch.cat((batch_pos_images1, batch_neg_images1), dim=0), \
            torch.cat((batch_pos_images2, batch_neg_images2), dim=0), torch.tensor(labels), \
            torch.cat((batch_pos_pids1, batch_neg_pids1), dim=0), torch.cat((batch_pos_pids2, batch_neg_pids2),
                                                                            dim=0), \
            torch.cat((batch_pos_orders1, batch_neg_orders1), dim=0), torch.cat(
            (batch_pos_orders2, batch_neg_orders2), dim=0), torch.cat((batch_pos_images1_0, batch_neg_images1_0),
                                                                      dim=0), torch.cat(
            (batch_pos_images2_0, batch_neg_images2_0), dim=0)


def meta_data_loaderRandomFold(batch_size, relat, k, test_fold, data_root='./data/KinFaceW-I/', jig_classes=100,
                               tile_transformer=None, patches=True,
                               bias_whole_image=None):
    # 构建新的fold时不是顺序划分为5个fold，而是随机划分
    fold_root = data_root + "labels/" + relat + ".csv"
    ngData = pd.read_csv(fold_root)
    ngData = ngData[ngData['fold'] != test_fold]
    ngData = ngData[ngData['label'] == 1]
    cc = len(ngData)
    each_fold = int(cc / 5)

    # 将csv文件内数据读出
    csv_data = pd.read_csv(fold_root)

    #   添加新的foldn
    ngList = []  # 准备一个列表，把新列的数据存入其中
    # 随机构造fold 20220514
    meta_k = random.sample(range(0, cc), each_fold)
    ngList = [1 if i in meta_k else 0 for i in range(0, cc)]
    ff1 = [index for index in range(0, cc) if ngList[index] != 1]  # 构造fold 1
    # 构造fold 2~5
    for i in range(2, 6):
        count = 0
        if i != 5:
            while count != each_fold:
                meta_k = random.sample(ff1, 1)
                if ngList[meta_k[0]] == 0:
                    ngList[meta_k[0]] = i
                    count = count + 1
        else:  # fold 5,剩下的都是fold 5
            for i in range(0, len(ngList)):
                if ngList[i] == 0:
                    ngList[i] = 5

    csv_data['foldn'] = ngList  # 注明列名，就可以直接添加新列
    k.append(0)
    for t in k:
        csv_data = csv_data[csv_data['foldn'] != t]
    csv_data = csv_data[csv_data['label'] == 1]

    data_name = data_root.split("/")[-2]
    if data_name == 'UBKinFace':
        images_root = data_root + "images/"
    elif data_name == 'KinFaceW-I' or data_name == 'KinFaceW-II' or data_name == 'TSKinFace' or data_name == 'CornellKinFace':
        temp_relation = rel_lookup[relat]
        images_root = data_root + "images/" + temp_relation + '/'

    pid1 = []
    pid2 = []
    for batch_idx, (p11, p12) in enumerate(zip(csv_data['p1'].values, csv_data['p2'].values)):
        p11 = p11.replace("'", "")
        p12 = p12.replace("'", "")
        if data_name == 'KinFaceW-I' or data_name == 'KinFaceW-II' or data_name == 'KinFaceW-I-II':
            label1 = int(p11.split('_')[1]) - 1
            label2 = int(p12.split('_')[1]) - 1

            pid1.append(label1)
            pid2.append(label2)
        elif data_name == 'CornellKinFace':
            label1 = int(p11.split('_')[1].split('.')[0]) - 1
            label2 = int(p12.split('_')[1].split('.')[0]) - 1
            pid1.append(label1)
            pid2.append(label2)
        elif data_name == 'TSKinFace':
            label1 = int(p11.split('-')[1]) - 1
            label2 = int(p12.split('-')[1]) - 1
            pid1.append(label1)
            pid2.append(label2)
        elif data_name == 'UBKinFace':
            label1 = int(p11.split('.')[0]) - 1
            label2 = int(p12.split('.')[0]) - 1
            pid1.append(label1)
            pid2.append(label2)
    if data_name == 'KinFaceW-I-II':
        csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                    'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values),
                    'pid1': pd.Series(pid1), 'pid2': pd.Series(pid2), 'rel': pd.Series(csv_data['rel'].values),
                    'dname': pd.Series(csv_data['dname'].values)}
    else:
        csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                    'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values),
                    'pid1': pd.Series(pid1), 'pid2': pd.Series(pid2)}

    data = pd.DataFrame(csv_data)
    data_size = len(data)
    images1 = []
    images2 = []
    if data_name == 'UBKinFace':
        if relat == 'set1':
            # old parent and child
            images_root1 = images_root + '03' + '/'
            images_root2 = images_root + '01' + '/'
        elif relat == 'set2':
            # young paren and child
            images_root1 = images_root + '02' + '/'
            images_root2 = images_root + '01' + '/'
    elif data_name == 'KinFaceW-I' or data_name == 'KinFaceW-II' or data_name == 'TSKinFace' or data_name == 'CornellKinFace':
        images_root1 = images_root
        images_root2 = images_root

    for i in range(len(data)):
        if data_name == 'KinFaceW-I-II':
            tem_rel = data.loc[i]['rel'].replace("'", "")
            tem_rel = rel_lookup[tem_rel]
            img1_path = os.path.join(
                './data/' + data.loc[i]['dname'].replace("'", "") + "/images/" + tem_rel + '/' +
                data.loc[i]['p1'].replace("'", ""))
            img2_path = os.path.join(
                './data/' + data.loc[i]['dname'].replace("'", "") + "/images/" + tem_rel + '/' +
                data.loc[i]['p2'].replace("'", ""))
        else:
            img1_path = os.path.join(images_root1 + data.loc[i]['p1'].replace("'", ""))  # 主要是csv数据问题，替换里面的单引号
            img2_path = os.path.join(images_root2 + data.loc[i]['p2'].replace("'", ""))
        images1.append(img1_path)
        images2.append(img2_path)
    # self-supervised
    returnFunc = RotateTransform()

    while True:
        batch_pos_images1 = []
        batch_pos_images2 = []
        batch_pos_images1_0 = []
        batch_pos_images2_0 = []
        batch_pos_orders1 = []
        batch_pos_orders2 = []
        batch_neg_images1 = []
        batch_neg_images2 = []
        batch_neg_images1_0 = []
        batch_neg_images2_0 = []
        batch_neg_orders1 = []
        batch_neg_orders2 = []
        pos_sample = sample(range(data_size), batch_size // 2)
        neg_sample = [sample(range(data_size), 2) for i in range(batch_size // 2)]
        neg_sample = np.array(neg_sample)
        # positive
        for i in pos_sample:
            img1 = read_img(images1[i], True)
            img1_0, order1 = rotate_image(images1[i], bias_whole_image, returnFunc)
            batch_pos_images1_0.append(torch.as_tensor(img1_0))
            batch_pos_images1.append(torch.as_tensor(img1))
            batch_pos_orders1.append(torch.as_tensor(order1, dtype=torch.int))
        # batch_pos_images1 = [read_img(images1[i], True) for i in pos_sample]
        batch_pos_pids1 = [torch.tensor(data.loc[i]['pid1'], dtype=torch.int) for i in pos_sample]
        batch_pos_pids1 = torch.stack(batch_pos_pids1, dim=0)

        batch_pos_images1 = torch.stack(batch_pos_images1, dim=0)
        batch_pos_images1_0 = torch.stack(batch_pos_images1_0, dim=0)
        batch_pos_orders1 = torch.stack(batch_pos_orders1, dim=0)

        for i in pos_sample:
            img2 = read_img(images2[i], True)
            img2_0, order2 = rotate_image(images2[i], bias_whole_image, returnFunc)
            batch_pos_images2_0.append(torch.as_tensor(img2_0))
            batch_pos_images2.append(torch.as_tensor(img2))
            batch_pos_orders2.append(torch.as_tensor(order2, dtype=torch.int))
        # batch_pos_images2 = [read_img(images2[i], True) for i in pos_sample]
        batch_pos_images2 = torch.stack(batch_pos_images2, dim=0)
        batch_pos_images2_0 = torch.stack(batch_pos_images2_0, dim=0)
        batch_pos_orders2 = torch.stack(batch_pos_orders2, dim=0)
        batch_pos_pids2 = [torch.tensor(data.loc[i]['pid2'], dtype=torch.int) for i in pos_sample]
        batch_pos_pids2 = torch.stack(batch_pos_pids2, dim=0)
        # negative
        for i in neg_sample[:, 0]:
            img1 = read_img(images1[i], True)
            img1_0, order1 = rotate_image(images1[i], bias_whole_image, returnFunc)
            batch_neg_images1.append(torch.as_tensor(img1))
            batch_neg_images1_0.append(torch.as_tensor(img1_0))
            batch_neg_orders1.append(torch.as_tensor(order1, dtype=torch.int))

        # batch_neg_images1 = [read_img(images1[i], True) for i in neg_sample[:, 0]]
        batch_neg_images1 = torch.stack(batch_neg_images1, dim=0)
        batch_neg_images1_0 = torch.stack(batch_neg_images1_0, dim=0)
        batch_neg_orders1 = torch.stack(batch_neg_orders1, dim=0)
        batch_neg_pids1 = [torch.tensor(data.loc[i]['pid1'], dtype=torch.int) for i in neg_sample[:, 0]]
        batch_neg_pids1 = torch.stack(batch_neg_pids1, dim=0)

        for i in neg_sample[:, 1]:
            img2 = read_img(images2[i], True)
            img2_0, order2 = rotate_image(images2[i], bias_whole_image, returnFunc)
            batch_neg_images2.append(torch.as_tensor(img2))
            batch_neg_images2_0.append(torch.as_tensor(img2_0))
            batch_neg_orders2.append(torch.as_tensor(order2, dtype=torch.int))

        # batch_neg_images2 = [read_img(images2[i], True) for i in neg_sample[:, 1]]
        batch_neg_images2 = torch.stack(batch_neg_images2, dim=0)
        batch_neg_images2_0 = torch.stack(batch_neg_images2_0, dim=0)
        batch_neg_orders2 = torch.stack(batch_neg_orders2, dim=0)
        batch_neg_pids2 = [torch.tensor(data.loc[i]['pid2'], dtype=torch.int) for i in neg_sample[:, 1]]
        batch_neg_pids2 = torch.stack(batch_neg_pids2, dim=0)
        labels = [1 if i < batch_size // 2 else 0 for i in range((batch_size // 2) * 2)]

        yield torch.cat((batch_pos_images1, batch_neg_images1), dim=0), \
            torch.cat((batch_pos_images2, batch_neg_images2), dim=0), torch.tensor(labels), \
            torch.cat((batch_pos_pids1, batch_neg_pids1), dim=0), torch.cat((batch_pos_pids2, batch_neg_pids2),
                                                                            dim=0), \
            torch.cat((batch_pos_orders1, batch_neg_orders1), dim=0), torch.cat(
            (batch_pos_orders2, batch_neg_orders2), dim=0), torch.cat((batch_pos_images1_0, batch_neg_images1_0),
                                                                      dim=0), torch.cat(
            (batch_pos_images2_0, batch_neg_images2_0), dim=0)


class train_pos_dataloader(data.Dataset):
    # positive pairs
    def __init__(self, relat, k, test_fold, data_root='./data/KinFaceW-I/', jig_classes=100, img_transformer=None,
                 tile_transformer=None, patches=True,
                 bias_whole_image=None):
        super(train_pos_dataloader, self).__init__()
        self.isTrain = True
        self.k = k
        self.data_root = data_root
        self.relation = relat
        self.data_name = data_root.split("/")[-2]
        if self.data_name == 'UBKinFace':
            self.images_root = self.data_root + "images/"
        elif self.data_name == 'KinFaceW-I' or self.data_name == 'KinFaceW-II' or self.data_name == 'TSKinFace' or self.data_name == 'CornellKinFace':
            temp_relation = rel_lookup[relat]
            self.images_root = self.data_root + "images/" + temp_relation + '/'
        elif self.data_name == 'FIW':
            self.images_root = self.data_root + "images/";

        fold_root = self.data_root + "labels/" + relat + ".csv"
        # 计算训练集要是重新分5折的话每折的数量
        ngData = pd.read_csv(fold_root)
        ngData = ngData[ngData['fold'] != test_fold]
        ngData = ngData[ngData['label'] == 1]
        cc = len(ngData)
        each_fold = int(cc / 5)

        # 将csv文件内数据读出
        csv_data = pd.read_csv(fold_root)

        #   添加新的foldn
        ngList = []  # 准备一个列表，把新列的数据存入其中
        index = 0
        for i, row in csv_data.iterrows():  # 遍历数据表，计算每一位名字的长度
            if row['fold'] != test_fold:
                if index < 4 * each_fold:
                    ngList.append(int(index / each_fold) + 1)
                else:
                    ngList.append(5)
                index = index + 1
            else:
                ngList.append(0)
        csv_data['foldn'] = ngList  # 注明列名，就可以直接添加新列
        # csv_data.to_csv('../data/namegender.csv', index=False)  # 把数据写入数据集，index=False表示不加索引
        # 注意这里的ngData['length']=ngList是直接在原有数据基础上加了一列新的数据，也就是说现在的ngData已经具备完整的3列数据
        # 不用再在to_csv中加mode=‘a’这个参数，实现不覆盖添加。

        # 查看修改后的csv文件
        # ngData1 = pd.read_csv('../data/namegender.csv')
        # print("new ngData:\n", ngData1)
        k.append(0)
        for t in k:
            csv_data = csv_data[csv_data['foldn'] != t]
        csv_data = csv_data[csv_data['label'] == 1]
        if self.data_name != 'FIW':
            pid = []
            for batch_idx, (p11, p12) in enumerate(zip(csv_data['p1'].values, csv_data['p2'].values)):
                p11 = p11.replace("'", "")
                p12 = p12.replace("'", "")
                if self.data_name == 'KinFaceW-I' or self.data_name == 'KinFaceW-II' or self.data_name == 'KinFaceW-I-II':
                    label1 = int(p11.split('_')[1]) - 1
                    label2 = int(p12.split('_')[1]) - 1

                    pid.append(label1)
                elif self.data_name == 'CornellKinFace':
                    label1 = int(p11.split('_')[1].split('.')[0]) - 1
                    label2 = int(p12.split('_')[1].split('.')[0]) - 1
                    pid.append(label1)
                elif self.data_name == 'TSKinFace':
                    label1 = int(p11.split('-')[1]) - 1
                    label2 = int(p12.split('-')[1]) - 1
                    pid.append(label1)
                elif self.data_name == 'UBKinFace':
                    label1 = int(p11.split('.')[0]) - 1
                    label2 = int(p12.split('.')[0]) - 1
                    pid.append(label1)
                else:
                    label1 = int(p11.split('_')[0]) - 1
                    label2 = int(p12.split('_')[0]) - 1
                    pid.append(label1)
            if self.data_name == 'KinFaceW-I-II':
                csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                            'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values),
                            'pid': pd.Series(pid), 'rel': pd.Series(csv_data['rel'].values),
                            'dname': pd.Series(csv_data['dname'].values)}
            else:
                csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                            'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values),
                            'pid': pd.Series(pid)}
            self.num_cls = len(np.unique(pid))
        else:
            # for FIW
            pid_container = set()
            for batch_idx, (p11, p12) in enumerate(zip(csv_data['p1'].values, csv_data['p2'].values)):
                p11 = p11.replace("'", "")
                p12 = p12.replace("'", "")
                label1 = p11.split('/')[2]
                label2 = p12.split('/')[2]
                pid_container.add(label1)
                pid_container.add(label2)
            self.pid2label = {pid: label for label, pid in enumerate(pid_container)}
            csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                        'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values)
                        }
            self.num_cls = len(np.unique(pid_container))
        self.data = pd.DataFrame(csv_data)

        # self-supervised
        self.bias_whole_image = bias_whole_image
        self.returnFunc = RotateTransform()

    def get_num_cls(self):
        return self.num_cls

    def __getitem__(self, i):
        if self.data_name == 'KinFaceW-I-II':
            tem_rel = self.data.loc[i]['rel'].replace("'", "")
            tem_rel = rel_lookup[tem_rel]
            img1_path = os.path.join(
                './data/' + self.data.loc[i]['dname'].replace("'", "") + "/images/" + tem_rel + '/' +
                self.data.loc[i]['p1'].replace("'", ""))
            img2_path = os.path.join(
                './data/' + self.data.loc[i]['dname'].replace("'", "") + "/images/" + tem_rel + '/' +
                self.data.loc[i]['p2'].replace("'", ""))
        elif self.data_name == 'UBKinFace':
            if self.relation == 'set1':
                # old parent and child
                images_root1 = self.images_root + '03' + '/'
                images_root2 = self.images_root + '01' + '/'
            elif self.relation == 'set2':
                # young paren and child
                images_root1 = self.images_root + '02' + '/'
                images_root2 = self.images_root + '01' + '/'
            img1_path = os.path.join(images_root1 + self.data.loc[i]['p1'].replace("'", ""))
            img2_path = os.path.join(images_root2 + self.data.loc[i]['p2'].replace("'", ""))
        elif self.data_name == 'FIW':
            file1 = self.data.loc[i]['p1'].replace("'", "")
            file2 = self.data.loc[i]['p2'].replace("'", "")
            img1_path = os.path.join(self.images_root + file1)
            img2_path = os.path.join(self.images_root + file2)
        else:
            img1_path = os.path.join(self.images_root + self.data.loc[i]['p1'].replace("'", ""))
            img2_path = os.path.join(self.images_root + self.data.loc[i]['p2'].replace("'", ""))

        img1 = read_img(img1_path, self.isTrain)
        img2 = read_img(img2_path, self.isTrain)
        assert isinstance(img1, torch.Tensor)
        assert isinstance(img2, torch.Tensor)
        img1_0, order1 = rotate_image(img1_path, self.bias_whole_image, self.returnFunc)
        img2_0, order2 = rotate_image(img2_path, self.bias_whole_image, self.returnFunc)
        if self.data_name == 'FIW':
            pid1 = self.pid2label[file1.split('/')[2]]
            pid2 = self.pid2label[file2.split('/')[2]]
        else:
            pid1 = self.data.loc[i]['pid']
            pid2 = self.data.loc[i]['pid']

        return img1, img2, torch.tensor(1, dtype=torch.int), torch.tensor(pid1, dtype=torch.int), \
            torch.tensor(pid2, dtype=torch.int), order1, order2, img1_0, img2_0

    def process_image(self, img):
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted (the original image)
        if self.bias_whole_image:  # float,是否倾向于原始的图像
            if self.bias_whole_image > random.random():
                order = 0
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)
        return self.returnFunc(data), int(order)

    def __len__(self):
        return len(self.data)


def train_target_dataloader(batch_size, relat="fs", k=1, data_root='./data/KinFaceW-I/'):
    # generate balanced samples
    temp_relation = rel_lookup[relat]
    images_root = data_root + "images/" + temp_relation + '/'
    fold_root = data_root + "labels/" + relat + ".csv"
    csv_data = pd.read_csv(fold_root)
    csv_data = csv_data[csv_data['fold'] == k]

    csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values)}

    data = pd.DataFrame(csv_data)
    data_size = len(data)
    images1 = []
    images2 = []
    for i in range(len(data)):
        img1_path = os.path.join(images_root + data.loc[i]['p1'].replace("'", ""))  # 主要是csv数据问题，替换里面的单引号
        img2_path = os.path.join(images_root + data.loc[i]['p2'].replace("'", ""))
        images1.append(img1_path)
        images2.append(img2_path)

    while True:
        pos_sample = sample(range(data_size), batch_size // 2)

        neg_sample = [sample(range(data_size), 2) for i in range(batch_size // 2)]
        neg_sample = np.array(neg_sample)
        batch_pos_images1 = [read_img(images1[i], True) for i in pos_sample]
        batch_pos_images1 = torch.stack(batch_pos_images1, dim=0)
        batch_pos_images2 = [read_img(images2[i], True) for i in pos_sample]
        batch_pos_images2 = torch.stack(batch_pos_images2, dim=0)

        batch_neg_images1 = [read_img(images1[i], True) for i in neg_sample[:, 0]]
        batch_neg_images1 = torch.stack(batch_neg_images1, dim=0)
        batch_neg_images2 = [read_img(images2[i], True) for i in neg_sample[:, 1]]
        batch_neg_images2 = torch.stack(batch_neg_images2, dim=0)

        yield torch.cat((batch_pos_images1, batch_neg_images1), dim=0), torch.cat(
            (batch_pos_images2, batch_neg_images2), dim=0)


class train_neg_dataloader(data.Dataset):
    # negative pairs
    def __init__(self, relat, k, test_fold, data_root='./data/KinFaceW-I/', c=2, jig_classes=100, img_transformer=None,
                 tile_transformer=None, patches=True,
                 bias_whole_image=None):
        super(train_neg_dataloader, self).__init__()
        self.isTrain = True
        self.k = k
        self.data_root = data_root
        # temp_relation = rel_lookup[relat]
        # self.images_root = self.data_root + "images/" + temp_relation + '/'
        self.relation = relat
        self.data_name = data_root.split("/")[-2]
        if self.data_name == 'UBKinFace':
            self.images_root = self.data_root + "images/"
        elif self.data_name == 'KinFaceW-I' or self.data_name == 'KinFaceW-II' or self.data_name == 'TSKinFace' or self.data_name == 'CornellKinFace':
            temp_relation = rel_lookup[relat]
            self.images_root = self.data_root + "images/" + temp_relation + '/'
        elif self.data_name == 'FIW':
            self.images_root = self.data_root + "images/";
        fold_root = self.data_root + "labels/" + relat + ".csv"

        ngData = pd.read_csv(fold_root)
        ngData = ngData[ngData['fold'] != test_fold]
        ngData = ngData[ngData['label'] == 1]
        cc = len(ngData)
        each_fold = int(cc / 5)

        # 将csv文件内数据读出
        csv_data = pd.read_csv(fold_root)

        #   添加新的foldn
        ngList = []  # 准备一个列表，把新列的数据存入其中
        index = 0
        for i, row in csv_data.iterrows():  # 遍历数据表，计算每一位名字的长度
            if row['fold'] != test_fold:
                if index < 4 * each_fold:
                    ngList.append(int(index / each_fold + 1))
                else:
                    ngList.append(5)
                index = index + 1
            else:
                ngList.append(0)
        csv_data['foldn'] = ngList  # 注明列名，就可以直接添加新列
        k.append(0)
        for t in k:
            csv_data = csv_data[csv_data['foldn'] != t]
        csv_data = csv_data[csv_data['label'] == 1]

        if self.data_name != 'FIW':
            pid1 = []
            pid2 = []
            for batch_idx, (p11, p12) in enumerate(zip(csv_data['p1'].values, csv_data['p2'].values)):
                p11 = p11.replace("'", "")
                p12 = p12.replace("'", "")
                if self.data_name == 'KinFaceW-I' or self.data_name == 'KinFaceW-II' or self.data_name == 'KinFaceW-I-II':
                    label1 = int(p11.split('_')[1]) - 1
                    label2 = int(p12.split('_')[1]) - 1

                    pid1.append(label1)
                    pid2.append(label2)
                elif self.data_name == 'CornellKinFace':
                    label1 = int(p11.split('_')[1].split('.')[0]) - 1
                    label2 = int(p12.split('_')[1].split('.')[0]) - 1
                    pid1.append(label1)
                    pid2.append(label2)
                elif self.data_name == 'TSKinFace':
                    label1 = int(p11.split('-')[1]) - 1
                    label2 = int(p12.split('-')[1]) - 1
                    pid1.append(label1)
                    pid2.append(label2)
                elif self.data_name == 'UBKinFace':
                    label1 = int(p11.split('.')[0]) - 1
                    label2 = int(p12.split('.')[0]) - 1
                    pid1.append(label1)
                    pid2.append(label2)
                else:
                    label1 = int(p11.split('_')[0]) - 1
                    label2 = int(p12.split('_')[0]) - 1

                    pid1.append(label1)
                    pid2.append(label2)
            if self.data_name == 'KinFaceW-I-II':
                csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                            'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values),
                            'pid1': pd.Series(pid1), 'pid2': pd.Series(pid2), 'rel': pd.Series(csv_data['rel'].values),
                            'dname': pd.Series(csv_data['dname'].values)}
            else:
                csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                            'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values),
                            'pid1': pd.Series(pid1), 'pid2': pd.Series(pid2)}
        else:
            pid_container = set()
            for batch_idx, (p11, p12) in enumerate(zip(csv_data['p1'].values, csv_data['p2'].values)):
                p11 = p11.replace("'", "")
                p12 = p12.replace("'", "")
                label1 = p11.split('/')[2]
                label2 = p12.split('/')[2]
                pid_container.add(label1)
                pid_container.add(label2)
            self.pid2label = {pid: label for label, pid in enumerate(pid_container)}
            csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                        'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values),
                        }

        self.data = pd.DataFrame(csv_data)
        # an unbalanced train batch with a positive to negative ratio of 1 :c (c> 1).
        self.data_len = int(len(self.data) * c)

        # self-supervised
        self.bias_whole_image = bias_whole_image
        self.returnFunc = RotateTransform()

    def __getitem__(self, i):
        neg_sample = sample(range(len(self.data)), 2)  # 随机选择2个索引，构造一个负样本
        temp = [1, 2]
        random.shuffle(temp)
        a1 = 'p' + str(temp[0])
        a2 = 'p' + str(temp[1])
        a11 = 'pid' + str(temp[0])
        a21 = 'pid' + str(temp[1])
        if self.data_name == 'KinFaceW-I-II':
            tem_rel = self.data.loc[neg_sample[0]]['rel'].replace("'", "")
            tem_rel = rel_lookup[tem_rel]
            img1_path = os.path.join(
                './data/' + self.data.loc[neg_sample[0]]['dname'].replace("'", "") + "/images/" + tem_rel + '/' +
                self.data.loc[neg_sample[0]][a1].replace("'", ""))
            tem_rel = self.data.loc[neg_sample[1]]['rel'].replace("'", "")
            tem_rel = rel_lookup[tem_rel]
            img2_path = os.path.join(
                './data/' + self.data.loc[neg_sample[1]]['dname'].replace("'", "") + "/images/" + tem_rel + '/' +
                self.data.loc[neg_sample[1]][a2].replace("'", ""))

        elif self.data_name == 'UBKinFace':
            if self.relation == 'set1':
                # old parent and child
                images_root1 = self.images_root + '03' + '/'
                images_root2 = self.images_root + '01' + '/'
            elif self.relation == 'set2':
                # young paren and child
                images_root1 = self.images_root + '02' + '/'
                images_root2 = self.images_root + '01' + '/'
            img1_path = os.path.join(images_root1 + self.data.loc[neg_sample[0]][a1].replace("'", ""))
            img2_path = os.path.join(images_root2 + self.data.loc[neg_sample[1]][a2].replace("'", ""))
        else:
            img1_path = os.path.join(self.images_root + self.data.loc[neg_sample[0]][a1].replace("'", ""))
            img2_path = os.path.join(self.images_root + self.data.loc[neg_sample[1]][a2].replace("'", ""))

        img1 = read_img(img1_path, self.isTrain)
        img2 = read_img(img2_path, self.isTrain)

        img1_0, order1 = rotate_image(img1_path, self.bias_whole_image, self.returnFunc)
        img2_0, order2 = rotate_image(img2_path, self.bias_whole_image, self.returnFunc)
        if self.data_name == 'FIW':
            temp=self.data.loc[neg_sample[0]][a1].replace("'", "")
            pid1 = self.pid2label[temp.split('/')[2]]
            temp = self.data.loc[neg_sample[0]][a2].replace("'", "")
            pid2 = self.pid2label[temp.split('/')[2]]
        else:
            pid1 = self.data.loc[neg_sample[0]][a11]
            pid2 = self.data.loc[neg_sample[1]][a21]
        return img1, img2, torch.tensor(0, dtype=torch.int), torch.tensor(pid1, dtype=torch.int), \
            torch.tensor(pid2, dtype=torch.int), order1, order2, img1_0, img2_0

    def __len__(self):
        return self.data_len

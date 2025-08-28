import argparse
import time

import torch.nn.functional as F
import torch
import torch.optim as optim
import AveMeter
import datetime
from DataIter_RotateF_MoUpdate_tri_single_newfold import *
from KinRelation import KinRelation, resnet18, KinRel
from resnet_zl import *
import ZL.loss as ZLoss



parser = argparse.ArgumentParser(description='PyTorch Kinship')
# for kinfacew-I->batch-c=4; kinfaceW-II: batch-c=2
parser.add_argument('--batch-c', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 2)')
parser.add_argument('--lr-decay-epoch', type=str, default='100,150',
                    help='epochs at which learning rate decays. default is 100,150.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
# The positive sample batch size on  the KinFaceW-II and TSKinFace datasets was set 16.
# The positive sample batch size on the KinFaceW-I and Cornell KinFace datasets was set to 8

parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--valid-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for validation (default: 32)')
# parser.add_argument('--save-model', type=str, default='Saved_Model/0.5_mean/',
#                         help='where you save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.00025, metavar='LR',
                    help='learning rate')  # 1e-4 real-kin
parser.add_argument('--lr2', type=float, default=1e-3, metavar='LR',
                    help='learning rate')  # 1e-3 virtual-kin
parser.add_argument('--meta-lr', type=float, default=1e-3, metavar='MLR',
                    help='meta learning rate')  # 1e-3 meta-miner
parser.add_argument('--max-epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--manualSeed', type=int, default=-1,
                    help='manual seed')
parser.add_argument('--num-workers', default=0, type=int,
                    help='number of load data workers (default: 4)')
parser.add_argument('--relat', default="md", type=str,
                    help='relationship among 4 classes ( fs, fd, ms, md )(default: fs)')
parser.add_argument('--log', type=str, default='./log/',
                    help='where you save log file')
parser.add_argument('-d', '--target', type=str, default='./data/FIW/',
                    help='target dataset')
parser.add_argument('--dataset_src1', type=str, default='./data/FIW/',
                    help='dataset_src1')
parser.add_argument('--dataset_src2', type=str, default='./data/FIW/',
                    help='dataset_src2')
parser.add_argument('--image_path', type=str, default='images/',
                    help='image_path')
parser.add_argument('--meta_data_path', type=str, default='meta_data/',
                    help='dmeta_data_path')
parser.add_argument('--bias_whole_image', default=0.0, type=float,
                    help='If set, will bias the training procedure to show more often the whole image. Its value is larger, the taining would be bias the whole image')
parser.add_argument('--rotate_n_classes', '-jc', type=int, default=4, help='Number of rotate angle')
parser.add_argument('--tile_random_grayscale', default=0.1, type=float,
                    help='Chance of randomly grayscaling a tile')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.manualSeed is None or args.manualSeed < 0:
    args.manualSeed = random.randint(1, 10000)

if not os.path.exists(args.log):
    os.mkdir(args.log)

torch.set_num_threads(1)
flag_log = False
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(73),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(0.5),  # added by xpchen
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(73),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'tile': transforms.Compose([
        transforms.RandomGrayscale(args.tile_random_grayscale),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def c_loss(x1, x2, lable):
    # d = torch.linalg.norm(x1-x2, 2, dim=-1)
    m = nn.Softmax(dim=1)
    x1 = m(x1)
    x2 = m(x2)
    d = torch.norm(x1 - x2, 2, dim=-1)
    margin = 1
    y = lable.type(torch.float32)
    max = margin - d
    max[max < 0] = 0
    loss = y * (torch.pow(d, 2)) + (1 - y) * (torch.pow(max, 2))
    return loss.mean()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(args.manualSeed)


def acc_run(kin_model, validloader):
    evaluate_result = []
    correct, total = .0, .0
    unloader = transforms.ToPILImage()
    lalala = 0
    for batch_idx, (x1, x2, labels) in enumerate(validloader):
        x1, x2 = Variable(x1, requires_grad=True).float(), Variable(x2, requires_grad=True).float()
        labels = torch.tensor(labels)
        bs = x1.size(0)
        x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()

        kin_prob, _, _, _, _ = kin_model(x1, x2)

        pred = (kin_prob > 0.5).long()
        results = (pred == (labels.long())).long()
        # print(kin_prob)

        ii = 0
        for op in kin_prob:
            if int(labels[ii]) == 1:
                evaluate_result.append('1 \t 0 \t ' + str(op.item()))
            if int(labels[ii]) == 0:
                evaluate_result.append('0 \t 1 \t ' + str(op.item()))
            ii += 1
    return evaluate_result


def momentum_update(model_q, model_k, beta=0.999):
    param_k = model_k.state_dict()
    param_q = model_q.named_parameters()
    for n, q in param_q:
        if n in param_k:
            param_k[n].data.copy_(beta * param_k[n].data + (1 - beta) * q.data)
    model_k.load_state_dict(param_k)


def momentum_update1(model_q, model_k, beta=0.999):
    """20220216PM"""
    param_k = model_k.state_dict()
    param_q = model_q.named_parameters()
    for n, q in param_q:
        if n in param_k:
            param_k[n].data.copy_((1 - beta) * param_k[n].data + beta * q.data)
    model_k.load_state_dict(param_k)


def save_model(relat, tosave_model, epoch, k):
    model_path = str(relat) + '_fold' + str(k) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '.pth'
    save_path = os.path.join(args.save_model, model_path)
    torch.save(tosave_model.state_dict(), save_path)
    with open(os.path.join(args.save_model, 'checkpoint.txt'), 'w') as fin:
        fin.write(model_path + ' ' + str(epoch) + '\n')


def load_model(unload_model):
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)
        print(args.save_model, 'is created!')
    if not os.path.exists(os.path.join(args.save_model, 'checkpoint.txt')):
        f = open(os.path.join(args.save_model, args.relat + '/checkpoint.txt'), 'w')
        print('checkpoint', 'is created!')

    start_index = 0
    with open(os.path.join(args.save_model, args.relat + '/checkpoint.txt'), 'r') as fin:
        lines = fin.readlines()
        if len(lines) > 0:
            model_path, model_index = lines[0].split()
            print('Resuming from', model_path)
            unload_model.load_state_dict(torch.load(os.path.join(args.save_model, model_path)))
            start_index = int(model_index) + 1
    return start_index


class KinModel(MetaModule):
    def __init__(self, num_features=512):
        super(KinModel, self).__init__()
        self.model = resnet18(pretrained=True)
        self.kin = KinRel(num_features)

    def forward(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)
        x = self.kin(x1, x2)
        return x, x1, x2


def build_model():
    feature = resnet18()
    # feature代表网络
    kin_model = KinRelation(feature, 512)
    kin_model.cuda()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    return kin_model


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()

        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            # dist_ap.append(dist[i][mask[i]].max())
            # dist_an.append(dist[i][mask[i] == 0].min())
            dist_ap.append(torch.tensor([float(dist[i][mask[i]].max())]))
            dist_an.append(torch.tensor([float(dist[i][mask[i] == 0].min())]))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss


def euclidean_dist(x, y, squared=True):
    """
    Compute (Squared) Euclidean distance between two tensors.
    Args:
        x: input tensor with size N x D.
        y: input tensor with size M x D.

        return: distance matrix with size N x M.
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception('Invalid input shape.')

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)

    if squared:
        return dist
    else:
        return torch.sqrt(dist + 1e-12)


def self_binary_cross_entropy(x, y, reduction='mean'):
    loss = -((x + 1e-9).log() * y + (1 - x + 1e-9).log() * (1 - y))
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss


all_acc = []
relationlist = ['fs','fd']
data_name = args.target.split("/")[-2]
run_name = os.path.splitext(os.path.basename(__file__))[0]  # 获取当前正在运行的文件的名称,不带后缀
if data_name == 'UBKinFace':
    relationlist = ['set1', 'set2']
if data_name == 'CornellKinFace':
    relationlist = ['all']
if data_name == 'CornellKinFace' or data_name == 'KinFaceW-I':
    meta_batch_size = args.batch_size // 2
else:
    meta_batch_size = args.batch_size
time1 = []

for relat in relationlist:
  
    train_relat = relat
    total_test = 0
    use_target = False
    nowtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    if not os.path.exists(args.log + relat):
        os.mkdir(args.log + relat)
    # prepare data
    courve_result = []
    loss_result = []
    best_all_fold = []
    train_time = 0
    test_time = 0
    for k in range(1, 6):
        validset = test_dataloader(relat=relat, k=k, data_root=args.target)
        validloader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch_size, shuffle=False,
                                                  num_workers=args.num_workers,
                                                  worker_init_fn=np.random.seed(args.manualSeed))
        total_test = total_test + len(validloader)
        # The k-th fold is for test, the remain folds are for training
        real_kin_model = train_Net_zl_RotateF(angle_classes=args.rotate_n_classes)
        real_kin_model.cuda()
        torch.backends.cudnn.benchmark = True
        print('Model Built!')
        num = random.randint(0, 9)
        # 如果是UBKinFace的话则使用I和II的数据集的合集来作为训练
        if data_name == 'UBKinFace' and args.dataset_src1 == 'KinFaceW-I-II':
            temp1 = args.dataset_src1
            temp2 = args.dataset_src1
        else:
            if num % 2 == 0:
                temp1 = args.dataset_src1
                temp2 = args.dataset_src2
            else:
                temp1 = args.dataset_src2
                temp2 = args.dataset_src1

        ##criterion = torch.nn.BCELoss().cuda()
        # Npaircriterion = NpairLoss(reg_lambda=0.002).cuda()
        triplet_criterion_v = ZLoss.TripletLoss(margin=0.0).cuda()
        kinloss = torch.nn.BCELoss().cuda()
        criterion = nn.CrossEntropyLoss().cuda()

        # triplet_criterion_v = HardTripletLoss(margin=0.2)
        criterion2 = torch.nn.MSELoss().cuda()
        # optimizer = Ranger(params=real_kin_model.parameters(), lr=args.lr)
        optimizer_kin = optim.Adam(params=real_kin_model.params(), lr=args.lr, weight_decay=0.0005)

        print("Data loaded!")


        def train_epoch(epoch, k, pos_fold, ret, meta_batch_size):

            # 1.meta-train 选取其中一个source dataset
            train_pos_set = train_pos_dataloader(relat=train_relat, k=pos_fold, test_fold=k,
                                                 data_root=args.dataset_src1,
                                                 jig_classes=args.rotate_n_classes,
                                                 img_transformer=data_transforms['train'],
                                                 tile_transformer=data_transforms['tile'], patches=False,
                                                 bias_whole_image=args.bias_whole_image)
            train_pos_loader = torch.utils.data.DataLoader(train_pos_set, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=args.num_workers,
                                                           worker_init_fn=np.random.seed(args.manualSeed))

            train_neg_set = train_neg_dataloader(relat=train_relat, k=pos_fold, test_fold=k,
                                                 data_root=args.dataset_src1,
                                                 c=args.batch_c,
                                                 jig_classes=args.rotate_n_classes,
                                                 img_transformer=data_transforms['train'],
                                                 tile_transformer=data_transforms['tile'], patches=False,
                                                 bias_whole_image=args.bias_whole_image)
            train_neg_loader = torch.utils.data.DataLoader(train_neg_set,
                                                           batch_size=args.batch_size * args.batch_c,
                                                           shuffle=True, num_workers=args.num_workers,
                                                           worker_init_fn=np.random.seed(args.manualSeed))
            # 2. meta-test：select another source dataset
            meta_test_loader = meta_data_loader(batch_size=meta_batch_size, relat=train_relat, k=ret, test_fold=k,
                                                data_root=args.dataset_src1, jig_classes=args.rotate_n_classes,
                                                tile_transformer=data_transforms['tile'], patches=False,
                                                bias_whole_image=args.bias_whole_image)

            torch.cuda.empty_cache()
            batch_time = AveMeter.AverageMeter()
            data_time = AveMeter.AverageMeter()
            losses = AveMeter.AverageMeter()
            acces = AveMeter.AverageMeter()
            metatrainlosses = AveMeter.AverageMeter()
            metatrainacces = AveMeter.AverageMeter()

            metalosses = AveMeter.AverageMeter()
            metaacces = AveMeter.AverageMeter()
            train_kinloss = AveMeter.AverageMeter()
            meta_kinloss = AveMeter.AverageMeter()
            train_triloss = AveMeter.AverageMeter()
            meta_triloss = AveMeter.AverageMeter()
            train_jigloss = AveMeter.AverageMeter()
            meta_jigloss = AveMeter.AverageMeter()

            end_time = time.time()

            log_path = args.log + relat + '/' + data_name + "_" + relat + "_k_fold_" + str(k) + "_lr_" + str(
                args.lr) + "_bs_" + str(args.batch_size) + "_" + nowtime + ".txt"
            if flag_log:
                log_file = open(log_path, 'a')
                log_file.write('Random seed:%d ' % args.manualSeed)
            total = 0
            correct_test = 0
            correct = 0
            total_test = 0
            for batch_idx, (pos_sam, neg_sam) in enumerate(zip(train_pos_loader, train_neg_loader)):
                """
                1. batch data prepare
                meta-train数据集
                """
                # a sample unbalance train batch
                p_x1, p_x2, p_labels, p_id1, p_id2, p_jig_l1, p_jig_l2, p_x1_0, p_x2_0 = pos_sam
                n_x1, n_x2, n_labels, n_id1, n_id2, n_jig_l1, n_jig_l2, n_x1_0, n_x2_0 = neg_sam

                x1 = torch.cat((p_x1, n_x1), 0)
                x2 = torch.cat((p_x2, n_x2), 0)
                x1_0 = torch.cat((p_x1_0, n_x1_0), 0)
                x2_0 = torch.cat((p_x2_0, n_x2_0), 0)
                labels = torch.cat((p_labels, n_labels), 0).view(-1, 1)
                pid1 = torch.cat((p_id1, n_id1), 0)
                pid2 = torch.cat((p_id2, n_id2), 0)
                orders1 = torch.cat((p_jig_l1, n_jig_l1), 0)
                orders2 = torch.cat((p_jig_l2, n_jig_l2), 0)
                # target

                bz = int(len(p_labels))

                x1, x2, x1_0, x2_0 = Variable(x1, requires_grad=False).float(), Variable(x2,
                                                                                         requires_grad=False).float(), Variable(
                    x1_0, requires_grad=False).float(), Variable(x2_0, requires_grad=False).float()
                labels = torch.as_tensor(labels)
                bs = x1.size(0)
                x1, x2, x1_0, x2_0, labels, pid1, pid2, orders1, orders2 = x1.cuda(), x2.cuda(), x1_0.cuda(), x2_0.cuda(), labels.cuda(), pid1.cuda(), pid2.cuda(), orders1.cuda(), orders2.cuda()
                # a sample balance train batch
                meta_x1, meta_x2, meta_labels, meta_id1, meta_id2, meta_jig_l1, meta_jig_l2, meta_x1_0, meta_x2_0, = next(
                    meta_test_loader)
                meta_jig_l1 = torch.as_tensor(meta_jig_l1, dtype=torch.long)
                meta_jig_l2 = torch.as_tensor(meta_jig_l2, dtype=torch.long)

                meta_labels = meta_labels.view(-1, 1)
                meta_x1, meta_x2 = Variable(meta_x1, requires_grad=False).float(), \
                                   Variable(meta_x2, requires_grad=False).float()
                meta_x1_0, meta_x2_0 = Variable(meta_x1_0, requires_grad=False).float(), \
                                       Variable(meta_x2_0, requires_grad=False).float()
                meta_labels = torch.as_tensor(meta_labels)
                meta_id1 = torch.as_tensor(meta_id1)
                meta_id2 = torch.as_tensor(meta_id2)
                meta_x1, meta_x2, meta_x1_0, meta_x2_0, meta_labels, meta_id1, meta_id2, meta_jig_l1, meta_jig_l2 = meta_x1.cuda(), meta_x2.cuda(), meta_x1_0.cuda(), meta_x2_0.cuda(), meta_labels.cuda(), meta_id1.cuda(), meta_id2.cuda(), meta_jig_l1.cuda(), meta_jig_l2.cuda()

                data_time.update(time.time() - end_time)

                '''
                2. One Step Gradient 
                '''
                real_kin_model.train()

                y_f_hat, v_x1, v_x2, jigsaw_logit1, jigsaw_logit2 = real_kin_model(x1, x2, x1_0, x2_0)

                x = []
                xtemp = (F.normalize(v_x1) + F.normalize(v_x2)) / 2.0
                center1 = torch.mean(xtemp, dim=0)
                x.append(center1)

                v_x1 = v_x1.view(v_x1.size(0), -1)
                v_x2 = v_x2.view(v_x2.size(0), -1)
                f_v = torch.cat((v_x1, v_x2), dim=0).cuda()
                v_image_label = torch.cat((pid1, pid2), dim=0).cuda()

                # cost_v = criterion(y_f_hat, labels.float())
                # cost_v = self_binary_cross_entropy(y_f_hat, labels.float(), reduction='mean')

                cost_v1 = triplet_criterion_v(f_v, v_image_label).cuda()
                cost_v2 = kinloss(y_f_hat, labels.float())
                jigsaw_loss = (criterion(jigsaw_logit1, orders1) + criterion(jigsaw_logit2, orders2)) * 0.5
                loss_c = c_loss(v_x1, v_x2, labels)
                # np_loss = Npaircriterion(f_v, v_image_label)  # npair-loss在这里用不了，因为不能保证每个样本都能找到一个正的，与负样本的构造有关
                ## simi = torch.cosine_similarity(v_x1, v_x2, dim=1).view(-1, 1)
                ## loss4 = criterion2(simi, labels.float().view(-1, 1))  # 余弦相似度

                loss_meta_train = 0.05 * cost_v1 + 1.5 * cost_v2 + 0.5 * jigsaw_loss  # 最原始的权重系数都为1-20220105 0.5 * jigsaw_loss
                # print('Meta-train: cost_v1=%.4f,cost_v2=%.4f,jigsaw_loss=%.4f,loss_c=%.4f' % (
                #     cost_v1, cost_v2, jigsaw_loss, loss_c))

                metatrainlosses.update(loss_meta_train.cpu().data.numpy())

                train_triloss.update(cost_v1.cpu().data.numpy())
                train_kinloss.update(cost_v2.cpu().data.numpy())
                train_jigloss.update(jigsaw_loss.cpu().data.numpy())

                pred0 = (y_f_hat > 0.5).long()
                results0 = (pred0 == (labels.long())).long()
                results0 = results0.cpu().data.numpy()
                acc = sum(results0) * 1.0 / len(results0)
                metatrainacces.update(acc)

                real_kin_model.zero_grad()
                grad_info = torch.autograd.grad(loss_meta_train, real_kin_model.params(), create_graph=True)
                
                newMeta = train_Net_zl_RotateF(
                    angle_classes=args.rotate_n_classes).cuda()  # create a new model for meta-test
                # creatmodel = time.time()
                # newMeta.load_state_dict(real_kin_model.state_dict())  # ?
                momentum_update(real_kin_model, newMeta)
                # newMeta.copyModel(real_kin_model)

                # copymodel = time.time()
                newMeta.update_params1(lr_inner=args.lr, source_params=grad_info, solver='adam')

                del grad_info

                y_g_hat, g_x1, g_x2, g_jigsaw1, g_jigsaw2 = newMeta(meta_x1, meta_x2, meta_x1_0, meta_x2_0)

                g_x1 = g_x1.view(g_x1.size(0), -1)
                g_x2 = g_x2.view(g_x2.size(0), -1)
                g_v = torch.cat((g_x1, g_x2), dim=0)
                g_image_label = torch.cat((meta_id1, meta_id2), dim=0)

                cost_g1 = triplet_criterion_v(g_v, g_image_label).cuda()
                cost_g2 = kinloss(y_g_hat, meta_labels.float())
                jigsaw_loss2 = (criterion(g_jigsaw1, meta_jig_l1) + criterion(g_jigsaw2, meta_jig_l2)) * 0.5
                ## simi2 = torch.cosine_similarity(g_x1, g_x2, dim=1).view(-1, 1)
                ## loss42 = criterion2(simi2, meta_labels.float().view(-1, 1))  # 余弦相似度
                loss_c2 = c_loss(g_x1, g_x2, meta_labels)
                # l_g_meta = criterion(y_g_hat, meta_labels.float())
                # l_g_meta = self_binary_cross_entropy(y_g_hat, meta_labels.float(), reduction='mean')
                l_g_meta = 0.05 * cost_g1 + 1.5 * cost_g2 + 0.5 * jigsaw_loss2
                # print('Meta-test: cost_g1=%.4f,cost_g2=%.4f,jigsaw_loss2=%.4f,loss_c2=%.4f' % (
                #     cost_g1, cost_g2, jigsaw_loss2, loss_c2))

                metalosses.update(l_g_meta.cpu().data.numpy())

                meta_triloss.update(cost_g1.cpu().data.numpy())
                meta_kinloss.update(cost_g2.cpu().data.numpy())
                meta_jigloss.update(jigsaw_loss2.cpu().data.numpy())

                pred = (y_g_hat > 0.5).long()
                results = (pred == (meta_labels.long())).long()
                results = results.cpu().data.numpy()
                acc = sum(results) * 1.0 / len(results)
                metaacces.update(acc)

                loss_final = loss_meta_train + l_g_meta
                losses.update(loss_final.cpu().data.numpy())

                optimizer_kin.zero_grad()
                loss_final.backward()
                optimizer_kin.step()

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                if batch_idx % args.print_freq == 0:
                    # print(cost_v,w_lambda_norm.view(-1))
                    print('[%s]: '
                          'K-Fold: [%d/5]  '
                          'Epoch: [%d/%d][%d/%d]  '
                          'Time %.3f (%.3f)  '
                          'Data %.3f (%.3f)  '
                          'MTRLoss %.3f (%.3f)  '
                          'MTRAcc %.4f (%.4f)  '
                          'MLoss %.3f (%.3f)  '
                          'MAcc %.4f (%.4f)  '
                          'Loss %.3f (%.3f)  ' % (relat, k, epoch, args.max_epochs, batch_idx, len(train_pos_loader),
                                                  batch_time.val, batch_time.avg, data_time.val, data_time.avg,
                                                  metatrainlosses.val, metatrainlosses.avg, metatrainacces.val,
                                                  metatrainacces.avg,
                                                  metalosses.val, metalosses.avg, metaacces.val, metaacces.avg,
                                                  losses.val, losses.avg))
                    print('Meta-train: cost_v1=%.4f,cost_v2=%.4f,jigsaw_loss=%.4f' % (
                        train_triloss.avg, train_kinloss.avg, train_jigloss.avg))
                    if flag_log:
                        log_file.write('Metra-train:'
                                       'Epoch: [%d][%d/%d]  '
                                       'cost_v1= %.4f   '
                                       'cost_v2= %.4f '
                                       'jigsaw_loss= %.4f '
                                       % (epoch, batch_idx, len(train_pos_loader), train_triloss.avg, train_kinloss.avg,
                                          train_jigloss.avg))
                        log_file.write('Meta-test:'
                                       'Epoch: [%d][%d/%d]  '
                                       'cost_v1= %.4f   '
                                       'cost_v2= %.4f '
                                       'jigsaw_loss= %.4f '
                                       % (epoch, batch_idx, len(train_pos_loader), meta_triloss.avg, meta_kinloss.avg,
                                          meta_jigloss.avg))
                        # print(www_p.item(), www_n.item())
                        log_file.write('[%s]: '
                                       'K-Fold: [%d/5]  '
                                       'Epoch: [%d][%d/%d]  '
                                       'Time %.3f (%.3f)  '
                                       'Data %.3f (%.3f)  '
                                       'Loss %.3f (%.3f)  '
                                       'Acc %.4f (%.4f)' % (relat, k, epoch, batch_idx, len(train_pos_loader),
                                                            batch_time.val, batch_time.avg, data_time.val,
                                                            data_time.avg,
                                                            losses.val, losses.avg, acces.val, acces.avg))
                        log_file.close()


        def valid_epoch(epoch, k):
            real_kin_model.eval()
            batch_time = AveMeter.AverageMeter()
            data_time = AveMeter.AverageMeter()
            losses = AveMeter.AverageMeter()
            acces = AveMeter.AverageMeter()

            end_time = time.time()

            log_path = args.log + relat + '/' + data_name + "_" + relat + "_k_fold_" + str(k) + "_lr_" + str(
                args.lr) + "_bs_" + str(args.batch_size) + "_" + nowtime + ".txt"
            if flag_log:
                log_file = open(log_path, 'a')
            total = 0
            total_correct = 0
            for batch_idx, (x1, x2, labels) in enumerate(validloader):
                data_time.update(time.time() - end_time)

                x1, x2 = Variable(x1, requires_grad=True).float(), Variable(x2, requires_grad=True).float()
                labels = torch.as_tensor(labels.view(-1, 1))
                bs = x1.size(0)
                x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()

                kin_prob, _, _, _, _ = real_kin_model(x1, x2, x1, x2)

                loss = kinloss(kin_prob, labels.float())

                losses.update(loss.cpu().data.numpy(), bs)

                pred = (kin_prob > 0.5).long()
                results = (pred == (labels.long())).long()
                results = results.cpu().data.numpy()
                total_correct += sum(results)
                total += len(results)
                acc = sum(results) * 1.000 / len(results)
                acces.update(acc, bs)

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                # if batch_idx % args.print_freq == 0:
                #     print('[%s]: '
                #           'K-Fold: [%d/5]  '
                #           'Valid Epoch: [%d][%d/%d]  '
                #           'Time %.3f (%.3f)  '
                #           'Data %.3f (%.3f)  '
                #           'Loss %.3f (%.3f)  '
                #           'Acc %.4f (%.4f)' % (relat, k, epoch, batch_idx, len(validloader),
                #                                batch_time.val, batch_time.avg, data_time.val, data_time.avg,
                #                                losses.val, losses.avg, acces.val, acces.avg))
                #     if flag_log:
                #         log_file.write('[%s]: '
                #                        'K-Fold: [%d/5]  '
                #                        'Valid Epoch: [%d][%d/%d]  '
                #                        'Time %.3f (%.3f)  '
                #                        'Data %.3f (%.3f)  '
                #                        'Loss %.3f (%.3f)  '
                #                        'Acc %.4f (%.4f)\n' % (relat, k, epoch, batch_idx, len(validloader),
                #                                               batch_time.val, batch_time.avg, data_time.val,
                #                                               data_time.avg,
                #                                               losses.val, losses.avg, acces.val, acces.avg))
            temp = total_correct * 1.0 / total
            print("Valid on %s: final acc: %.4f, %.4f,\033[0;31m current best acc= %.4f \033[0m" % (
                relat, acces.avg, temp, best_acc))
            if flag_log:
                log_file.write("Valid: final acc: %.4f\n" % acces.avg)
                log_file.close()
            return acces.avg


        best_acc = -0.1
        no_update = 0
        for epoch_iter in range(1, args.max_epochs + 1):

            lr_decay = args.lr_decay
            lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')] + [np.inf]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_kin, milestones=lr_decay_epoch,
                                                                gamma=lr_decay,
                                                                last_epoch=-1)
            lr_scheduler.step()
            # k_th fold is for test
            foldall = [i for i in range(1, 6)]
            meta_k = random.sample(foldall, 1)  # meta_test folds

            ret = [i for i in foldall if i not in meta_k]  # ret存储不能是meta-test的fold

            # pos_fold = [t for t in foldall if t != meta_k[0]]
            pos_fold = []  # 用来存储不能为meta-train的那些fold
            for i in range(len(meta_k)):
                pos_fold.append(meta_k[i])
            start1 = time.time()
            train_epoch(epoch_iter, k, pos_fold, ret, meta_batch_size)
            end1 = time.time()
            train_time = train_time + end1 - start1

            start1 = time.time()
            with torch.no_grad():
                valid_acc = valid_epoch(epoch_iter, k)
            end1 = time.time()
            test_time = test_time + end1 - start1
            log_path = args.log + relat + '/' + data_name + "_" + relat + "_k_fold_" + str(k) + "_lr_" + str(
                args.lr) + "_bs_" + str(args.batch_size) + "_" + nowtime + ".txt"
            if flag_log:
                log_file = open(log_path, 'a')

            if best_acc < valid_acc:
                no_update = 1
                # evaluate_result = acc_run(real_kin_model, validloader)
                print("The best acc on Fold %d is getting better from %.4f to %.4f" % (k, best_acc, valid_acc))
                if flag_log:
                    log_file.write(
                        "The best acc on Fold %d is getting better from %.4f to %.4f\n" % (k, best_acc, valid_acc))
                    log_file.close()

                best_acc = valid_acc
                # save_model(relat, real_kin_model,epoch_iter,k)
            else:
                no_update = no_update + 1
            # if (
            #         epoch_iter > args.max_epochs * 0.75 and no_update >= args.max_epochs / 5) or no_update >= args.max_epochs*2 // 3:
            #     break
            print("epoch=%d, no_upate=%d" % (epoch_iter, no_update))
        log_path1 = args.log + relat + '/' + data_name + "_" + relat + "_all_fold_lr_" + str(args.lr) + "_bs_" + str(
            args.batch_size) + "_" + nowtime + ".txt"
        log_file1 = open(log_path1, 'a')
        log_file1.write("%d \t %.4f \n" % (k, best_acc))
        log_file1.close()
        best_all_fold.append(best_acc * 100)
        # evaluate_result += evaluate_result
    av = sum(best_all_fold) / 5.0

    time1.append(train_time / args.max_epochs)
    time1.append(test_time / (total_test * args.max_epochs))
    all_acc.append(av)
    log_path1 = args.log + relat + '/' + data_name + "_" + relat + "_all_fold_lr_" + str(args.lr) + "_bs_" + str(
        args.batch_size) + "_" + nowtime + ".txt"
    log_file1 = open(log_path1, 'a')
    log_file1.write("avg \t %.4f \n" % av)
    log_file1.write("Program:%s  \n" % run_name)
    log_file1.close()
    print(relat, " ".join(str(float('%.4f' % i)) for i in best_all_fold), 'Average:', float('%.4f' % av))
    # np.save('./unbalance/KFW1/ROC_' + relat + '.npy', evaluate_result)
    np.save('./unbalance/KFW1/run1/courve_' + relat + '.npy', courve_result)
    np.save('./unbalance/KFW1/run1/loss_' + relat + '.npy', loss_result)
av1 = sum(all_acc) / len(all_acc)
all_acc.append(av1)
print("\t".join(str(float('%.4f' % i)) for i in all_acc), 'Average:', float('%.4f' % av1))
print("\t".join(str(float('%.4f' % i)) for i in time1))

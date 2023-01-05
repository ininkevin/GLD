import argparse
from email.policy import default
import json
import pickle
from collections import defaultdict, Counter
from os.path import dirname, join
import os
from pyexpat import model
import torch
from torchsummary import summary
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from dataset import Dictionary, VQAFeatureDataset
import utils
import click
import skimage.io as io
import torch
import torch.nn as nn
from collections import OrderedDict
from tensorboardX import SummaryWriter
from attention import Attention, NewAttention, SelfAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet, MLP
from torch.nn import functional as F
import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import pickle as cPickle
import copy
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

image_dir = 'data/images/mscoco/images/' # path to mscoco/val2014, containing all mscoco val images
name = 'val'  # train or val
answer_path = os.path.join('data', 'cp-cache', '%s_target.pkl' % name)
name = "train" if name == "train" else "test"
question_path = os.path.join('data', 'vqacp_v2_%s_questions.json' % name)

with open(question_path) as f:
    questions = json.load(f)
with open(answer_path, 'rb') as f:
    answers = cPickle.load(f)
questions.sort(key=lambda x: x['question_id'])
answers.sort(key=lambda x: x['question_id'])


def parse_args():
    parser = argparse.ArgumentParser("GLD")

    parser.add_argument(
        '-dataset', default = 'cpv2',
        choices=["cpv2", "v2"])
    parser.add_argument(
        '-mode', default="base",
        choices=["base", "gld_joint", "gld_iter", "gld_reg"])
    parser.add_argument('-scale',default = "sin",
    choices=["sin", "increase", None] )
    parser.add_argument('-visual', default = False)
    parser.add_argument('-qid', default = 140)
    parser.add_argument('-epochs', type=int, default=25)
    parser.add_argument('-num_hid', type=int, default=1024)
    parser.add_argument('-output', type=str, default='base')
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-seed', type=int, default=1111, help='random seed')
    
    args = parser.parse_args()
    return args


def invert_dict(d):
    return {v: k for k, v in d.items()}

def expand_batch(*args):
    return (t.unsqueeze(0) for t in args)

def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)

def plot_rect(image, boxes):
    img = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(img)
    for k in range(15):
        box = boxes[k,:]
        drawrect(draw, box, outline='green', width=3)
    img = np.asarray(img)
    return img
def drawrect(drawcontext, xy, outline=None, width=0):
    x1, y1, x2, y2 = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

def _load_image(img_id, dset):
    """ Load an image """
    if img_id in dset.image_id2ix['train'].keys():
        split = 'train'
        img_idx = dset.image_id2ix['train'][img_id]
    else:
        split = 'val'
        img_idx = dset.image_id2ix['val'][img_id]

    name = (12 - len(str(img_id))) * '0' + str(img_id)
    img = io.imread(os.path.join(image_dir, split+'2014', 'COCO_'+split+'2014_' + name + '.jpg'))
    bboxes = torch.from_numpy(np.array(dset.spatial[split][img_idx][:, :4]))
    return img, bboxes

def plot_attention(img, boxes, att):
    white = np.asarray([255, 255, 255])
    pixel_peak = np.zeros((img.shape[0], img.shape[1]))
    for k in range(36):
        for i in range(int(boxes[k][1]), int(boxes[k][3])):
            for j in range(int(boxes[k][0]), int(boxes[k][2])):
                pixel_peak[i,j] = max(pixel_peak[i,j], att[k])
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            img[i,j] = white * (1-pixel_peak[i,j]) + img[i,j] * pixel_peak[i,j]
    if torch.max(att) > 0.5:
        red_box = boxes[torch.argmax(att),:]
        img = Image.fromarray(np.uint8(img))
        draw = ImageDraw.Draw(img)
        drawrect(draw, red_box, outline='red', width=4)
    img = np.asarray(img)
    return img

def get_bias(train_dset,eval_dset):
    answer_voc_size = train_dset.num_ans_candidates

    question_type_to_probs = defaultdict(Counter)

    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score
    question_type_to_prob_array = {}

    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    for ds in [train_dset,eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]

def mask_softmax(x,mask):
    mask=mask.unsqueeze(2).float()
    x2=torch.exp(x-torch.max(x))
    x3=x2*mask
    epsilon=1e-5
    x3_sum=torch.sum(x3,dim=1,keepdim=True)+epsilon
    x4=x3/x3_sum.expand_as(x3)
    return x4

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_att, q_net, q_net_2, v_net, classifier, c_1,c_2):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_att = q_att
        self.q_net = q_net
        self.q_net_2 = q_net_2
        self.v_net = v_net
        self.classifier = classifier
        self.c_1=c_1
        self.c_2=c_2
        self.vision_lin = torch.nn.Linear(1024, 1)
        self.question_lin = torch.nn.Linear(1024, 1)

    def forward(self, v, q, labels, bias, loss_type = None, weight = 0.9):
        
        # print(v.size())
        # print(q.size())
        # print(labels.size())
        # print(bias.size())
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)
        att = nn.functional.softmax(att, 1)

        v_emb = (att * v).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = v_repr * q_repr
        logits = self.classifier(joint_repr)

        q_pred=self.c_1(q_emb.detach())

        q_out=self.c_2(q_pred)

        if labels is not None:
            if loss_type == 'iter_q':
                y_gradient = torch.clamp(labels - bias, min=0, max=1.).detach()      
                loss = F.binary_cross_entropy_with_logits(q_out, y_gradient)
                loss *= labels.size(1)
            elif loss_type == 'iter_vq':
                ref_logits = torch.sigmoid(q_out) + bias
                ref_logits = torch.clamp(ref_logits, min=0., max=1.) * labels
                y_gradient = torch.clamp(labels - ref_logits, min=0, max=1.).detach()   
                loss = F.binary_cross_entropy_with_logits(logits, y_gradient)
                loss *= labels.size(1)
            elif loss_type == 'joint':
                y_gradient = torch.clamp(labels - bias, min=0, max=1.).detach()
                loss_q = F.binary_cross_entropy_with_logits(q_out, y_gradient)
                ref_logits = torch.sigmoid(q_out) + bias
                ref_logits = torch.clamp(ref_logits, min=0., max=1.) * labels
                y_gradient = torch.clamp(labels - ref_logits, min=0, max=1.).detach()
                loss = F.binary_cross_entropy_with_logits(logits, y_gradient) + loss_q
                loss *= labels.size(1)
            elif loss_type == 'reg_q':
                loss = -(q_out.log_softmax(-1) * labels).mean()
                loss *= labels.size(1)
            elif loss_type == 'reg_vq':
                ref_logits = (q_out.softmax(1) + bias) 
                ref_logits = torch.clamp(ref_logits, min=0., max=1.) * labels
                loss_1 = -(logits.log_softmax(-1) * labels).mean()
                loss_2 = -(logits.log_softmax(-1) * ref_logits).mean()
                loss = loss_1 - weight * (loss_2)
                loss *= labels.size(1)
            elif loss_type == 'base':
                loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').mean(-1).mean(0)
                loss *= labels.size(1)
            else:
                loss = None

        return logits, loss, att

def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_att = SelfAttention(q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    q_net_2 = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

    c_1=MLP(input_dim=q_emb.num_hid,dimensions=[1024,1024,dataset.num_ans_candidates])
    c_2=nn.Linear(dataset.num_ans_candidates,dataset.num_ans_candidates)

    return BaseModel(w_emb, q_emb, v_att, q_att, q_net, q_net_2, v_net, classifier, c_1, c_2)

def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train(model, train_loader, eval_loader, qid2type, args, tbx):
    num_epochs=args.epochs
    run_eval=True
    mode = args.mode
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    total_step = 0
    best_eval_score = 0
    score_list = []
    eval_score_list =[]
    loss_list = []
    eval_loss_list = []

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        
        t = time.time()
        for i, (v, q, a, b,type_mask,notype_mask,q_mask) in tqdm(enumerate(train_loader), ncols=100,
                                                   desc="Epoch %d" % (epoch), total=len(train_loader)):
            total_step += 1
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            q_mask=Variable(q_mask).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda().requires_grad_()
            type_mask=Variable(type_mask).float().cuda()
            notype_mask=Variable(notype_mask).float().cuda()

            if mode == "gld_iter":
                pred, loss, _ = model(v, q, a, b, loss_type = 'iter_q')
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()

                pred, loss, _ = model(v, q, a, b, loss_type = 'iter_vq')
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()
            elif mode == "gld_joint":
                pred, loss, _ = model(v, q, a, b, loss_type = 'joint')
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()
            elif mode == "gld_reg":
                if args.scale == "sin":
                    scale = math.sin(math.pi/2 * (epoch+30) / (num_epochs+30))
                elif args.scale == "increase":
                    scale = (epoch+50) / (num_epochs+50)
                else:
                    scale = args.scale
                pred, loss, _ = model(v, q, a, b, loss_type = 'reg_q', weight = scale)
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()

                pred, loss, _ = model(v, q, a, b, loss_type = 'reg_vq', weight = scale)
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()
            elif mode == "base":
                pred, loss, _ = model(v, q, a, b, loss_type = 'base')
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

            total_loss += loss.item() * q.size(0)
            batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score


            


        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        loss_list.append(total_loss)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))


        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, qid2type)
            results["epoch"] = epoch
            results["step"] = total_step
            results["train_score"] = train_score
            model.train(True)
            gap = train_score - 100 * results["score"]
            record = dict(
                score= 100 * results["score"],
                score_yesno= 100 * results['score_yesno'],
                score_other=100 * results['score_other'],
                score_number=100 * results['score_number'],
                total_loss = results['total_loss'],
                gap = gap
            )
            results_list = OrderedDict(record)
            for k, v in results_list.items():
                tbx.add_scalar(f'dev/{k}', v, epoch)
            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']
            eval_loss = results['total_loss']
            eval_score_list.append(100 * eval_score)
            eval_loss_list.append(eval_loss)
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))

            if eval_score > best_eval_score:
                path = 'base_model_%d.pt'% (epoch)
                model_path = os.path.join(args.output, path)
                torch.save(model.state_dict(), model_path)
                best_eval_score = eval_score

        
        model_path = os.path.join(args.output, 'base_model_final.pt')
        torch.save(model.state_dict(), model_path)

def evaluate(model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0 
    total_loss = 0

    for v, q, a, b, qids, q_mask in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        q_mask=Variable(q_mask).cuda()
        a = Variable(a).cuda()
        b = Variable(b).cuda().requires_grad_()
        pred, loss, _ = model(v, q, a, b, loss_type = 'base', weight = 0.)
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        total_loss += loss.item() * q.size(0)
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number
    total_loss /= len(dataloader.dataset)

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
        total_loss = total_loss
    )
    return results
def visualize(model, index, dset):
    
    question = questions[index]
    img_id = question['image_id']

    img, bbox = _load_image(img_id, dset)
        
    print(question['question'])
    name = 'image/' + str(index) + question['question']+ '_ori.jpg'
    im = Image.fromarray(img)
    im.save(name)
    plot_img = plot_rect(copy.copy(img), bbox)

    v, q, a, b, qid, q_mask = dset.__getitem__(index)
    utils.assert_eq(question['question_id'], qid)
    v = Variable(v, requires_grad=False).cuda()
    q = Variable(q, requires_grad=False).cuda()

    model.eval()
    pred, _,atts = model(v.unsqueeze(0), q.unsqueeze(0), None, None, loss_type = None)
    label = torch.argmax(a).data.cpu()
    print(dset.label2ans[label])
    pred = F.softmax(pred.squeeze(0), dim=0).cpu()
    values, indices = pred.topk(5,dim=0, largest=True, sorted=True)
    for i in indices:
        print(dset.label2ans[i])
    name = 'image/' + str(index) + question['question']+ '.jpg'
    im = Image.fromarray(plot_img)
    im.save(name)

    if atts.max() < 0.5:
        scale = 0.55 / atts.max()
    else:
        scale = 1.
    plot_img = plot_attention(copy.copy(img), bbox, atts.squeeze(0) * scale)
    name = 'debiased_att_vis/' + str(index) + '.jpg'
    im = Image.fromarray(plot_img)
    im.save(name)

def get_model_size(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size)/1024/1024
    print('total model size: {:.3f}MB\nparameter_size: {:.3f}B'.format(all_size, param_size))
    return(param_size, param_sum, buffer_size, buffer_sum, all_size)

    

def main():
    args = parse_args()
    dataset=args.dataset
    args.output=os.path.join('logs',args.output)
    if not os.path.isdir(args.output):
        utils.create_dir(args.output)
    else:
        if click.confirm('Directory already exists in {}. Erase?'
                                 .format(args.output, default=False)):
            os.system('rm -r ' + args.output)
            utils.create_dir(args.output)
    tbx = SummaryWriter(args.output)

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building dataset...")
    train_dset = VQAFeatureDataset('train', dictionary, dataset=dataset,
                                cache_image_features=False)
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                cache_image_features=False)
    get_bias(train_dset,eval_dset)

    model = build_baseline0_newatt(train_dset, num_hid=1024).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    get_model_size(model)

    with open('util/qid2type_%s.json'%dataset,'r') as f:
        qid2type=json.load(f)


    model=model.cuda()
    # summary(model.long(), [(512,36,2048), (512,14), (512,2274), (512,2274)])
    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)

    print("Starting training...")
    train(model, train_loader, eval_loader, qid2type, args, tbx)

    if args.visual:
        dset = VQAFeatureDataset('val', dictionary, dataset='cpv2', cache_image_features=False)
        visualize(model, index = args.qid, dset=dset)

    


if __name__ == '__main__':
    main()
import argparse
import json
import pickle
from collections import defaultdict, Counter
from os.path import dirname, join
import os
from dataset import Dictionary, VQAFeatureDataset
from torch.utils.data import DataLoader
import numpy as np
import click

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
from attention import Attention, NewAttention, SelfAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet, MLP
import pickle as cPickle
from torch.nn import functional as F
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import copy
import skimage.io as io

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

def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    parser.add_argument(
        '-dataset', default='cpv2',
        choices=["v2", "cpv2", "cpv1"],
        help="Run on VQA-2.0 instead of VQA-CP 2.0"
    )
    parser.add_argument(
        '-p', "--entropy_penalty", default=0.36, type=float,
        help="Entropy regularizer weight for the learned_mixin model")
    parser.add_argument(
        '-mode', default="updn",
        choices=["updn", "q_debias","v_debias","q_v_debias"],
        help="Kind of ensemble loss to use")
    parser.add_argument(
        '-debias', default="learned_mixin",
        choices=["learned_mixin", "reweight", "bias_product", "none",'focal' , 'rubi', 'gradient'],
        help="Kind of ensemble loss to use")
    

    # Arguments from the original model, we leave this default, except we
    parser.add_argument('-num_hid', type=int, default=1024)
    parser.add_argument('-output', type=str, default='analysis/exp0')
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-seed', type=int, default=1111, help='random seed')
    parser.add_argument('-load_checkpoint_path', type=str, default='logs/base')
    parser.add_argument('-load_qid2score', type=str, default=None)
    parser.add_argument('-visual', default = False)
    parser.add_argument('-qid', type=int, default = 140)
    args = parser.parse_args()
    return args

def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def evaluate(model, dataloader, qid2type):
    score = 0.
    upper_bound = 0.

    for v, q, a, b, qids, q_mask in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=True).cuda()
        q = Variable(q, requires_grad=False).cuda()
        a = Variable(a, requires_grad=False).cuda()
        pred, _, atts = model(v, q, None, None, loss_type = None)

        batch_scores = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_scores.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()

            

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    results = dict(
        upper_bound=upper_bound,
        acc_score = score,
                 
        )
    return results

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

def visualize(model, index, dset, args):
    
    question = questions[index]
    img_id = question['image_id']

    img, bbox = _load_image(img_id, dset)
        
    print(question['question'])
    im = Image.fromarray(img)
    im.save(os.path.join(args.output, 'ori.jpg'))
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
    im = Image.fromarray(plot_img)
    im.save(os.path.join(args.output, 'box.jpg'))

    if atts.max() < 0.5:
        scale = 0.55 / atts.max()
    else:
        scale = 1.
    plot_img = plot_attention(copy.copy(img), bbox, atts.squeeze(0) * scale)
    im = Image.fromarray(plot_img)
    im.save(os.path.join(args.output, 'att.jpg'))

def main():
    args = parse_args()
    dataset=args.dataset
    if not os.path.isdir(args.output):
        utils.create_dir(args.output)
    else:
        if click.confirm('Exp directory already exists in {}. Erase?'
                                 .format(args.output, default=False)):
            os.system('rm -r ' + args.output)
            utils.create_dir(args.output)


    logger = utils.Logger(os.path.join(args.output, 'log.txt'))

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                #   cache_image_features=args.cache_features)
                                cache_image_features=False)

    model = build_baseline0_newatt(eval_dset, num_hid=1024).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')


    with open('util/qid2type_%s.json'%args.dataset,'r') as f:
        qid2type=json.load(f)

    ckpt = torch.load(os.path.join(args.load_checkpoint_path, 'base_model_final.pt'))
    if 'epoch' in ckpt:
        states_ = ckpt['model_state_dict']
    else:
        states_ = ckpt

    # model.load_state_dict(states_)

    model_dict = model.state_dict()
    ckpt = {k: v for k, v in states_.items() if k in model_dict}
    model_dict.update(ckpt)
    model.load_state_dict(model_dict)

    model=model.cuda()
    batch_size = args.batch_size

    torch.backends.cudnn.benchmark = True

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)

    print("Start to test %s..." % args.load_checkpoint_path)
    model.train(False)
    results = evaluate(model, eval_loader, qid2type)

    eval_score = results["acc_score"]
    bound = results["upper_bound"]
    

    logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
    if args.visual:
        visualize(model, index = args.qid, dset=eval_dset, args=args)


 

if __name__ == '__main__':
    main()    
import json
import pickle
from collections import defaultdict, Counter
from os.path import dirname, join
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from dataset import Dictionary, VQAFeatureDataset
import utils
import click

import torch
import torch.nn as nn
from attention import Attention, NewAttention, SelfAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet, MLP
from torch.nn import functional as F
import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt


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

    def forward(self, v, q, labels, bias, v_mask, q_mask, loss_type = None, weight = 0.9):
     
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)

        if v_mask is None:
            att = nn.functional.softmax(att, 1)
        else:
            att= mask_softmax(att,v_mask)

        v_emb = (att * v).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = v_repr * q_repr
        logits = self.classifier(joint_repr)

        q_pred=self.c_1(q_emb.detach())

        q_out=self.c_2(q_pred)

        if labels is not None:
            if loss_type == 'q':
                loss = -(q_out.log_softmax(-1) * labels).mean()
                loss *= labels.size(1)
            elif loss_type == 'vq':
                ref_logits = (q_out.softmax(1) + bias) 
                ref_logits = torch.clamp(ref_logits, min=0., max=1.) * labels
                loss_1 = -(logits.log_softmax(-1) * labels).mean()
                loss_2 = -(logits.log_softmax(-1) * ref_logits).mean()

                loss = loss_1 - weight * (loss_2)
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

def train(model, train_loader, eval_loader, qid2type, output):
    num_epochs=25
    run_eval=True
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0
    score_list = []
    eval_score_list =[]
    loss_list = []
    eval_loss_list = []
    scale_list = []
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        
        t = time.time()
        for i, (v, q, a, b, hintscore,type_mask,notype_mask,q_mask) in tqdm(enumerate(train_loader), ncols=100,
                                                   desc="Epoch %d" % (epoch), total=len(train_loader)):
            total_step += 1

            # scale = math.sin(math.pi/2 * (epoch+30) / (num_epochs+30))
            scale = 1
            # scale = (epoch+50) / (num_epochs+50)
            # scale = math.sin(math.pi/2 * (epoch+50) / (num_epochs+50))
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            q_mask=Variable(q_mask).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda().requires_grad_()
            hintscore = Variable(hintscore).cuda()
            type_mask=Variable(type_mask).float().cuda()
            notype_mask=Variable(notype_mask).float().cuda()

            pred, loss, _ = model(v, q, a, b, None, q_mask, loss_type = 'q', weight = scale)
            if (loss != loss).any():
                raise ValueError("NaN loss")
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()

            pred, loss, _ = model(v, q, a, b, None, q_mask, loss_type = 'vq', weight = scale)
            if (loss != loss).any():
                raise ValueError("NaN loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()
            
            model.train(False)
            _, compare_loss, _ = model(v, q, a, b, None, q_mask, loss_type = 'vq', weight = 0.)
            model.train(True)


            total_loss += compare_loss.item() * q.size(0)
            batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score
            

        scale_list.append(scale)
        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        score_list.append(train_score)
        loss_list.append(total_loss)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))


        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, qid2type,scale)
            results["epoch"] = epoch
            results["step"] = total_step
            results["train_score"] = train_score
            model.train(True)

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
                model_path = os.path.join(output, path)
                torch.save(model.state_dict(), model_path)
                best_eval_score = eval_score

        
        model_path = os.path.join(output, 'base_model_final.pt')
        torch.save(model.state_dict(), model_path)
    fig_path = os.path.join(output, 'scale.jpg')
    plt.plot(np.arange(epoch+1),scale_list,'r-')
    plt.title( chr(947)+' vs. epoch')
    plt.ylabel('weight')
    plt.savefig(fig_path)
    plt.show()
    plt.clf()
    fig_path = os.path.join(output, 'score.jpg')
    plot_cruve(score_list, eval_score_list, epoch, fig_path, type= 'acc.')
    fig_path = os.path.join(output, 'loss.jpg')
    plot_cruve(loss_list, eval_loss_list, epoch, fig_path, type = 'loss')

def plot_cruve(list, eval_list, epoch, fig_path, type):
    x1 = np.arange(epoch+1)
    x2 = np.arange(epoch+1)
    y1 = list
    y2 = eval_list
    if type == 'acc.':
        a=np.array(list)
        np.save('train_score.npy',a)
        b=np.array(eval_list)
        np.save('eval_score.npy',b)
    elif type == 'loss':
        a=np.array(list)
        np.save('train_loss.npy',a)
        b=np.array(eval_list)
        np.save('eval_loss.npy',b)
    plt.plot(x1,y1,'r-')
    plt.plot(x2,y2,'b-')
    plt.legend(['train','test'])
    plt.title('%s vs. epoch'%type)
    plt.ylabel('%s'%type)
    plt.savefig(fig_path)
    plt.show()
    plt.clf()

def evaluate(model, dataloader, qid2type,scale):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0 
    total_loss = 0

    for v, q, a, b, qids, _, q_mask in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        q_mask=Variable(q_mask).cuda()
        a = Variable(a).cuda()
        b = Variable(b).cuda().requires_grad_()
        # pred, _, _ = model(v, q, None, None, None, q_mask, loss_type = None)
        pred, loss, _ = model(v, q, a, b, None, q_mask, loss_type = 'vq', weight = 0.)
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

def main():
    dataset='v2'
    output='gld_tades_iter_ce'
    output=os.path.join('logs_v2',output)
    load_checkpoint_path=None
    if not os.path.isdir(output):
        utils.create_dir(output)
    else:
        if click.confirm('Exp directory already exists in {}. Erase?'
                                 .format(output, default=False)):
            os.system('rm -r ' + output)
            utils.create_dir(output)

        else:
            if load_checkpoint_path is None:
                os._exit(1)
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building train dataset...")
    train_dset = VQAFeatureDataset('train', dictionary, dataset=dataset,
                                cache_image_features=False)
    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                cache_image_features=False)
    get_bias(train_dset,eval_dset)

    model = build_baseline0_newatt(train_dset, num_hid=1024).cuda()

    model.w_emb.init_embedding('data/glove6b_init_300d.npy')


    with open('util/qid2type_%s.json'%dataset,'r') as f:
        qid2type=json.load(f)

    if load_checkpoint_path is not None:
        ckpt = torch.load(os.path.join('logs', load_checkpoint_path, 'model.pt'))
        # states_ = ckpt
        # model.load_state_dict(states_)
        model_dict = model.state_dict()
        ckpt = {k: v for k, v in ckpt.items() if k in model_dict}
        model_dict.update(model_dict)
        model.load_state_dict(model_dict)

    model=model.cuda()
    batch_size = 512

    torch.manual_seed(1111)
    torch.cuda.manual_seed(1111)
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)

    print("Starting training...")
    train(model, train_loader, eval_loader, qid2type, output)

if __name__ == '__main__':
    main()
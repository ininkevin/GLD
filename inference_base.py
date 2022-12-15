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

def train(model, train_loader, eval_loader, qid2type, output, tbx):
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

            scale = math.sin(math.pi/2 * (epoch+30) / (num_epochs+30))
            # scale = 1
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

            pred, loss, _ = model(v, q, a, b, None, q_mask, loss_type = None)
            if (loss != loss).any():
                raise ValueError("NaN loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            total_loss += loss.item() * q.size(0)
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
            record = dict(
                score= 100 * results["score"],
                score_yesno= 100 * results['score_yesno'],
                score_other=100 * results['score_other'],
                score_number=100 * results['score_number'],
                total_loss = results['total_loss']
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
        pred, loss, _ = model(v, q, a, b, None, q_mask, loss_type = None, weight = 0.)
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

    v, q, a, b, qid, hint_score, q_mask = dset.__getitem__(index)
    utils.assert_eq(question['question_id'], qid)
    v = Variable(v, requires_grad=False).cuda()
    q = Variable(q, requires_grad=False).cuda()
    hint_score = Variable(hint_score, requires_grad=False).cuda()

    model.eval()
    pred, _,atts = model(v.unsqueeze(0), q.unsqueeze(0), None, None, None, q_mask, loss_type = None)
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
    name = 'att_vis/' + str(index) + '.jpg'
    im = Image.fromarray(plot_img)
    im.save(name)

def main():
    dataset='cpv2'
    output='base_inf'
    output=os.path.join('logs',output)
    tbx = SummaryWriter(output)
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


    model=model.cuda()
    batch_size = 512

    torch.manual_seed(1111)
    torch.cuda.manual_seed(1111)
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)

    print("Starting training...")
    train(model, train_loader, eval_loader, qid2type, output, tbx)

    # dset = VQAFeatureDataset('val', dictionary, dataset='cpv2',
    #                         #   cache_image_features=args.cache_features)
    #                         cache_image_features=False)
    
    # visualize(model, index = 141, dset=dset)
    # visualize(model, index = 142, dset=dset)
    # visualize(model, index = 143, dset=dset)
    # visualize(model, index = 144, dset=dset)
    # visualize(model, index = 145, dset=dset)
    # visualize(model, index = 146, dset=dset)
    # visualize(model, index = 147, dset=dset)
    # visualize(model, index = 148, dset=dset)
    # visualize(model, index = 149, dset=dset)
    # visualize(model, index = 150, dset=dset)
    
    # visualize(model, index = 151, dset=dset)
    # visualize(model, index = 152, dset=dset)
    # visualize(model, index = 153, dset=dset)
    # visualize(model, index = 154, dset=dset)
    # visualize(model, index = 155, dset=dset)
    # visualize(model, index = 156, dset=dset)
    # visualize(model, index = 157, dset=dset)
    # visualize(model, index = 158, dset=dset)
    # visualize(model, index = 159, dset=dset)
    # visualize(model, index = 160, dset=dset)
################################################################################################################################


    


if __name__ == '__main__':
    main()
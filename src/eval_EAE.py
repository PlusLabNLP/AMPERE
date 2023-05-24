import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import GenerativeModel
from data import EEDataset, amr_preprocess
from utils import compute_f1
from argparse import ArgumentParser, Namespace
from pattern import patterns
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('-ceae', '--eae_config', required=True)
parser.add_argument('-eae', '--eae_model', required=True)
parser.add_argument('--no_dev', action='store_true', default=False)
parser.add_argument('--eval_batch_size', type=int)
parser.add_argument('--write_file', type=str)
args = parser.parse_args()
with open(args.eae_config) as fp:
    eae_config = json.load(fp)
eae_config = Namespace(**eae_config)

from template_generate import event_template, eve_template_generator

# fix random seed
np.random.seed(eae_config.seed)
torch.manual_seed(eae_config.seed)
torch.backends.cudnn.enabled = False

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)

def get_span_idx(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    This function is how we map the generated prediction back to span prediction.

    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span". 

    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    words = []
    for s in span.split(' '):
        words.extend(tokenizer.encode(s, add_special_tokens=False))
    
    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]

def cal_scores(gold_triggers, pred_triggers, gold_roles, pred_roles):
    assert len(gold_triggers) == len(pred_triggers)
    assert len(gold_roles) == len(pred_roles)    
    # tri_id
    gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        gold_set = set([(t[0], t[1]) for t in gold_trigger])
        pred_set = set([(t[0], t[1]) for t in pred_trigger])
        gold_tri_id_num += len(gold_set)
        pred_tri_id_num += len(pred_set)
        match_tri_id_num += len(gold_set & pred_set)
    
    # tri_cls
    gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        gold_set = set(gold_trigger)
        pred_set = set(pred_trigger)
        gold_tri_cls_num += len(gold_set)
        pred_tri_cls_num += len(pred_set)
        match_tri_cls_num += len(gold_set & pred_set)
    
    # arg_id
    gold_arg_id_num, pred_arg_id_num, match_arg_id_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],)+r[1][:-1] for r in gold_role])
        pred_set = set([(r[0][2],)+r[1][:-1] for r in pred_role])
        
        gold_arg_id_num += len(gold_set)
        pred_arg_id_num += len(pred_set)
        match_arg_id_num += len(gold_set & pred_set)
        
    # arg_cls
    gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],)+r[1] for r in gold_role])
        pred_set = set([(r[0][2],)+r[1] for r in pred_role])
        
        gold_arg_cls_num += len(gold_set)
        pred_arg_cls_num += len(pred_set)
        match_arg_cls_num += len(gold_set & pred_set)
    
    scores = {
        'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
        'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num),
        'arg_id': (gold_arg_id_num, pred_arg_id_num, match_arg_id_num) + compute_f1(pred_arg_id_num, gold_arg_id_num, match_arg_id_num),
        'arg_cls': (gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num) + compute_f1(pred_arg_cls_num, gold_arg_cls_num, match_arg_cls_num),
    }
    
    return scores

# set GPU device
torch.cuda.set_device(eae_config.gpu_device)

# check valid styles
assert np.all([style in ['event_type_sent', 'triggers', 'template', 'na_token'] for style in eae_config.input_style])
assert np.all([style in ['argument:sentence'] for style in eae_config.output_style])
              
# tokenizer
eae_tokenizer = AutoTokenizer.from_pretrained(eae_config.model_name, cache_dir=eae_config.cache_dir, use_fast=False)
eae_tokenizer_sequence = AutoTokenizer.from_pretrained(eae_config.model_name, cache_dir=eae_config.cache_dir, use_fast=False, add_prefix_space=True)
special_tokens = ['<Trigger>', '<sep>', '<None>', '<AND>']
eae_tokenizer.add_tokens(special_tokens)
eae_tokenizer_sequence.add_tokens(special_tokens)

if args.eval_batch_size:
    eae_config.eval_batch_size=args.eval_batch_size

# load data
dev_set = EEDataset(eae_tokenizer, eae_config.dev_file, max_length=eae_config.max_length)
test_set = EEDataset(eae_tokenizer, eae_config.test_file, max_length=eae_config.max_length)
dev_batch_num = len(dev_set) // eae_config.eval_batch_size + (len(dev_set) % eae_config.eval_batch_size != 0)
test_batch_num = len(test_set) // eae_config.eval_batch_size + (len(test_set) % eae_config.eval_batch_size != 0)
with open(eae_config.vocab_file) as f:
    vocab = json.load(f)

# load model
logger.info(f"Loading model from {args.eae_model}")
eae_model = GenerativeModel(eae_config, eae_tokenizer_sequence)
eae_model.load_state_dict(torch.load(args.eae_model, map_location=f'cuda:{eae_config.gpu_device}'))
eae_model.cuda(device=eae_config.gpu_device)
eae_model.eval()

# eval dev set
if not args.no_dev:
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev')
    dev_gold_triggers, dev_gold_roles, dev_pred_triggers, dev_pred_roles = [], [], [], []
    for batch in DataLoader(dev_set, batch_size=eae_config.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn):
        progress.update(1)
        p_triggers = batch.triggers
        
        # argument predictions
        p_roles = [[] for _ in range(len(batch.tokens))]
        event_templates = []
        for bid, (tokens, triggers) in enumerate(zip(batch.tokens, p_triggers)):
            event_templates.append((bid,
                eve_template_generator(eae_config.dataset, tokens, triggers, [], eae_config.input_style, eae_config.output_style, vocab, False)))            
        inputs = []
        events = []
        bids = []
        amrgraphs = []
        supplement_input = []
        for i, event_temp in event_templates:
            for data in event_temp.get_training_data():
                inputs.append(data[0])
                events.append(data[2])
                supplement_input.append(data[6])
                amrgraphs.append(batch.amrgraph[i])
                bids.append(i)
        
        if len(inputs) > 0:
            """
            additional processing
            """
            processed_amrgraph = []
            for amrgraph in amrgraphs:
                # handle amr graph
                if hasattr(eae_config, "AMR_model_path") and (eae_config.AMR_model_path.startswith('xfbai/AMRBART') or eae_config.AMR_model_path.startswith('roberta')):
                    try:
                        lineared_amr, lineared_amr_token, amr_adjacency = amr_preprocess(amrgraph, eae_config.AMR_model_path)
                    except:
                        lineared_amr = ''
                        lineared_amr_token = []
                        amr_adjacency = np.zeros((0,0))
                    processed_amrgraph.append(lineared_amr)
                else:
                    lineared_amr, lineared_amr_token, amr_adjacency = amr_preprocess(amrgraph)
                    processed_amrgraph.append(lineared_amr)
                    
            if hasattr(eae_config, "add_amrgraph_input") and eae_config.add_amrgraph_input:
                inputs = [x + ' \n '+ y for x, y in zip(inputs, processed_amrgraph)]
                # TODO: hand-coded SEP here, should be modified afterward.

            if hasattr(eae_config, "only_amrgraph_input") and eae_config.only_amrgraph_input:
                inputs = [x + y for x, y in zip(processed_amrgraph, supplement_input)]

            """
            END of additional processing
            """
            inputs = eae_tokenizer_sequence(inputs, return_tensors='pt', padding=True)
            enc_idxs = inputs['input_ids'].cuda()
            enc_attn = inputs['attention_mask'].cuda()

            final_outputs = eae_model.generate(enc_idxs, enc_attn, num_beams=eae_config.beam_size, max_length=eae_config.max_output_length, amrgraph=processed_amrgraph)
            for p_text, info, bid in zip(final_outputs, events, bids):
                template = event_template(info['event type'], patterns[eae_config.dataset][info['event type']], 
                eae_config.input_style, eae_config.output_style, info['tokens'], info)
                pred_object = template.decode(p_text)

                for span, role_type, _ in pred_object:
                    sid, eid = get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, eae_tokenizer, trigger_span=info['trigger span'])
                    if sid == -1:
                        continue
                    p_roles[bid].append(((info['trigger span']+(info['event type'],)), (sid, eid, role_type)))

        p_roles = [list(set(role)) for role in p_roles]
        
        dev_gold_triggers.extend(batch.triggers)
        dev_gold_roles.extend(batch.roles)
        dev_pred_triggers.extend(p_triggers)
        dev_pred_roles.extend(p_roles)
                
    progress.close()
    
    # calculate scores
    dev_scores = cal_scores(dev_gold_triggers, dev_pred_triggers, dev_gold_roles, dev_pred_roles)
    
    print("---------------------------------------------------------------------")
    print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['tri_id'][3] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][1], 
        dev_scores['tri_id'][4] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][0], dev_scores['tri_id'][5] * 100.0))
    print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['tri_cls'][3] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][1], 
        dev_scores['tri_cls'][4] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][0], dev_scores['tri_cls'][5] * 100.0))
    print("---------------------------------------------------------------------")
    print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['arg_id'][3] * 100.0, dev_scores['arg_id'][2], dev_scores['arg_id'][1], 
        dev_scores['arg_id'][4] * 100.0, dev_scores['arg_id'][2], dev_scores['arg_id'][0], dev_scores['arg_id'][5] * 100.0))
    print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['arg_cls'][3] * 100.0, dev_scores['arg_cls'][2], dev_scores['arg_cls'][1], 
        dev_scores['arg_cls'][4] * 100.0, dev_scores['arg_cls'][2], dev_scores['arg_cls'][0], dev_scores['arg_cls'][5] * 100.0))
    print("---------------------------------------------------------------------")
      
# test set
progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test')
test_gold_triggers, test_gold_roles, test_pred_triggers, test_pred_roles = [], [], [], []
write_object = []
for batch in DataLoader(test_set, batch_size=eae_config.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn):
    progress.update(1)
    
    p_tri_texts = [[] for _ in range(len(batch.tokens))]
    p_arg_texts = [[] for _ in range(len(batch.tokens))]
    p_triggers = batch.triggers
    
    # argument predictions
    p_roles = [[] for _ in range(len(batch.tokens))]
    event_templates = []
    for bid, (tokens, triggers) in enumerate(zip(batch.tokens, p_triggers)):
        event_templates.append((bid,
            eve_template_generator(eae_config.dataset, tokens, triggers, [], eae_config.input_style, eae_config.output_style, vocab, False)))
        
    inputs = []
    events = []
    bids = []
    amrgraphs = []
    supplement_input = []
    for i, event_temp in event_templates:
        for data in event_temp.get_training_data():
            inputs.append(data[0])
            events.append(data[2])
            supplement_input.append(data[6])
            amrgraphs.append(batch.amrgraph[i])
            bids.append(i)

    if len(inputs) > 0:
        """
        additional processing
        """
        processed_amrgraph = []
        for amrgraph in amrgraphs:
            # handle amr graph
            if hasattr(eae_config, "AMR_model_path") and (eae_config.AMR_model_path.startswith('xfbai/AMRBART') or eae_config.AMR_model_path.startswith('roberta')):
                try:
                    lineared_amr, lineared_amr_token, amr_adjacency = amr_preprocess(amrgraph, eae_config.AMR_model_path)
                except:
                    lineared_amr = ''
                    lineared_amr_token = []
                    amr_adjacency = np.zeros((0,0))
                processed_amrgraph.append(lineared_amr)
            else:
                lineared_amr, lineared_amr_token, amr_adjacency = amr_preprocess(amrgraph)
                processed_amrgraph.append(lineared_amr)
                
        if hasattr(eae_config, "add_amrgraph_input") and eae_config.add_amrgraph_input:
            inputs = [x + ' \n '+ y for x, y in zip(inputs, processed_amrgraph)]
            # TODO: hand-coded SEP here, should be modified afterward.

        if hasattr(eae_config, "only_amrgraph_input") and eae_config.only_amrgraph_input:
            inputs = [x + y for x, y in zip(processed_amrgraph, supplement_input)]
        """
        END of additional processing
        """
        inputs = eae_tokenizer_sequence(inputs, return_tensors='pt', padding=True)
        enc_idxs = inputs['input_ids'].cuda()
        enc_attn = inputs['attention_mask'].cuda()
        final_outputs = eae_model.generate(enc_idxs, enc_attn, num_beams=eae_config.beam_size, max_length=eae_config.max_output_length, amrgraph=processed_amrgraph)
        for p_text, info, bid in zip(final_outputs, events, bids):
            template = event_template(info['event type'], patterns[eae_config.dataset][info['event type']], 
                eae_config.input_style, eae_config.output_style, info['tokens'], info)
            pred_object = template.decode(p_text)
            p_arg_texts[bid].append(p_text)

            for span, role_type, _ in pred_object:
                sid, eid = get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, eae_tokenizer, trigger_span=info['trigger span'])
                if sid == -1:
                    continue
                p_roles[bid].append(((info['trigger span']+(info['event type'],)), (sid, eid, role_type)))

    p_roles = [list(set(role)) for role in p_roles]
    
    test_gold_triggers.extend(batch.triggers)
    test_gold_roles.extend(batch.roles)
    test_pred_triggers.extend(p_triggers)
    test_pred_roles.extend(p_roles)
    for gt, gr, pt, pr, tte, ate in zip(batch.triggers, batch.roles, p_triggers, p_roles, p_tri_texts, p_arg_texts):
        write_object.append({
            "pred trigger text": tte,
            "pred role text": ate,
            "pred triggers": pt,
            "gold triggers": gt,
            "pred roles": pr,
            "gold roles": gr
        })
            
progress.close()
    
# calculate scores
test_scores = cal_scores(test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles)

print("---------------------------------------------------------------------")
print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['tri_id'][3] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][1], 
    test_scores['tri_id'][4] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][0], test_scores['tri_id'][5] * 100.0))
print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['tri_cls'][3] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][1], 
    test_scores['tri_cls'][4] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][0], test_scores['tri_cls'][5] * 100.0))
print("---------------------------------------------------------------------")
print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['arg_id'][3] * 100.0, test_scores['arg_id'][2], test_scores['arg_id'][1], 
    test_scores['arg_id'][4] * 100.0, test_scores['arg_id'][2], test_scores['arg_id'][0], test_scores['arg_id'][5] * 100.0))
print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['arg_cls'][3] * 100.0, test_scores['arg_cls'][2], test_scores['arg_cls'][1], 
    test_scores['arg_cls'][4] * 100.0, test_scores['arg_cls'][2], test_scores['arg_cls'][0], test_scores['arg_cls'][5] * 100.0))
print("---------------------------------------------------------------------")

if args.write_file:
    with open(args.write_file, 'w') as fw:
        json.dump(write_object, fw, indent=4)          

import json, logging, pickle, re
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from pattern import patterns
import copy
import penman
from penman.models.noop import NoOpModel
from penman import loads as loads_
import ipdb
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

ee_instance_fields = ['doc_id', 'wnd_id', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'token_start_idxs', 'triggers', 'roles', 'amrgraph']
ee_batch_fields = ['tokens', 'pieces', 'piece_idxs', 'token_lens', 'token_start_idxs', 'triggers', 'roles', 'wnd_ids', 'amrgraph']
EEInstance = namedtuple('EEInstance', field_names=ee_instance_fields, defaults=[None] * len(ee_instance_fields))
EEBatch = namedtuple('EEBatch', field_names=ee_batch_fields, defaults=[None] * len(ee_batch_fields))

gen_batch_fields = ['input_text', 'target_text', 'enc_idxs', 'enc_attn', 'dec_idxs', 'dec_attn', 'lbl_idxs', 'raw_lbl_idxs', 'infos', 'amrgraph']
GenBatch = namedtuple('GenBatch', field_names=gen_batch_fields, defaults=[None] * len(gen_batch_fields))

def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity['start'], entity['end']
        break_flag = False
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]
                break_flag = True
        if break_flag:
            continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    return entities_, id_map

def get_role_list(entities, events, id_map):
    entity_idxs = {entity['id']: (i,entity) for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(events))]
    role_list = []
    role_list = []
    for i, event in enumerate(events):
        for arg in event['arguments']:
            entity_idx = entity_idxs[id_map.get(arg['entity_id'], arg['entity_id'])]
            
            # This will automatically remove multi role scenario
            if visited[i][entity_idx[0]] == 0:
                # ((trigger start, trigger end, trigger type), (argument start, argument end, role type))
                temp = ((event['trigger']['start'], event['trigger']['end'], event['event_type']),
                        (entity_idx[1]['start'], entity_idx[1]['end'], arg['role']))
                role_list.append(temp)
                visited[i][entity_idx[0]] = 1
    role_list.sort(key=lambda x: (x[0][0], x[1][0]))
    return role_list

class EEDataset(Dataset):
    def __init__(self, tokenizer, path, max_length=200, fair_compare=True):
        self.tokenizer = tokenizer
        self.path = path
        self.data = []
        self.insts = []
        self.max_length = max_length
        self.fair_compare = fair_compare
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __iadd__(self, other_dataset):
        self.insts += other_dataset.insts
        self.data += other_dataset.data
        return self

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.insts:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.insts:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        self.insts = []
        for line in lines:
            inst = json.loads(line)
            inst_len = len(inst['pieces'])
            if inst_len > self.max_length:
                print('Warning: Over length {}!!'.format(inst_len))
                continue
            self.insts.append(inst)

        for inst in self.insts:
            doc_id = inst['doc_id']
            wnd_id = inst['wnd_id']
            tokens = inst['tokens']
            pieces = inst['pieces']
            
            entities = inst['entity_mentions']
            if self.fair_compare:
                entities, entity_id_map = remove_overlap_entities(entities)
            else:
                entities = entities
                entity_id_map = {}
                
            events = inst['event_mentions']
            events.sort(key=lambda x: x['trigger']['start'])
            
            token_num = len(tokens)
            token_lens = inst['token_lens']
            
            piece_idxs = self.tokenizer.convert_tokens_to_ids(pieces)
            assert sum(token_lens) == len(piece_idxs)
                        
            triggers = [(e['trigger']['start'], e['trigger']['end'], e['event_type']) for e in events]
            roles = get_role_list(entities, events, entity_id_map)
            
            token_start_idxs = [sum(token_lens[:_]) for _ in range(len(token_lens))] + [sum(token_lens)]
            
            amrgraph = inst['amrgraph']
            
            instance = EEInstance(
                doc_id=doc_id,
                wnd_id=wnd_id,
                tokens=tokens,
                pieces=pieces,
                piece_idxs=piece_idxs,
                token_lens=token_lens,
                token_start_idxs=token_start_idxs,
                triggers = triggers,
                roles = roles,
                amrgraph = amrgraph,
            )
            self.data.append(instance)
            
        logger.info(f'Loaded {len(self)}/{len(lines)} instances from {self.path}')

    def collate_fn(self, batch):
        tokens = [inst.tokens for inst in batch]
        pieces = [inst.pieces for inst in batch]
        piece_idxs = [inst.piece_idxs for inst in batch]
        token_lens = [inst.token_lens for inst in batch]
        token_start_idxs = [inst.token_start_idxs for inst in batch]
        triggers = [inst.triggers for inst in batch]
        roles = [inst.roles for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        amrgraph = [inst.amrgraph for inst in batch]

        return EEBatch(
            tokens=tokens,
            pieces=pieces,
            piece_idxs=piece_idxs,
            token_lens=token_lens,
            token_start_idxs=token_start_idxs,
            triggers=triggers,
            roles=roles,
            wnd_ids=wnd_ids,
            amrgraph=amrgraph
        )

class GenDataset(Dataset):
    def __init__(self, config, tokenizer, max_length, path, max_output_length=None, unseen_types=[], no_bos=False):
        self.config = config
        self.tokenizer = tokenizer

        self.max_length = self.max_output_length = max_length
        if max_output_length is not None:
            self.max_output_length = max_output_length
        self.path = path
        self.no_bos = no_bos # if you use bart, then this should be False; if you use t5, then this should be True
        self.data = []
        self.load_data(unseen_types)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self, unseen_types):
        with open(self.path, 'rb') as f:
            data = pickle.load(f)

        for l_in, l_out, l_info in zip(data['input'], data['target'], data['all']):
            event_type = l_info[1]
            passage = ' '.join(l_info[2])
            supplement_input = l_info[3]
            if len(unseen_types) > 0 and event_type in unseen_types:
                continue

            amrgraph = l_info[4]

            if hasattr(self.config, "AMR_model_path") and (self.config.AMR_model_path.startswith('xfbai/AMRBART') or self.config.AMR_model_path.startswith('roberta')):
                try:
                    lineared_amr, lineared_amr_token, amr_adjacency = amr_preprocess(amrgraph, self.config.AMR_model_path)
                except:
                    print('AMR error! Continue..')
                    continue
            else:
                lineared_amr, lineared_amr_token, amr_adjacency = amr_preprocess(amrgraph)
                
            self.data.append({
                'input': l_in,
                'target': l_out,
                'info': l_info,
                'event_type': event_type,
                'passage': passage,
                'supplement_input': supplement_input,
                'amrgraph': lineared_amr,
                'amrtokens': lineared_amr_token,
                'amr_adjacency': amr_adjacency
            })
        logger.info(f'Loaded {len(self)} instances from {self.path}')

    def collate_fn(self, batch):
        input_text = [x['input'] for x in batch]
        target_text = [x['target'] for x in batch]
        amrgraph = [x['amrgraph'] for x in batch]

        if hasattr(self.config, "add_amrgraph_input") and self.config.add_amrgraph_input:
            input_text = [x + ' \n '+ y for x, y in zip(input_text, amrgraph)]
            # TODO: hand-coded SEP here, should be modified afterward if different base-model is used.

        if hasattr(self.config, "only_amrgraph_input") and self.config.only_amrgraph_input:
            supplement_input = [x['supplement_input'] for x in batch]
            input_text = [x + y for x, y in zip(amrgraph, supplement_input)]

        # encoder inputs
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True)
        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']

        # decoder inputs
        targets = self.tokenizer(target_text, return_tensors='pt', padding=True)
        batch_size = enc_idxs.size(0)

        if self.no_bos:
            # t5 case
            padding = torch.ones((batch_size, 1), dtype=torch.long)
            padding[:] = self.tokenizer.pad_token_id
            # for t5, the decoder input should be:
            # PAD => A
            # A => B
        else:
            padding = torch.ones((batch_size, 1), dtype=torch.long)
            padding[:] = self.tokenizer.eos_token_id
            # for BART, the decoder input should be:
            # PAD => BOS
            # BOS => A
            # A => B          
        dec_idxs = torch.cat((padding, targets['input_ids']), dim=1)
        dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), targets['attention_mask']), dim=1)
        # dec_idxs = targets['input_ids']
        # dec_idxs[:, 0] = self.tokenizer.eos_token_id
        # dec_attn = targets['attention_mask']
            
        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        raw_lbl_idxs = raw_lbl_idxs.cuda()
        lbl_idxs = lbl_idxs.cuda()
        
        return GenBatch(
            input_text=input_text,
            target_text=target_text,
            amrgraph=amrgraph,
            enc_idxs=enc_idxs,
            enc_attn=enc_attn,
            dec_idxs=dec_idxs,
            dec_attn=dec_attn,
            lbl_idxs=lbl_idxs,
            raw_lbl_idxs=raw_lbl_idxs,
            infos=[x['info'] for x in batch]
        )

"""
AMR processing
"""
def amr_preprocess(gstring, amr_model_type="None"):
    # if amr_model_type.startswith('xfbai/AMRBART'):
    if amr_model_type.startswith('xfbai/AMRBART') or amr_model_type.startswith('roberta'):
        model = NoOpModel()
        out = loads_(string=gstring, model=model)
        lin_tokens, adjacency_matrix = dfs_linearize(out[0])
        gstring = " ".join(lin_tokens)
    
    else:
        meta_lines  = []
        graph_lines = []
        for line in gstring.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('# ::'):
                meta_lines.append(line)
            elif line.startswith('#'):
                continue
            else:
                graph_lines.append(line)
        gstring = ' '.join(graph_lines)
        gstring = re.sub(' +', ' ', gstring)

        # In this case, it's basically a baseline model
        lin_tokens = None
        adjacency_matrix = None

    # return gstring
    return gstring, lin_tokens, adjacency_matrix

def tokenize_encoded_graph(encoded):
    linearized = re.sub(r"(\".+?\")", r" \1 ", encoded)
    pieces = []
    for piece in linearized.split():
        if piece.startswith('"') and piece.endswith('"'):
            pieces.append(piece)
        else:
            piece = piece.replace("(", " ( ")
            piece = piece.replace(")", " ) ")
            piece = piece.replace(":", " :")
            piece = piece.replace("/", " / ")
            piece = piece.strip()
            pieces.append(piece)
    linearized = re.sub(r"\s+", " ", " ".join(pieces)).strip()
    return linearized.split(" ")

def dfs_linearize(graph):
    graph_ = copy.deepcopy(graph)
    graph_.metadata = {}
    linearized = penman.encode(graph_)
    linearized_nodes = tokenize_encoded_graph(linearized)
    remap = {}
    for i in range(1, len(linearized_nodes)):
        nxt = linearized_nodes[i]
        lst = linearized_nodes[i - 1]
        if nxt == "/":
            remap[lst] = f"<pointer:{len(remap)}>"
    i = 1
    linearized_nodes_ = [linearized_nodes[0]]
    while i < (len(linearized_nodes)):
        nxt = linearized_nodes[i]
        lst = linearized_nodes_[-1]
        if nxt in remap:
            if lst == "(" and linearized_nodes[i + 1] == "/":
                nxt = remap[nxt]
                i += 1
            elif lst.startswith(":"):
                nxt = remap[nxt]
        linearized_nodes_.append(nxt)
        i += 1
    linearized_nodes = linearized_nodes_

    """
    Add: create graph on linearized nodes
    Will only be used for GNN
    """
    id2text = {ins.source: ins.target for ins in graph.instances()}
    id2position = {idx: get_position(linearized_nodes, text) for idx, text in id2text.items()}

    adjacency_matrix = np.zeros((len(linearized_nodes), len(linearized_nodes)))

    for e in graph.edges():
        node1_pos = id2position[e.source]
        node2_pos = id2position[e.target]
        # assert node1_pos < node2_pos
        try:
            edge_pos = get_position(linearized_nodes[node1_pos:], e.role) + node1_pos
        except:
            edge_pos = get_position(linearized_nodes[node1_pos:], e.role+'-of') + node1_pos
        adjacency_matrix[node1_pos][edge_pos] = 1
        adjacency_matrix[edge_pos][node2_pos] = 1
    
    for e in graph.attributes():
        node1_pos = id2position[e.source]
        node2_pos = get_position(linearized_nodes[node1_pos:], e.target) + node1_pos
        # assert node1_pos < node2_pos
        edge_pos = get_position(linearized_nodes[node1_pos:], e.role) + node1_pos
        adjacency_matrix[node1_pos][edge_pos] = 1
        adjacency_matrix[edge_pos][node2_pos] = 1

    # Visualization
    # G = GraphVisualization()
    # for i, row in enumerate(adjacency_matrix):
    #     for j, value in enumerate(row):
    #         if value:
    #             G.addEdge(str(i)+":"+linearized_nodes[i], str(j)+":"+linearized_nodes[j])
    # for i, value in enumerate(linearized_nodes):
        # G.addNode(str(i)+":"+value)
    # G.visualize()
    
    return linearized_nodes, adjacency_matrix

def get_position(linearized_nodes, token):
    return linearized_nodes.index(token)

class GraphVisualization:
    def __init__(self):
        # visual is a list which stores all 
        # the set of edges that constitutes a
        # graph
        self.visual = []
        self.visual_node = []
          
    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)
          
    def addNode(self, node_name):
        self.visual_node.append(node_name)
    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        if len(self.visual_node) > 0:
            G.add_nodes_from(self.visual_node)
        fig = plt.figure(1, figsize=(15, 15), dpi=60)
        nx.draw_networkx(G)
        plt.show()
        plt.savefig('fig1.png')

import argparse
import json
from collections import Counter, defaultdict
import os
import ipdb
import amrlib


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", type=str, required=True)
parser.add_argument("-o", "--output_path", type=str, required=True)
parser.add_argument("-b", "--batch_size", type=int, default=32)
args = parser.parse_args()

stog = amrlib.load_stog_model(device='cuda:0', batch_size=args.batch_size, num_beams=4)

inp = [json.loads(line) for line in open(args.input_path, 'r')]

with open(args.output_path, 'w') as f:
    counter = 0
    sents = []
    stored_doc = []
    for doc in inp:
        sents.append(doc['sentence'])
        stored_doc.append(doc)
        counter += 1
        if counter % args.batch_size == 0:
            graphs = stog.parse_sents(sents)
            # write file
            for graph, d in zip(graphs, stored_doc):
                d['amrgraph'] = graph
                f.write(json.dumps(d) + '\n')
            sents = []
            stored_doc = []
            print('Processed {} number of instances'.format(counter))
            counter = 0
    if len(sents) > 0:
        graphs = stog.parse_sents(sents)
        # write file
        for graph, d in zip(graphs, stored_doc):
            d['amrgraph'] = graph
            f.write(json.dumps(d) + '\n')
        print('Processed {} number of instances'.format(counter))
        counter = 0


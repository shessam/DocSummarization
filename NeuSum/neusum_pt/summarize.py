from __future__ import division
from rouge import Rouge
import neusum
import torch
import argparse
import math
import time
import logging
import numpy as np
# import collections
import hazm

logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
file_handler = logging.FileHandler(time.strftime("%Y%m%d-%H%M%S") + '.log.txt', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)


datahome = '/home/sam/Desktop/NLP_NEU/data/news_data'
exehome = '/home/sam/Desktop/NLP_NEU/code/NeuSum/neusum_pt'
savepath = datahome + '/models/neusum'

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-epochs', default=100,
                    help='Path to model .pt file')
parser.add_argument('-model', default=savepath + "/model_e{}.pt",
                    help='Path to model .pt file')
parser.add_argument('-src', default=datahome + "/test/test.txt.src.1",
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt', default=datahome + "/test/test.txt.tgt.1",
                    help='True target sequence (optional)')
parser.add_argument('-k', default=4,
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-output', default=datahome + '/test.src.txt.neusum_out',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-scores', default=datahome + '/test.src.txt.scores',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size', type=int, default=1,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=64,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=2000,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-verbose', action="store_true", default=True,
                    help='logger.info scores and predictions for each sentence')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-max_doc_len', type=int, default=50)
parser.add_argument('-max_decode_step', type=int, default=3)
parser.add_argument('-force_max_len', action="store_true", default=True)

parser.add_argument('-gpu', type=int, default=0,
                    help="Device to run on")


def reportScore(name, scoreTotal, wordsTotal):
    logger.info("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal / wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None


def addPair(f1, f2):
    for x, y1 in zip(f1, f2):
        yield (x, y1)
    yield (None, None)


def main():
    rouge = Rouge()

    opt = parser.parse_args()
    seq_length = opt.max_sent_length
    logger.info(opt)
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = neusum.Summarizer(opt, logger=logger)

    outF = open(opt.output + "_{}".format(opt.k), 'w', encoding='utf-8')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch = [], []
    src_raw, tgt_raw = [], []
    preds = []
    predIds = []
    predBatchs = []

    count = 0
    with open(opt.tgt + ".nums", 'r') as file0:
        tgtNums = [i.strip().split(" ") for i in file0.readlines()]

    for i in range(len(tgtNums)):
        for j in range(len(tgtNums[i])):
            tgtNums[i][j] = int(tgtNums[i][j])

    tgtF = open(opt.tgt) if opt.tgt else None

    for line in addone(open(opt.src, encoding='utf-8')):
        if line is not None:
            sline = line.strip()
            srcSents = sline.split('##SENT##')
            srcWords = [x.split(' ')[:seq_length] for x in srcSents]

            src_raw.append(srcSents)
            srcBatch.append(srcWords)

            # if tgtF:
            #     tgtTokens = tgtF.readline().split(' ') if tgtF else None
            #     tgtBatch += [tgtTokens]
            # tgt_raw.append(tgtWords)

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        predBatch, predId, predScore, goldScore = translator.translate(srcBatch, src_raw, None)
        preds += predBatch

        # print("----------------------------------")
        # print("total docs:", len(predBatch))
        # predScore = [int(float(score[0]) * 100) / 100 for score in predScore]
        # print("predBatch", predBatch, '\n', "predId", predId, '\n', "predScore", predScore)

        predIds = predIds + predId
        # predPrecisionTotal1 += 0
        predScoreTotal += sum(predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)

        normalizer = hazm.Normalizer()
        predBatch = [" ".join([normalizer.normalize(j) for j in i]) for i in predBatch]
        predBatchs = predBatchs + predBatch

        srcBatch, tgtBatch = [], []
        src_raw, tgt_raw = [], []

    print("----------------")
    # --- calc rouge score
    tgt = tgtF.readlines()
    tgt = [" ".join(i.strip().split("##SENT##")) for i in tgt]
    preds = [" ".join(i) for i in preds]
    rouge1, rouge2, rougeL = 0, 0, 0

    for i in range(len(tgt)):
        scores = rouge.get_scores(preds[i], tgt[i])[0]
        rouge1 += scores['rouge-1']['f'] / len(tgt)
        rouge2 += scores['rouge-2']['f'] / len(tgt)
        rougeL += scores['rouge-l']['f'] / len(tgt)

    print("total rouge1: {:.3f}".format(rouge1))
    print("total rouge2: {:.3f}".format(rouge2))
    print("total rougeL: {:.3f}".format(rougeL))

    # --- calc precision
    precs1 = 0
    precs2 = 0
    precs3 = 0
    predIds = [list(i[0]) for i in predIds]

    for i in range(len(predIds)):
        # print(predIds[i], tgtNums[i])
        num_intersection = len(set(predIds[i]) & set(tgtNums[i]))
        # print(num_intersection)
        if num_intersection > 3:
            num_intersection = 3
        precs3 += num_intersection / 3
        if num_intersection > 2:
            num_intersection = 2
        precs2 += num_intersection / 2
        if num_intersection > 1:
            num_intersection = 1
        precs1 += num_intersection
    precs1 = precs1 / len(tgtNums)
    precs2 = precs2 / len(tgtNums)
    precs3 = precs3 / len(tgtNums)

    # calc histogram
    predIds2 = []
    for i in predIds:
        if isinstance(i, list):
            for j in i:
                predIds2.append(j)
        else:
            predIds2.append(i)
    predIds = predIds2
    predIds = np.array(predIds)
    ids, counts = np.unique(predIds, return_counts=True)
    counts = np.array(counts[:30]) / len(predIds) * 100     # SELECTING 30 first sentence
    ids = np.array(ids[:30])   # SELECTING 30 first sentence
    # print(ids, counts)

    print("totalPrecision@1:\t{:.3f}".format(precs1))
    print("totalPrecision@2:\t{:.3f}".format(precs2))
    print("totalPrecision@3:\t{:.3f}".format(precs3))

    lines = ["totalRouge1:\t{:.3f}\n".format(rouge1),
             "totalRouge2:\t{:.3f}\n".format(rouge2),
             "totalRougeL:\t{:.3f}\n".format(rougeL),
             "totalPrecision@1:\t{:.3f}\n".format(precs1),
             "totalPrecision@2:\t{:.3f}\n".format(precs2),
             "totalPrecision@3:\t{:.3f}\n".format(precs3),
             ]

    for i in range(len(counts)):
        lines.append("{}\t{:.3f}\n".format(ids[i], counts[i]))

    with open(opt.scores + "_{}".format(opt.k), 'w', encoding='utf-8') as file0:
        file0.writelines(lines)

    normalizer = hazm.Normalizer()
    # ----- save final output
    # print(len(predBatchs))
    with open(opt.src, 'r') as file0:
        with open(opt.tgt, 'r') as file1:
            for i in range(len(predBatchs)):
                count += 1
                text = " ".join([normalizer.normalize(i) for i in file0.readline().strip().split("##SENT##")])
                outF.write('متن:\t{}'.format(text) + '\n')
                abstract = " ".join([normalizer.normalize(i) for i in file1.readline().strip().split("##SENT##")])
                outF.write('خلاصه اصلی:\t\t{}'.format(abstract) + '\n')
                outF.write('خلاصه تولید شده:\t\t{}'.format(predBatchs[i]) + '\n')
                outF.write('------------------------------------' + '\n')
                outF.flush()

    if tgtF:
        tgtF.close()


if __name__ == "__main__":
    main()

import os


epochs = 100
folds = 5
batch_size = 64
embedding_dim = 256
layers_num = 1


for k in range(folds):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("\t\t\t K = {} out of {}".format(k, folds))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    os.chdir("/home/sam/Desktop/NLP_NEU/data")

    # prepare the text dataset
    print("**** preparing text dataset ... ****")
    os.system("python3 make_datafiles.py {} {}".format(k, folds))
    # get oracle number
    print("**** calculating oracle number ... ****")
    os.chdir("cnndm_acl18-master/")
    os.system("python3 find_oracle_para.py")
    # get oracle gain number
    print("**** calculating oracle gain number ... ****")
    os.system("python3 get_mmr_regression_gain.py")
    # train the network
    print("**** Training the nework ... ****")
    os.chdir("../../code/NeuSum/neusum_pt")
    os.system("python3 train.py -epochs {} -batch_size {} -word_vec_size {} -layers {} -sent_enc_size {} -doc_enc_size {} -dec_rnn_size {} -sent_dropout {} -doc_dropout {}".format(epochs, batch_size, embedding_dim, layers_num, 256, 256, 256, 0.5, 0.4))
    # test the network
    print("**** testing the network ... ****")
    os.system("python3 summarize.py -k {} -epochs {} -batch_size {}".format(k, epochs, batch_size))

# printing the results
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
rouge1, rouge2, rougeL = 0, 0, 0
precs1 = 0
precs2 = 0
precs3 = 0

counts = {}
for k in range(folds):
    scores_home = '/home/sam/Desktop/NLP_NEU/data/news_data' + '/test.src.txt.scores' + "_{}".format(k)
    with open(scores_home, 'r') as file0:
        lines = [i.strip() for i in file0.readlines()]  # get scores from the file
        rouge1 += float(lines[0].split('\t')[1]) / folds
        rouge2 += float(lines[1].split('\t')[1]) / folds
        rougeL += float(lines[2].split('\t')[1]) / folds
        precs1 += float(lines[3].split('\t')[1]) / folds
        precs2 += float(lines[4].split('\t')[1]) / folds
        precs3 += float(lines[5].split('\t')[1]) / folds
        for k in range(6, len(lines)):
            counts[int(lines[k].split('\t')[0])] = float(lines[k].split('\t')[1])

print("total {}-fold rouge1: {:.3f}".format(folds, rouge1))
print("total {}-fold rouge2: {:.3f}".format(folds, rouge2))
print("total {}-fold rougeL: {:.3f}".format(folds, rougeL))
print("total {}-fold precision@1: {:.3f}".format(folds, precs1))
print("total {}-fold precision@2: {:.3f}".format(folds, precs2))
print("total {}-fold precision@3: {:.3f}".format(folds, precs3))
print(counts)

# -------- concat all text files --------
lines = []
for i in range(folds):
    with open("/home/sam/Desktop/NLP_NEU/data/news_data/test.src.txt.neusum_out_{}".format(i), 'r') as file0:
        lines = lines + file0.readlines()

with open("/home/sam/Desktop/NLP_NEU/data/news_data/test.src.txt.neusum_out", 'w') as file0:
    file0.writelines(lines)

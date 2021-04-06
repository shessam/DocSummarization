import argparse

try:
    import ipdb
except ImportError:
    pass

datahome = '/home/sam/Desktop/NLP_NEU/data/news_data'
exehome = '/home/sam/Desktop/NLP_NEU/code/NeuSum/neusum_pt'

num_of_docs_per_line = '.1'


def add_data_options(parser):
    # Data options
    parser.add_argument('-save_path', default=datahome + '/models/neusum')

    # tmp solution for load issue

    parser.add_argument('-online_process_data', action="store_true", default=True)
    parser.add_argument('-process_shuffle', action="store_true", default=False)
    parser.add_argument('-train_src', default=datahome + '/train/train.txt.src' + num_of_docs_per_line)
    parser.add_argument('-src_vocab', default=datahome + '/train/vocab')
    parser.add_argument('-train_tgt', default=datahome + '/train/train.txt.tgt' + num_of_docs_per_line)
    parser.add_argument('-tgt_vocab', default=datahome + '/train/vocab')
    parser.add_argument('-train_oracle', default=datahome + '/train/train.rouge_bigram_F1.oracle' + num_of_docs_per_line + '.regGain')
    parser.add_argument('-train_src_rouge', default=datahome + '/train/train.rouge_bigram_F1.oracle' + num_of_docs_per_line + '.regGain')
    parser.add_argument('-max_doc_len', type=int, default=50)     # max doc per line

    # Test options
    parser.add_argument('-dev_input_src')  # , default=datahome + '/dev/val.txt.src.shuffle.4k')
    parser.add_argument('-dev_ref')  # , default=datahome + '/dev/val.txt.tgt.shuffle.4k')
    parser.add_argument('-beam_size', type=int, default=1, help='Beam size')
    parser.add_argument('-max_sent_length', type=int, default=2000)  # max words in a sentence
    parser.add_argument('-max_decode_step', type=int, default=3)
    parser.add_argument('-force_max_len', action="store_true", default=True)


def add_model_options(parser):
    # Model options
    parser.add_argument('-layers', type=int, default=1,
                        help='Number of layers in the LSTM encoder/decoder')
    parser.add_argument('-sent_enc_size', type=int, default=256,
                        help='Size of LSTM hidden states')
    parser.add_argument('-doc_enc_size', type=int, default=256,
                        help='Size of LSTM hidden states')
    parser.add_argument('-dec_rnn_size', type=int, default=256,
                        help='Size of LSTM hidden states')
    parser.add_argument('-word_vec_size', type=int, default=50,
                        help='Word embedding sizes')
    parser.add_argument('-att_vec_size', type=int, default=256,
                        help='Concat attention vector sizes')
    parser.add_argument('-maxout_pool_size', type=int, default=2,
                        help='Pooling size for MaxOut layer.')
    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")
    # parser.add_argument('-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")
    parser.add_argument('-sent_brnn', action='store_true', default=True,
                        help='Use a bidirectional sentence encoder')
    parser.add_argument('-doc_brnn', action='store_true', default=True,
                        help='Use a bidirectional document encoder')
    parser.add_argument('-brnn_merge', default='concat',
                        help="""Merge action for the bidirectional hidden states:
                        [concat|sum]""")
    parser.add_argument('-use_self_att', action='store_true',
                        help='Use self attention in document encoder.')
    parser.add_argument('-self_att_size', type=int, default=512)
    parser.add_argument('-norm_lambda', type=int, default=20,
                        help='The scale factor for normalizing the ROUGE regression score. exp(ax)/sum(exp(ax))')
    parser.add_argument('-dec_init', default='simple',
    help = 'Sentence encoder type: simple | att')


def add_train_options(parser):
    # Optimization options
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
    parser.add_argument('-optim', default='adam',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-max_weight_value', type=float, default=15,
                        help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-sent_dropout', type=float, default=0.3,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-doc_dropout', type=float, default=0.2,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-dec_dropout', type=float, default=0,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-curriculum', type=int, default=0,
                        help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
    parser.add_argument('-extra_shuffle', action="store_true", default=True,
    help = """By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                    epoch""")
    parser.add_argument('-start_eval_batch', type=int, default=1000,
                        help="""evaluate on dev per x batches.""")
    parser.add_argument('-eval_per_batch', type=int, default=1000,
                        help="""evaluate on dev per x batches.""")
    parser.add_argument('-halve_lr_bad_count', type=int, default=100000,
                        help="""evaluate on dev per x batches.""")

    # pretrained word vectors
    parser.add_argument('-pre_word_vecs_enc',  # default=datahome + '/glove/glove.6B.50d.txt',
                        help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
    parser.add_argument('-freeze_word_vecs_enc', action="store_true", default=True,
                        help="""Update encoder word vectors.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

    # GPU
    parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")

    parser.add_argument('-log_interval', type=int, default=100,
                        help="logger.info stats at this interval.")

    parser.add_argument('-seed', type=int, default=123456,
                        help="""Random seed used for the experiments
                    reproducibility.""")
    parser.add_argument('-cuda_seed', type=int, default=123456,
                        help="""Random CUDA seed used for the experiments
                    reproducibility.""")

    parser.add_argument('-log_home', default=datahome + '/models/neusum',
                        help="""log home""")

    parser.add_argument('-dump_epoch_checkpoint', action="store_true", default=True)

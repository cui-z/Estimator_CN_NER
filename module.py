import tensorflow  as tf
import pickle
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood


tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }

class InputSample():
    def __init__(self,input_text,labels,seq_len):
        self.input_text = input_text
        self.labels = labels
        self.seq_len = seq_len

def get_word2id(word2id_path):
    with open(word2id_path, 'rb') as fr:
        word2id = pickle.load(fr)
    return word2id

def sent2id(sent,word2id):
    result = []
# 数据特征可以在这进行处理
    for char in sent:
        # 判断是否是数字
        if char.isdigit():
            char = "<NUM>"
        #判断是都为英文
        elif ('\u0041' <= char <= '\u005a') or ('\u0061' <= char <= '\u007a'):
            char = '<ENG>'
        if char not in word2id:
            char = '<UNK>'
        result.append(word2id[char])
    return result

def labels2num(labels):
    return [ tag2label[l] for l in labels]

def pad_sequeneces(text,max_len):
    pass

def create_model(input_ids,labels,sequence_lengths,vocab_size,embedding_dim,lstm_dim,dropout_pl):

    with tf.variable_scope("words_embedding"):
        embedding_table = tf.get_variable(
            name="word_embedding",
            shape=[vocab_size, embedding_dim],
            initializer=tf.truncated_normal_initializer(stddev=0.02),dtype=tf.float32)
        word_embedding = tf.nn.embedding_lookup(embedding_table,input_ids,"words_embedding")
        print("word_embedding  {0}".format(word_embedding.shape))
    with tf.variable_scope("bi-lstm"):
        cell_fw = LSTMCell(lstm_dim)
        cell_bw = LSTMCell(lstm_dim)
        (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=word_embedding,
            sequence_length=sequence_lengths,
            dtype=tf.float32)
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        output = tf.nn.dropout(output, dropout_pl)
        print("output   {0}".format(output.shape))
    with tf.variable_scope("proj"):
        num_tags =len(tag2label)
        W = tf.get_variable(name="W",
                            shape=[2 * lstm_dim, num_tags],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)
        b = tf.get_variable(name="b",
                            shape=[num_tags],
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)
        # 获取维度
        s = output.shape.as_list()
        output = tf.reshape(output, [-1, 2 * lstm_dim])
        pred = tf.matmul(output, W) + b
        logits = tf.reshape(pred, [-1, s[1], num_tags])
    with tf.variable_scope("crf"):
        #labels = tf.cast(labels,tf.float32)
        # log_likelihood 可以计算loss
        log_likelihood, transition_params = crf_log_likelihood(inputs=logits,
                                                                    tag_indices=labels,
                                                                    sequence_lengths=sequence_lengths)
        loss = -tf.reduce_mean(log_likelihood)

        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(logits, transition_params, sequence_lengths)

    return loss,viterbi_sequence,viterbi_score


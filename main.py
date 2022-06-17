import tensorflow as tf
import os
import collections
from module import InputSample, sent2id, labels2num, get_word2id, create_model,tag2label
from tf_metrics import precision,recall,f1

def get_samples(input_path, word2id_path):
    word2id = get_word2id(word2id_path)
    input_result = []
    text = []
    labels = []

    for line in open(input_path, encoding='utf-8'):
        if line != "\n":
            [char, label] = line.strip().split()
            text.append(char)
            labels.append(label)
        else:
            input_result.append(InputSample(sent2id(text, word2id), labels2num(labels), [len(text)]))
            text = []
            labels = []

    max_len = max([len(ir.input_text) for ir in input_result])
    return input_result, max_len


def examples_to_features_tf_record(examples, output_file, max_len, is_training):
    # 定义一个TFRecordWriter，用来写TFRecords文件
    writer = tf.python_io.TFRecordWriter(output_file)

    # 进行类型转换 接收参数是一个list
    # 有三种类型 tf.train.BytesList(value=[value])
    # tf.train.FloatList(value=[value])
    # tf.train.Int64List(value=[value])
    def create_float_feature(values):
        if is_training:
            values = values[:max_len] + [0] * max(max_len - len(values), 0)
        f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return f

    def create_int_feature(values):
        if is_training:
            values = values[:max_len] + [0] * max(max_len - len(values), 0)
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    for example in examples:
        # 用一个字典来存储
        features = collections.OrderedDict()
        features["text"] = create_int_feature(example.input_text)
        features["labels"] = create_int_feature(example.labels)
        features["seq_len"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(example.seq_len)))
        # 定义一个example
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        # serializaToSreing  序列化  写入
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, batch_size, is_training, max_len):
    # 定义一个特征词典，和写TFRecords时的特征词典相对应
    name_to_features = {
        "text": tf.FixedLenFeature([max_len], tf.int64),
        "labels": tf.FixedLenFeature([max_len], tf.int64),
        "seq_len": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        # 根据上面的特征解析单个数据（保存时的单个Example）
        example = tf.parse_single_example(record, name_to_features)
        return example

    def input_fn(params):
        """The actual input function."""
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        # input_file 可以是个list 包含多个tf_record
        # 使用TFRecordDataset即可以作为tensorflow代码所写的模型的输入，也可以作为keras模型的输入
        d = tf.data.TFRecordDataset(input_file)
        # 可以让评估停下里
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size))
        """
        d = tf.data.TFRecordDataset(input_file)
        d = d.map(_decode_record).batch(4).repeat() 这样也可
        """
        return d

    return input_fn


def my_model(features, labels, mode, params):
    text = features["text"]
    labels = features["labels"]
    seq_len = features["seq_len"]
    loss, viterbi_sequence, viterbi_score = create_model(text, labels, seq_len, params["vocab_size"],
                                                         params["embedding_dim"], params["lstm_dim"],
                                                         params["dropout_pl"])
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, name="train", global_step=tf.train.get_global_step())
        output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(seq_length, max_len, label_ids, pred_ids,num_labels):
            indices = [1, 2, 3, 4, 5, 6]  # indice参数告诉评估矩阵评估哪些标签
            # Metrics
            weights = tf.sequence_mask(seq_length, maxlen=max_len)
            metrics = {
                'acc': tf.metrics.accuracy(label_ids, pred_ids, weights),
                'precision': precision(label_ids, pred_ids, num_labels, indices, weights),
                'recall': recall(label_ids, pred_ids, num_labels, indices, weights),
                'f1': f1(label_ids, pred_ids, num_labels, indices, weights),
            }
            for metric_name, op in metrics.items():
                tf.summary.scalar(metric_name, op[1])
            return metrics

        viterbi_sequence = tf.cast(viterbi_sequence,tf.int64)
        eval_metrics = metric_fn(seq_len,params["max_len"],labels, viterbi_sequence,len(tag2label))
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
        output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=viterbi_sequence)
    return output_spec


def main():
    to_do = "train"
    train_path = "./data/train_data"
    word2id_path = "./data/word2id.pkl"
    train_batch_size = 128
    num_train_epochs = 6
    train_record_path = "./data/train_data.tf_record"

    vocab_size = len(get_word2id(word2id_path))
    train_samples, max_len = get_samples(train_path, word2id_path)


    params = {"vocab_size": vocab_size, "embedding_dim": 128, "lstm_dim": 300, "dropout_pl": 0.5,"max_len":max_len}
    run_config = tf.estimator.RunConfig(model_dir='output_dir', save_checkpoints_steps=10, )
    estimator = tf.estimator.Estimator(model_fn=my_model, config=run_config, params=params)



    if to_do == "train":

        num_train_steps = int(len(train_samples) / train_batch_size * num_train_epochs)
        print("num_train_steps:{0}".format(num_train_steps))
        print("max_len_seq:{0}".format(max_len))
        examples_to_features_tf_record(train_samples, train_record_path, max_len, is_training=True)
        train_input_fn = file_based_input_fn_builder(train_record_path, train_batch_size, is_training=True,
                                                     max_len=max_len)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        to_do = "eval"
    if to_do == "eval":
        eval_batch_size = 128
        eval_path = "./data/test_data"
        eval_output_path = "./data/test_data.tf_record"
        test_samples, test_max_len = get_samples(eval_path, word2id_path)
        examples_to_features_tf_record(test_samples, eval_output_path, max_len, is_training=True)
        eval_input_fn = file_based_input_fn_builder(eval_output_path, eval_batch_size, is_training=False,
                                                     max_len=max_len)
        result = estimator.evaluate(eval_input_fn,)
        for key in sorted(result.keys()):
            print(" {0} = {1}".format(key, str(result[key])))
    if to_do == "predict":
        eval_batch_size = 1
        eval_path = "./data/test_data"
        eval_output_path = "./data/test_data.tf_record"
        test_samples, test_max_len = get_samples(eval_path, word2id_path)
        examples_to_features_tf_record(test_samples, eval_output_path, max_len, is_training=True)
        eval_input_fn = file_based_input_fn_builder(eval_output_path, eval_batch_size, is_training=False,
                                                    max_len=max_len)
        result = estimator.predict(eval_input_fn, )
        for re in result:
            print(re)







if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np
import pandas as pd
import re,os,json
from collections import Counter,OrderedDict
from itertools import chain

shuffle_buffer_size=1000
save_checkpoints_steps=1000
model_dir='./output'

params = {
        "embedding_size": 200,
        "max_seq_len": 100,
        "filter_num": 100,
        "filter_sizes": [2, 3, 4],
        "batch_size": 1024,
        "drop_prob": 0.25,
        "class_num": 14,
        "epoch": 3,
        "lr": 0.001,
        "pad_word": '<PAD>',
        "unk_word": '<UNK>'
    }

class DataProcessor():
    def __init__(self,path_vocab,path_train,params):
        self.path_vocab = path_vocab
        self.path_train = path_train
        self.params = params

        if not os.path.exists(self.path_vocab):
            self.build_vocab()
        self.vocab = self.load_vocab()
        self.params['vocab_size'] = len(self.vocab)


    def convert_text_to_tokens(self,text):
        text = re.sub(r"[^A-Za-z0-9\'\`]", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\`", "\'", text)
        text = text.strip().lower()

        tokens = text.split(' ')
        tokens = [t.strip("'") for t in tokens if len(t.strip("'")) > 0]
        return tokens

    def post_process_of_tokenization(self,tokens):
        # repalce out-of-vocabulary words by <UNK> and add <PAD> to
        # make tokens list to a fix length
        for index, value in enumerate(tokens):
            if value not in self.vocab:
                tokens[index] = self.params['unk_word']
        n = len(tokens)
        if n > self.params['max_seq_len']:
            tokens = tokens[:self.params['max_seq_len']]
        if n < self.params['max_seq_len']:
            tokens += [self.params['pad_word']] * (self.params['max_seq_len'] - n)
        assert len(tokens) == self.params['max_seq_len']
        return tokens

    def convert_tokens_to_ids(self,token_list):
        return [self.vocab[t] for t in token_list]

    def process_text(self,text):
        tokens = self.convert_text_to_tokens(text)
        tokens = self.post_process_of_tokenization(tokens)
        feature = self.convert_tokens_to_ids(tokens)
        return feature

    def parse_single_line(self,line):
        label = int(line[0]) - 1
        text = line[2]
        tokens = self.convert_text_to_tokens(text)
        tokens = self.post_process_of_tokenization(tokens)
        feature = self.convert_tokens_to_ids(tokens)
        return feature, label

    def build_tfrecord_file(self,path_csv, path_output):
        print('Building tfrecord file...')
        writer = tf.python_io.TFRecordWriter(path_output)

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        df = pd.read_csv(path_csv, header=None)
        df = df.sample(frac=1)
        examples = df.values
        for ex_index, example in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
            feature, label = self.parse_single_line(example)
            features = OrderedDict()
            features['input_ids'] = create_int_feature(feature)
            features['label_ids'] = create_int_feature([label])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

        writer.close()

    def build_vocab(self):
        print('Building vocabulary...')

        df = pd.read_csv(self.path_train,header=None)
        df[2] = df[2].apply(lambda x: self.convert_text_to_tokens(x))
        all_words = list(chain(* (list(df[2].values))))

        counter = Counter(all_words)
        # count_pairs = counter.most_common(params['vocab_size'] - 1)
        # words, _ = list(zip(*count_pairs))
        words = list(counter.keys())
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = [self.params['pad_word'],self.params['unk_word']] + list(words)
        with open(self.path_vocab, mode='w') as f:
            f.write('\n'.join(words) + '\n')

    def load_vocab(self):
        print("Loading vocabulary from disk...")
        vocab = OrderedDict()
        index = 0
        with open(self.path_vocab,'r') as f:
            for line in f.readlines():
                if len(line.strip())>0:
                    vocab[line.strip()] = index
                    index += 1
        return vocab

    def load_labels(self,path_labels):
        with open(path_labels) as f:
            labels = json.load(f)
        labels_dict = {}
        for k,v in enumerate(labels):
            labels_dict[k] = v
        return labels, labels_dict


def create_model(features,vocab_size,embedding_size,filter_sizes,
                 filter_num,drop_prob,max_seq_len,is_training,class_num):

    x = features['input_ids']
    embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                 shape=[vocab_size, embedding_size])

    # shape [batch, sentence_length, embedding_size]
    net = tf.nn.embedding_lookup(embeddings, x)

    # add a channel dim, required by the conv2d and max_pooling2d method
    # shape [batch, sentence_length, embedding_size, 1]
    net = tf.expand_dims(net, -1)

    pooled_outputs = []
    for filter_size in filter_sizes:
        # shape [batch, max_sentence_len-filter_size+1, 1, filter_num]
        conv = tf.layers.conv2d(net, filters=filter_num,
                                kernel_size=[filter_size, embedding_size],
                                strides=(1, 1), padding='VALID', activation=tf.nn.relu)

        # shape [batch, 1, 1, filter_num]
        pool = tf.layers.max_pooling2d(conv, pool_size=[max_seq_len - filter_size + 1, 1],
                                       strides=(1, 1), padding='VALID')

        pooled_outputs.append(pool)

    # shape [batch, 1, 1, filter_num*len(filter_sizes) ]
    pooled = tf.concat(pooled_outputs, 3)
    # shape [batch, filter_num*len(filter_sizes) ]
    pooled_flat = tf.reshape(pooled, [-1, filter_num * len(filter_sizes)])

    if is_training and drop_prob > 0.0:
        pooled_flat = tf.layers.dropout(pooled_flat, drop_prob)

    logits = tf.layers.dense(pooled_flat, class_num, activation=None)
    return logits


def model_fn_builder(learning_rate):

    # Args:
    #
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn,
    #           see e.g. train_input_fn for these two.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   User-defined hyper-parameters, e.g. learning-rate.

    def model_fn(features,labels,mode,params):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        is_evaluate = mode == tf.estimator.ModeKeys.EVAL
        is_predict = mode == tf.estimator.ModeKeys.PREDICT

        if not is_predict:
            # labels = tf.reshape(labels, [-1, 1])
            tf.logging.info('labels shape:' + str(labels.shape))

        logits = create_model(features,params['vocab_size'],params['embedding_size'],
                              params['filter_sizes'],params['filter_num'],params['drop_prob'],
                              params['max_seq_len'],is_training,params['class_num'])

        tf.logging.info('logits shape:' + str(logits.shape))

        y_pred = tf.nn.softmax(logits)
        tf.logging.info('y_pred shape:'+ str(y_pred.shape))

        y_pred_cls = tf.argmax(y_pred,axis=1)
        tf.logging.info('y_pred_cls shape:' + str(y_pred_cls.shape))

        if not is_predict:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                           logits=logits)
            tf.logging.info('cross_entropy shape:' + str(cross_entropy.shape))

            loss = tf.reduce_mean(cross_entropy)
            tf.logging.info('loss shape:' + str(loss.shape))

            def metric_fn(labels, predictions):
                '''
                define metrics
                '''
                accuracy, accuracy_update = tf.metrics.accuracy(labels=labels, predictions=predictions,
                                                                name='text_accuracy')
                return {'accuracy': (accuracy, accuracy_update)}
            if is_evaluate:
                return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=metric_fn(labels,y_pred_cls))

            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss,
                                                                                    global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,
                                              eval_metric_ops=metric_fn(labels,y_pred_cls))
        else:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred_cls)

    return model_fn



def input_fn(path_tfr, is_training):

    dataset = tf.data.TFRecordDataset(path_tfr)

    name_to_features = {
        "input_ids":tf.FixedLenFeature([params['max_seq_len']],tf.int64),
        "label_ids":tf.FixedLenFeature([],tf.int64)
    }
    def decode_record(record,name_to_features):
        example = tf.parse_single_example(record,name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(params['epoch'])

    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        lambda record: decode_record(record, name_to_features),
        batch_size= params['batch_size']))
    iterator = dataset.make_one_shot_iterator()
    one_ele = iterator.get_next()
    features, labels = {"input_ids":one_ele['input_ids']}, one_ele['label_ids']

    return  features,labels

def serving_input_receiver_fn():

    input_ph = tf.placeholder(tf.int64,shape=[None,params['max_seq_len']])
    receiver_tensors = {"text_ids":input_ph}
    features = {"input_ids":input_ph}

    return tf.estimator.export.ServingInputReceiver(features,receiver_tensors)




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    tf.logging.set_verbosity(tf.logging.INFO)



    path_train = './data/dbpedia_csv/train.csv'
    path_train_tfr = './data/dbpedia_csv/train.tfrecord'
    path_test = './data/dbpedia_csv/test.csv'
    path_test_tfr = './data/dbpedia_csv/test.tfrecord'
    path_vocab = './data/dbpedia_csv/vocab.txt'

    processor = DataProcessor(path_vocab,path_train,params)


    if not os.path.exists(path_train_tfr):
        processor.build_tfrecord_file(path_train,path_train_tfr)
    if not os.path.exists(path_test_tfr):
        processor.build_tfrecord_file(path_test,path_test_tfr)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn_builder(params['lr']),
        params=params,
        config=tf.estimator.RunConfig(model_dir=model_dir,
                                       save_checkpoints_steps=save_checkpoints_steps)
    )

    input_fn_train = lambda : input_fn(path_train_tfr,True)
    input_fn_test = lambda : input_fn(path_test_tfr,False)


    print('start train and eval...')
    df_test = pd.read_csv(path_test,header=None)
    eval_step = int(len(df_test) / params['batch_size'])
    classifier.train(input_fn=input_fn_train)
    r = classifier.evaluate(input_fn=input_fn_test,steps=eval_step)
    print(r)
    classifier.export_savedmodel('export_model',serving_input_receiver_fn =serving_input_receiver_fn)

    # f,l = input_fn(path_train_tfr,True)
    #
    #
    # with tf.Session() as sess:
    #     for i in range(3):
    #         fe,la = sess.run([f,l])
    #         print("feature",fe)
    #         print("label",la)


    # df_train = pd.read_csv(path_train,header=None,nrows=2)
    # df_train[0] = df_train[0].apply(lambda x:x-1)
    #
    # for sentence in df_train[2].values:
    #     print (sentence)
    #     t = convert_text_to_tokens(sentence)
    #     t = post_process_of_tokenization(t,vocab)
    #     ids = convert_tokens_to_ids(t,vocab)
    #     print(ids)
    #












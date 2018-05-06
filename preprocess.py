# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import multiprocessing
import json

from gensim.models import Doc2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.doc2vec import TaggedDocument
import numpy as np

MAX_SIZE = 125


class TextLoader(object):

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        text = open(self.path, encoding='utf-8')
        length = 0
        for index, line in enumerate(text):
            print(index)
            js_obj = json.loads(line)
            sentences = js_obj['doc'].split('\n')
            for sentence in sentences:
                length += 1
                yield TaggedDocument(sentence.split(' '), [str(length)])


def export_label(path):
    text = open(path)
    writer = open('data/cnn_labels.txt', 'w')
    for index, line in enumerate(text):
        js_obj = json.loads(line)
        labels = js_obj['labels'].split('\n')
        for label in labels:
            writer.write(label + '\n')
    writer.close()


def compute_len():
    with open('data/cnn_val.json', encoding='utf-8') as f:
        max_lenth = 0
        for line in f.readlines():
            doc = json.loads(line)
            labels = doc['labels'].split('\n')
            docsize = len(labels)
            if docsize > max_lenth:
                max_lenth = docsize
                print(max_lenth)

        print('MAX_SIZE : ', max_lenth)


def label_process():
    fout = open('data/label_preprocess.npy', 'wb')
    outlist = []
    doc_length = []
    with open('data/cnn_val.json', encoding='utf-8') as f:
        for line in f.readlines():
            doc = json.loads(line)
            labels = doc['labels'].split('\n')
            docsize = len(labels)
            for i in range(MAX_SIZE - docsize):
                labels.append('0')
            outline = np.array(labels).astype(np.float).tolist()
            outlist.append(outline)
            doc_length_one = np.zeros(MAX_SIZE)
            doc_length_one[:docsize] = 1
            doc_length.append(doc_length_one)
    np.save('data/label_preprocess.npy', np.array(outlist))
    np.save('data/doc_len.npy',np.array(doc_length))
    fout.close()


def vector_process():
    fout = open('data/vector_preprocess.npz', 'wb')

    with open('data/cnn_val.json', encoding='utf-8') as f1:
        with open('data/cnn_doc2vec_100.vector') as f2:
            vectors = f2.readlines()
            cursor = 0
            res1 = []
            res2 = []
            res3 = []
            i = 0
            for line in f1.readlines():
                doc = json.loads(line)
                labels = doc['labels'].split('\n')
                docsize = len(labels)
                outlines = vectors[cursor:cursor + docsize]
                outlines = [x.split('\t')[1].strip().split(' ') for x in outlines]
                outlines_array = np.array(outlines).astype(np.float)
                padding = np.zeros([MAX_SIZE - docsize, 100], np.float)
                onedoc = np.vstack([outlines_array, padding]).tolist()
                if i <= 800000:
                    res1.append(onedoc)
                elif i <= 1600000:
                    res2.append(onedoc)
                else:
                    res3.append(onedoc)
                print(cursor, ' ', cursor + docsize)
                cursor += docsize
                i += 1
            res1 = np.array(res1)
            res2 = np.array(res2)
            res3 = np.array(res3)
            np.savez(fout, res1=res1, res2=res2, res3=res3)

    fout.close()


# if __name__ == "__main__":
#     export_label('/data/cnn_val.json')


if __name__ == "__main__":
    label_process()

# if __name__ == '__main__':
#     program = os.path.basename(sys.argv[0])
#     logger = logging.getLogger(program)
#
#     logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
#     logging.root.setLevel(level=logging.INFO)
#     logger.info("running %s" % ' '.join(sys.argv))
#
#     path = 'data/'
#     output_doc2vector_model = os.path.join(path, 'cnn_doc2vec_model.txt')
#     output_word2vector_100 = os.path.join(path, 'cnn_word2vec_100.vector')
#     output_doc2vector_100 = os.path.join(path, 'cnn_doc2vec_100.vector')
#     model = Doc2Vec(TextLoader('data/cnn_val.json'), size=100,
#                     window=5, min_count=0, workers=multiprocessing.cpu_count(), dm=0,
#                     hs=0, negative=10, dbow_words=1, iter=10)
#
#     model.save(output_doc2vector_model)  # save dov2vec model
#     model.wv.save_word2vec_format(output_word2vector_100, binary=False)  # save word2vec向量
#     outid = open(output_doc2vector_100, 'w')
#     print("doc2vecs length:", len(model.docvecs))
#     for id in range(len(model.docvecs)):
#         outid.write(str(id) + "\t")
#         for idx, lv in enumerate(model.docvecs[id]):
#             outid.write(str(lv) + " ")
#         outid.write("\n")
#
#     outid.close()

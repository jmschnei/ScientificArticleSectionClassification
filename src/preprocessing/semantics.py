from sentence_transformers import SentenceTransformer, util
import csv
import argparse
import logging


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # multi-language model

labels = ['Abstarct',
          'Introdution',
          'Related Work',
          'Methods',
          'Results',
          'Evaluation',
          'Discussion',
          'Summary/Conclusions',
          'Acknowledgements',
          'References'
]

sentences = []
dict_label = {}

logger = logging.getLogger()

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def load_sentences():
    global sentences
    global dict_label    
    # TODO read CSV file
    with open('ScientificArticleSectionClassification/data/ACL-section-annotations.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            # extract texts and labels
            # add them to sentences and dict_labels
            for l in labels:
                if not row[l]==None:
                    t = row[l].strip()
                    if not t=='':
                        if not t=='-':
                            if not t in sentences:
                                sentences.append(t)
                            if not t in dict_label:
                                dict_label[t] = l

load_sentences()

'''
sentences = [
    'what is the weather tomorrow',
    'will it rain tomorrow',
    'Will the weather be hot in the future',
    'what time is it',
    'could you help me translate this setence',
    'play some jazz music'
]

dict_label = {
    'what is the weather tomorrow':'intro',
    'will it rain tomorrow':'method',
    'Will the weather be hot in the future':'related',
    'what time is it':'other1',
    'could you help me translate this setence':'other2',
    'play some jazz music':'conclusion'
}
'''

def load_table():
    # First we have to load the table from the CSV file
    load_sentences()
    embedding = model.encode(sentences, convert_to_tensor=False)
    #print(embedding.shape)
    return embedding

table_embeddings = load_table()

def getSemanticLabel(t):
    label = None
    logger.info('Semantic Label search for "%s"' % t)

    sentences2 = [t]
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(table_embeddings, embeddings2)
    
    #for i in range(len(sentences)):
    #    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences2[0], cosine_scores[i][0]))

    d = {}
    for i in range(len(sentences)):
        #print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences2[0], cosine_scores[i][0]))
        d[sentences[i]] = cosine_scores[i][0].item()
    d_sorted = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
    #print(d_sorted)

    label = dict_label[list(d_sorted.keys())[0]]
    #print(label)

    logger.info('DONE with label "%s"' % label)
    return label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #section names classification
    parser.add_argument("-text", default='', type=str)

    args = parser.parse_args()
    init_logger()
    getSemanticLabel(args.text)
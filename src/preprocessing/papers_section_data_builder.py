import json
import os
import shutil

import logging

import spacy
import pickle

from semantics import *

##########################################################################################################

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

def generate_grouped_titles(args):
    init_logger(args.log_file)

    nlp = spacy.load("en_core_web_sm")

    save_path = args.save_path
    save_file_TITLES = save_path+'/'+args.titles_filename
    #save_file_DATA = save_path+'/'+args.data_filename
    save_file_NUM = save_path+'/'+args.num_papers_filename

    logger.info('Load %s' % save_file_TITLES)   
    with open(save_file_TITLES, 'rb') as handle:
        titles = json.load(handle)
    logger.info('Load %s' % save_file_NUM)
    with open(save_file_NUM, 'rb') as handle:
        titles = json.load(handle)

    titles2 = dict()
    for t in titles:
        if titles[t]/num_papers>0.01:
            #print(t, ': ', titles[t])
            doc = nlp(t)
            if len(doc)==1:
                #print(t, ': ', titles[t])
                lemma = doc[0].lemma_
                if lemma in titles2:
                    titles2[lemma].append(doc[0].text)
                else:
                    titles2[lemma] = [doc[0].text]
            else:
                contain = False
                lemma = ''
                for token in doc:
                    #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop)
                    if token.lemma_ in titles2:
                        contain=True
                        break
                    if not token.is_stop:
                        lemma = lemma + ' ' + token.lemma_
                if contain:
                    titles2[token.lemma_].append(t)
                else:
                    titles2[lemma.strip()] = [t]
                    #titles2[lemma] = doc[0].text
    logger.info('Total Grouped Titles : %i ' % (len(titles2)))
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    
    save_file_TITLES_CLEAN = save_path+'/'+args.titles_clean_filename    
    logger.info('Saving to %s' % save_file_TITLES_CLEAN)   
    with open(save_file_TITLES_CLEAN, 'wb') as handle:
        pickle.dump(titles2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('DONE')


def print_titles_with_percentaje(args):
    init_logger(args.log_file)

    save_path = args.save_path
    save_file_TITLES = save_path+'/'+args.titles_filename
    save_file_DATA = save_path+'/'+args.data_filename
    save_file_NUM = save_path+'/'+args.num_papers_filename

    with open(save_file_TITLES, 'r') as handle:
        titles = json.load(handle)
    with open(save_file_NUM, 'r') as handle:
        num_papers = json.load(handle)

    lower = int(args.lower_percentage)
    logger.info('Number of papers: %i' % num_papers)
    if lower<50:
        logger.info('-----50%-----')
        titles_level1 = titles
        #print(titles_level1)
        for t in titles_level1:
            if titles_level1[t]/num_papers>0.5:
                logger.info('%s : %s ' % (t, titles_level1[t]))
    if lower<25:
        logger.info('-----25%-----')
        for t in titles_level1:
            if titles_level1[t]/num_papers>0.25 and titles_level1[t]/num_papers<0.5:
                logger.info('%s : %s ' % (t, titles_level1[t]))
    if lower<10:
        logger.info('-----10%-----')
        for t in titles_level1:
            if titles_level1[t]/num_papers>0.10 and titles_level1[t]/num_papers<0.25:
                logger.info('%s : %s ' % (t, titles_level1[t]))
    if lower<5:
        logger.info('-----5%-----')
        for t in titles_level1:
            if titles_level1[t]/num_papers>0.05 and titles_level1[t]/num_papers<0.10:
                logger.info('%s : %s ' % (t, titles_level1[t]))
    if lower<1:
        logger.info('-----1%-----')
        for t in titles_level1:
            if titles_level1[t]/num_papers>0.01 and titles_level1[t]/num_papers<0.05:
                logger.info('%s : %s ' % (t, titles_level1[t]))

    logger.info('DONE')  

classification = {
    'introduction':['introduction'],
    'conclusion':['conclusion', 'conclusions'],
    'acknowledgment':['acknowledgments'],
    'background':['background'],
    'discussion':['discussion'],
    'material':['materials and methods'],
    'result':['results', 'results and discussion', 'experimental results'],
    'method':['methods', 'materials and methods', 'patients and methods', 'statistical methods'],
    'outcome':['outcome'],
    'statistic':['statistics'],
    'data':['data collection'],
    'implementation':['implementation'],
    'requirement':['availability and requirements'],
    'assessment':['assessments'],
    'material':['materials', 'materials and method'],
    'model':['dominant model', 'recessive model'],
    'evaluation':['evaluation'],
    'analysis':['analysis', 'phylogenetic analyses', 'data analysis', 'statistical analyses'],
    'appendix':['appendix']
}

classified_titles = dict()
nlp = spacy.load("en_core_web_sm")

def load_classified_titles(args):
    init_logger(args.log_file)
    save_path = args.save_path
    save_file_CLASSIFIED = save_path+'/'+args.classified_titles_filename

    global classified_titles
    logger.info('Loading classified titles from %s' % save_file_CLASSIFIED)
    if args.use_classified_titles:
        try:
            with open(save_file_CLASSIFIED, 'rb') as handle:
                classified_titles = json.load(handle)
        except:
            classified_titles = dict()
    else:
        classified_titles = dict()
    logger.info('DONE')


def save_classified_titles(args):
    init_logger(args.log_file)
    save_path = args.save_path
    save_file_CLASSIFIED = save_path+'/'+args.classified_titles_filename

    global classified_titles
    logger.info('Saving classified titles to %s' % save_file_CLASSIFIED)   
    if args.use_classified_titles:
        with open(save_file_CLASSIFIED, 'w+') as save:
            save.write(json.dumps(classified_titles))

    logger.info('DONE')


def getLabel(t):
    #logger.info('Label search for "%s"' % t)
    if t in classified_titles:
        return classified_titles[t]
    else:
        doc = nlp(t)
        if len(doc)==1:
            #print(t, ': ', titles[t])
            lemma = doc[0].lemma_
            if lemma in classification:
                classified_titles[t]=lemma
                logger.info('DONE with label-1 "%s"' % lemma)
                return lemma
            else:
                pass
        else:
            contain = False
            label = ''
            for token in doc:
                #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop)
                if token.lemma_ in classification:
                    contain=True
                    #label = token.lemma_
                    label = label + ',' + token.lemma_
                    break
                #if not token.is_stop:
                #    label = label + ',' + token.lemma_
            if contain:
                classified_titles[t] = label
                logger.info('DONE with label-2 "%s"' % label)
                return label
            else:
                pass
        # We have to manage the semantic similarity
        label = getSemanticLabel(t)
        classified_titles[t] = label


def getDataFromPMC(args):
    init_logger(args.log_file)
    folder = args.folder_path
    num_papers = 0
    titles=dict()
    data = []
    for filename in os.listdir(folder)[1:200]:
        #print(filename)
        if filename.endswith('.txt'):
            num_papers += 1
            #print(folder + filename)
            try:
                with open(os.path.join(folder, filename)) as f:
                    lines = f.readlines()
            except:
                continue
            #print('-----LINES-----')
            inBody = False
            full_text = ''
            for l in lines:
                if l.strip()=='==== Refs':
                    inBody = False
                if inBody:
                    if l.strip()=='':
                        pass
                    elif len(l.strip().split())<5:
                        #data.append({'label':head_text,'title':head_text, 'text':full_text})
                        head_text = l.strip().lower()
                        if head_text in titles:
                            titles[head_text] = titles[head_text] + 1
                        else:
                            titles[head_text] = 1
                        full_text = ''
                    else:
                        pass
                        #full_text = full_text + ' ' l
                        #print('\t'+l)
                if l.strip()=='==== Body':
                    inBody = True
            #print('-----LINES END-----')
            #if not full_text.strip()=='':
            #  data.append({'label':head_text,'title':head_text, 'text':full_text})

    logger.info('Total Titles : %i ' % (len(titles)))
    logger.info('Total Data : %i ' % (len(data)))
    logger.info('Num Papers : %i ' % (num_papers))
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    
    save_file_TITLES = save_path+'/'+args.titles_filename
    save_file_DATA = save_path+'/'+args.data_filename
    save_file_NUM = save_path+'/'+args.num_papers_filename
    
    logger.info('Saving to %s' % save_file_TITLES)   
    with open(save_file_TITLES, 'w+') as save:
        # save.write('\n'.join(dataset))
        save.write(json.dumps(titles))
    logger.info('DONE')

    logger.info('Saving to %s' % save_file_DATA)
    with open(save_file_DATA, 'w+') as save:
        # save.write('\n'.join(dataset))
        save.write(json.dumps(data))
    logger.info('DONE')

    logger.info('Saving to %s' % save_file_NUM)
    with open(save_file_NUM, 'w+') as save:
        # save.write('\n'.join(dataset))
        save.write(json.dumps(num_papers))
    logger.info('DONE')


def generateLabeledDataFromPMC(args):
    init_logger(args.log_file)
    load_classified_titles(args)
    folder = args.folder_path
    num_papers = 0
    titles=dict()
    data = []
    for filename in os.listdir(folder)[1:50]:
        #print(filename)
        if filename.endswith('.txt'):
            num_papers += 1
            #print(folder + filename)
            try:
                with open(os.path.join(folder, filename)) as f:
                    lines = f.readlines()
            except:
                continue
            #print('-----LINES-----')
            inBody = False
            full_text = ''
            label = ''
            head_text = None
            for l in lines:
                if l.strip()=='==== Refs':
                    inBody = False
                if inBody:
                    if l.strip()=='':
                        pass
                    elif len(l.strip().split())<5:
                        if not head_text==None:
                            data.append({'label':label,'title':head_text, 'text':full_text})
                        head_text = l.strip().lower()
                        label = getLabel(head_text)
                        if head_text in titles:
                            titles[head_text] = titles[head_text] + 1
                        else:
                            titles[head_text] = 1
                        full_text = ''
                    else:
                        full_text = full_text + ' ' + l
                        #print('\t'+l)
                if l.strip()=='==== Body':
                    inBody = True
            #print('-----LINES END-----')
            #if not full_text.strip()=='':
            #  data.append({'label':head_text,'title':head_text, 'text':full_text})

    logger.info('Total Titles : %i ' % (len(titles)))
    logger.info('Total Data : %i ' % (len(data)))
    logger.info('Num Papers : %i ' % (num_papers))
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    
    save_file_TITLES = save_path+'/'+args.titles_filename
    save_file_DATA = save_path+'/'+args.data_filename
    save_file_NUM = save_path+'/'+args.num_papers_filename
    
    logger.info('Saving to %s' % save_file_TITLES)   
    with open(save_file_TITLES, 'w+') as save:
        save.write(json.dumps(titles))
    logger.info('DONE')

    logger.info('Saving to %s' % save_file_DATA)
    with open(save_file_DATA, 'w+') as save:
        save.write(json.dumps(data))
    logger.info('DONE')

    logger.info('Saving to %s' % save_file_NUM)
    with open(save_file_NUM, 'w+') as save:
        save.write(json.dumps(num_papers))
    logger.info('DONE')

    save_classified_titles(args)
    logger.info('DONE')
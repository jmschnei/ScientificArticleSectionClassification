#encoding=utf-8
import argparse
import time
import os
import papers_section_data_builder

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #section names classification
    parser.add_argument("-mode", default='getDataFromPMC', type=str)
    parser.add_argument("-sn_dic_path", default='')
    parser.add_argument("-raw_path", default='')
    parser.add_argument("-save_path", default='saved_classified_files')
    parser.add_argument('-dataset', default='pubmed')
    parser.add_argument('-log_file', default='')
    
    parser.add_argument("-folder_path", default='/Users/julianmorenoschneider/Downloads/PMC000')

    parser.add_argument("-titles_filename", default='titles.json')
    parser.add_argument("-data_filename", default='data.json')
    parser.add_argument("-num_papers_filename", default='num_papers.json')
    parser.add_argument("-classified_titles_filename", default='classified_titles.json')
    parser.add_argument("-use_classified_titles", default=True)

    parser.add_argument('-lower_percentage', default='5')

    #############################################################################################
    parser.add_argument("-base_LM", default='bert-base', type=str, choices=['bert-base', 'bert-large', 'roberta-base','longformer-base-4096','longformer-large-4096','bigbird-pegasus-large-arxiv','bigbird-pegasus-large-pubmed'])
    parser.add_argument("-temp_dir", default='temp')
    parser.add_argument("-corenlp_path", default='', type=str)#
    parser.add_argument("-vocab_file", default='', type=str)#
    parser.add_argument("-obtain_tok_se", type=str2bool, nargs='?',const=True,default=True)#
    
    parser.add_argument("-summ_select_mode", default='greedy', type=str)
    parser.add_argument("-summ_size", default=3, type=int)#
    
    parser.add_argument("-map_path", default='../../data/')
    
    parser.add_argument("-tok_sent_path", default='../../data/')#
    parser.add_argument("-tok_para_path", default='../../data/')#
    parser.add_argument("-histruct_path", default='../../data/')#

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=3, type=int)#0,3
    parser.add_argument('-max_src_nsents', default=0, type=int)#1000,0 means no limit of max
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)#0,5
    parser.add_argument('-max_src_ntokens_per_sent', default=0, type=int)#2000, 0 means no limit of max
    parser.add_argument('-min_tgt_ntokens', default=5, type=int)#0,5
    parser.add_argument('-max_tgt_ntokens', default=0, type=int)#5000, 0 means no limit of max

    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)#
    parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)#
    parser.add_argument('-n_cpus', default=1, type=int)
    #section names encoding (pubmed&arxiv)
    parser.add_argument("-sn_embed_comb_mode", default='sum', type=str, choices=['sum', 'mean'])
    ##bigbird_pegasus
    parser.add_argument("-is_encoder_decoder", type=str2bool, nargs='?',const=False,default=False)
    args = parser.parse_args()

    #init_logger(args.log_file)
    
    if args.dataset=='arxiv' or args.dataset=='pubmed':
        print('papers_section_data_builder.'+args.mode + '(args)')
        eval('papers_section_data_builder.'+args.mode + '(args)')       

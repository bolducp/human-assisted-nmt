import os
from hnmt.feedback_requester.data import run_source_sent_embeddings, run_final_preprocessing

current_dir = os.path.dirname(os.path.realpath(__file__))

saved_nmt_out_file = '/nmt/nmt_out_sample.p' #make sure this matches locally saved file
sent_embeds_file = current_dir + '/preprocessing_outputs/source_sent_embeddings_sample.p'

# Must do these 2 separately, with different BERT-as-a-service server settings!
#
# 1. First, make sure that the BERT-as-a-service server is running with pooling_strategy (default is REDUCE_MEAN)
#    e.g. "bert-serving-start -model_dir multi_cased_L-12_H-768_A-12/ -num_worker=4 -max_seq_len=NONE"
#    then generate and save the source sentence embeddings to use in the subsequent preprocessing
# run_source_sent_embeddings(saved_nmt_out_file, sent_embeds_file)

# 2. Change the BERT-as-a-service server to run with pooling_strategy set to NONE
#    e.g. "bert-serving-start -model_dir multi_cased_L-12_H-768_A-12/ -pooling_strategy=NONE -num_worker=4 -max_seq_len=NONE"
run_final_preprocessing(saved_nmt_out_file, sent_embeds_file,  current_dir + '/preprocessing_outputs/final_out_sample.p')
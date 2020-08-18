import os
from hnmt.feedback_requester.data import run_source_sent_embeddings, run_final_preprocessing

current_dir = os.path.dirname(os.path.realpath(__file__))

saved_nmt_out_file = '/nmt/nmt_out_sample.p' #make sure this matches locally saved file
sent_embeds_file = current_dir + '/preprocessing_outputs/source_sent_embeddings.p'

run_source_sent_embeddings(saved_nmt_out_file, sent_embeds_file)
run_final_preprocessing(saved_nmt_out_file, sent_embeds_file,  current_dir + '/preprocessing_outputs/final_out_sample.p')
import os
import sys
from typing import List, Tuple
import sentencepiece as spm
import sacrebleu
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from hnmt.utils import calculate_effort
from hnmt.nmt.main import get_document_nmt_output
from hnmt.feedback_requester.data import generate_source_sent_embeddings, generate_word_piece_sequential_input, generate_word_piece_sequential_input_for_inference
from hnmt.feedback_requester.model import LSTMClassifier
from hnmt.feedback_requester.data import NMTOutputDataset, collate_pad_fn, prediction_collate_pad_fn

current_dir = os.path.dirname(os.path.realpath(__file__))

POST_FEEDBACK_STRUCT = Tuple[torch.Tensor, int, str, str] # [(model_pred, was_asked, hypo_str, final_str)]


def main(
    model_weights_path: str,
    sent_piece_model_path: str,
    threshold: float,
):
    model = LSTMClassifier(1586, 1586)
    model.load_state_dict(torch.load(model_weights_path))
    optimizer = optim.Adam(model.parameters())

    tokenizer = spm.SentencePieceProcessor(model_file=sent_piece_model_path)

    def document_feedback_interaction():
        model.eval()

        print("\n\n\n\nPaste your document here, one sentence per line. When finished, type Crtl+D.")
        document_to_translate = sys.stdin.read().split("\n")
        sents_to_translate = [sent for sent in document_to_translate if sent]

        tokenized_doc = [" ".join(tokenizer.encode(sent, out_type=str))
                                for sent in sents_to_translate]

        print("\n\n\n\nGenerating NMT system suggestions. Please be patient.")

        nmt_output = get_document_nmt_output(tokenized_doc)
        source_sent_embeds = generate_source_sent_embeddings(nmt_output)
        final_output = generate_word_piece_sequential_input_for_inference(
                                                            sents_to_translate,
                                                            nmt_output,
                                                            source_sent_embeds)

        dataloader = DataLoader(final_output,
                                batch_size=32,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=prediction_collate_pad_fn,
                                pin_memory=True)

        nmt_output = [x[2] for x in final_output]

        print("\nStarting feedback-requestor predictions.")

        post_interactive = [] # [(model_pred, was_asked, hypo_str, final_str)]
        document_effort = 0


        for batch in dataloader:
            predictions = model(batch[0]).squeeze()

            for i, prediction in enumerate(predictions):
                nmt_hypo_sent = batch[1][i]

                if prediction >= threshold:
                    print("\nSource:", batch[2][i])
                    print("Translation:", nmt_hypo_sent)
                    user_correction = input("\nPlease correct or press 'Enter' if no change is needed.\n")

                    if not user_correction:
                        user_correction = nmt_hypo_sent

                    sent_effort_score = calculate_effort(nmt_hypo_sent, user_correction)
                    document_effort += sent_effort_score
                    post_interactive.append((prediction, 1, nmt_hypo_sent, user_correction))

                else:
                    post_interactive.append((prediction, 0, nmt_hypo_sent, nmt_hypo_sent))

        return post_interactive

    interacting = True

    while interacting:
        post_interactive = document_feedback_interaction()
        posted_edited = post_editing(post_interactive)
        update_model(model, optimizer, post_interactive, posted_edited)

        print("\n\nModel updated.\n")

        save_weights = input("To save the model updated model weights, type 's'. Otherwise, enter any key.\n")
        if save_weights.lower() == 's':
            torch.save(model.state_dict(), current_dir + "/saved_state_dicts/updated.pt")
            print("\nModel weights saved at {}.\n".format(current_dir + "/saved_state_dicts/updated.pt"))

        translate_new_doc = input("\nTo translate another document enter 'y', to end enter 'n'.\n")
        if translate_new_doc.lower() == 'n':
            interacting = False


def post_editing(
    post_interactive: List[POST_FEEDBACK_STRUCT]
):
    print("Here is the final document:\n")

    for line in post_interactive:
        print(line[3])

    print("\nMake futher changes and re-submit, line by line.\n")
    print("When finished correcting, type Ctrl+D.\n")

    posted_edited_doc = [sent for sent in sys.stdin.read().split("\n") if sent]
    assert len(post_interactive) == len(posted_edited_doc)
    return posted_edited_doc


def update_model(
        model: LSTMClassifier,
        optimizer,
        post_interactive: List[POST_FEEDBACK_STRUCT],
        post_edited: List[str]
    ) -> float:
    model.train()
    optimizer.zero_grad()
    loss = calculate_post_edited_loss(post_interactive, post_edited)
    # print("Doc loss", loss)
    loss.backward()
    optimizer.step()
    return loss.item()


def calculate_post_edited_loss(
    post_interactive: List[POST_FEEDBACK_STRUCT],
    post_edited: List[str]
):
    loss = torch.tensor(0)

    for i in range(len(post_interactive)):
        model_pred = post_interactive[i][0]
        nmt_hypo = post_interactive[i][2]
        final_sent = post_edited[i]

        if was_asked(post_interactive[i]):
            chrf_score = torch.tensor(sacrebleu.sentence_chrf(nmt_hypo, [final_sent]).score)
            loss = loss + (model_pred * chrf_score) + ((1 - model_pred) * (1 - chrf_score))
        elif was_post_edited(post_interactive[i], post_edited[i]):
            chrf_score = torch.tensor(sacrebleu.sentence_chrf(nmt_hypo, [final_sent]).score)
            loss = loss + ((1 - model_pred) * (1 - chrf_score))
        else:
            loss = loss + model_pred
    return loss


def was_asked(
    post_interactive: POST_FEEDBACK_STRUCT
) -> bool:
    return post_interactive[1] == 1


def was_post_edited(
    post_interactive: POST_FEEDBACK_STRUCT,
    post_edited: str
) -> bool:
    return post_interactive[3] != post_edited


if __name__ == "__main__":
    MODEL_WEIGHTS_PATH = current_dir + "/saved_state_dicts/epoch_9.pt"
    SENT_PIECE_MODEL = '/Users/paigefink/human-assisted-nmt/hnmt/nmt/corpus/enja_spm_models/spm.ja.nopretok.model'

    main(MODEL_WEIGHTS_PATH, SENT_PIECE_MODEL, 0.5)

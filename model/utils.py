from typing import List, Iterator
import logging
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min

def sequence_mask(sequence_lengths: torch.LongTensor, max_len=None) -> torch.tensor:
    """
    Create a sequence mask that masks out all indices larger than some sequence length as defined by
    sequence_lengths entries.
    :param sequence_lengths: [batch_size] sequence lengths per example in batch
    :param max_len: int defining the maximum sequence length in the batch
    :return: [batch_size, max_len] boolean mask
    """
    if max_len is None:
        max_len = sequence_lengths.data.max()
    batch_size = sequence_lengths.size(0)
    sequence_range = torch.arange(0, max_len).long().to(device=device)

    # [batch_size, max_len]
    sequence_range_expand = sequence_range.unsqueeze(0).expand(batch_size, max_len)

    # [batch_size, max_len]
    seq_length_expand = (sequence_lengths.unsqueeze(1).expand_as(sequence_range_expand))

    # [batch_size, max_len](boolean array of which elements to include)
    return sequence_range_expand < seq_length_expand



def masked_softmax(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1, memory_efficient: bool = True,
) -> torch.Tensor:
    """
    `torch.nn.functional.softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular softmax.
    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If `memory_efficient` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and `memory_efficient` is false, this function
    returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (
                result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
            )
        else:
            masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result



def log_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Total parameters: %d" % n_params)
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.info("%s : %s" % (name, list(p.size())))


def sequence_accuracy(prediction: List[int], target: List[int]) -> float:
    correct = 0
    total = 0
    prediction = prediction.copy()
    target = target.copy()
    if len(prediction) < len(target):
        difference = len(target) - len(prediction)
        prediction.extend([0] * difference)
    if len(target) < len(prediction):
        difference = len(prediction) - len(target)
        target.extend([-1] * difference)
    for i, target_int in enumerate(target):
        if i >= len(prediction):
            break
        prediction_int = prediction[i]
        if prediction_int == target_int:
            correct += 1
        total += 1
    if not total:
        return 0.
    return (correct / total) * 100

def array_to_sentence(self, sentence_array, vocab):
    return[vocab.itos[word_idx] for word_idx in sentence_array]


def predict_and_save(dataset_iterator, model, output_file_path, max_decoding_steps, pad_idx, sos_idx, eos_idx, input_vocab, target_vocab,
                    max_testing_examples=None,  **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (x, output_sequence, attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target) in predict(
                dataset_iterator, model=model, max_decoding_steps=max_decoding_steps,
                pad_idx=pad_idx, sos_idx=sos_idx,
                eos_idx=eos_idx):

                i += 1
                input_sequence = x.input
                target_sequence = x.target

                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                input_str_sequence = array_to_sentence(input_sequence[0].tolist(), vocab=input_vocab)
                input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                target_str_sequence = array_to_sentence(target_sequence[0].tolist(), vocab=target_vocab)
                target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = array_to_sentence(output_sequence, vocabulary="target")
                output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                               #"derivation": derivation_spec,
                               "target": target_str_sequence, 
                               #"situation": situation_spec,
                               "attention_weights_input": attention_weights_commands,
                               "attention_weights_situation": attention_weights_situations,
                               "accuracy": accuracy,
                               "exact_match": True if accuracy == 100 else False,
                               "position_accuracy":  auxiliary_accuracy_target})
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path


def predict(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for x in data_iterator:
        i += 1
        if max_examples_to_evaluate and i > max_examples_to_evaluate: break

        encoded_input = model.encode_input(x.input, x.situation)

        # For efficiency
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]
        
        output_sequence = []
        contexts_situation = []
        #\TODO: this initialize hidden state from encoder output.
        hidden = model.attention_decoder.initialize_hidden(
            model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))

        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []

        while token != eos_idx and decoding_iteration <= max_decoding_steps:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
                input_lengths=x.input, encoded_situations=projected_keys_visual)
            output = F.log_softmax(output, dim=-1)
            token = output.max(dim=-1)[1]
            output_sequence.append(token.data[0].item())
            attention_weights_commands.append(attention_weights_command.tolist())
            attention_weights_situations.append(attention_weights_situation.tolist())
            contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            attention_weights_commands.pop()
            attention_weights_situations.pop()

        if model.auxiliary_task:
            target_position_scores = model.auxiliary_task_forward(torch.cat(contexts_situation, dim=1).sum(dim=1))
            auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, x.target) #\TODO x.target add ground truth position information.
        else:
            auxiliary_accuracy_target = 0
            auxiliary_accuracy_agent = 0
            #\TODO we never even predict aux_acc_agent
        yield (x, output_sequence, attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)
        # yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
        #        attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))
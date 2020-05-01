import argparse
import logging
import os


import torch
import torchtext as tt
import torchtext.data as data
from torch.optim.lr_scheduler import LambdaLR

from model.utils import *
from model.config import cfg
from model.model import GSCAN_model

# import models
def dataloader(data_path, batch_size=32, device=torch.device('cpu'), fix_length=None, input_vocab=None, target_vocab=None):
    INPUT_FIELD = tt.data.Field(sequential=True, include_lengths=True, batch_first=True, fix_length=fix_length)
    TARGET_FIELD = tt.data.Field(sequential=True, include_lengths=True, batch_first=True, is_target=True,
                                 fix_length=fix_length)
    SITUATION_FIELD = tt.data.RawField(postprocessing=lambda x: torch.FloatTensor(x)) if device == torch.device('cpu') \
        else tt.data.RawField(postprocessing=lambda x: torch.cuda.FloatTensor(x))
    dataset = tt.data.TabularDataset(path=data_path, format="json",
                                     fields={'input': ('input', INPUT_FIELD),
                                             'target': ('target', TARGET_FIELD),
                                             'situation': ('situation', SITUATION_FIELD)}
                                     )
    if input_vocab is None:
        INPUT_FIELD.build_vocab(dataset)
    else:
        INPUT_FIELD.vocab = input_vocab
    if target_vocab is None:
        TARGET_FIELD.build_vocab(dataset)
    else:
        TARGET_FIELD.vocab = target_vocab
    iterator = tt.data.Iterator(dataset, batch_size=batch_size)
    return iterator, INPUT_FIELD.vocab, TARGET_FIELD.vocab

def evaluate():
    accuracies = []
    target_accuracies = []
    exact_math = 0
    for input_sequence, _, _, output_sequence, target_sequence, _, _, aux_acc_target in predict(
            data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps, pad_idx=pad_idx,
            sos_idx=sos_idx, eos_idx=eos_idx, max_examples_to_evaluate=max_examples_to_evaluate):
        accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
        if accuracy == 100:
            exact_match += 1
        accuracies.append(accuracy)
        target_accuracies.append(aux_acc_target)
    return (float(np.mean(np.array(accuracies))), (exact_match / len(accuracies)) * 100,
            float(np.mean(np.array(target_accuracies))))



def train(train_data_path, val_data_path):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    

    logger.info("Loading Training set...")
    train_iter, train_input_vocab, train_target_vocab = dataloader(train_data_path, 
                                                                    batch_size=cfg.TRAIN.BATCH_SIZE) #\TODO add k and statistics and shuffling
    pad_idx, sos_idx, eos_idx = train_target_vocab.stoi['<pad>'], train_target_vocab.stoi['<sos>'], train_target_vocab.stoi['<eos>']

    train_input_vocab_size, train_target_vocab_size = len(train_input_vocab.itos), len(train_target_vocab.itos)

    #\TODO decide whether to add sos token and eos token to each input command/target label
    # Each data item dimension
    '''
    Input (command) [0]: batch_size x max_cmd_len       [1]: batch_size x 0 (len for each cmd)
    Situation: batch_size x grid x grid x feat_size
    Target (action) [0]: batch_size x max_action_len    [1]: batch_size x 0 (len for each action sequence)

    max_cmd_len = 6, max_action_len = 16
    '''
    # training_set = GroundedScanDataset(data_path, data_directory, split="train",
    #                                    input_vocabulary_file=input_vocab_path,
    #                                    target_vocabulary_file=target_vocab_path,
    #                                    generate_vocabulary=generate_vocabularies, k=k)
    # training_set.read_dataset(max_examples=max_training_examples,
    #                           simple_situation_representation=simple_situation_representation)
    logger.info("Done Loading Training set.")
    # logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    # logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    # logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    # logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    # logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))


    # I think this part is basically saving generated vocabs.
    # if generate_vocabularies:
    #     training_set.save_vocabularies(input_vocab_path, target_vocab_path)
    #     logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    logger.info("Loading Dev. set...")
    val_iter, val_input_vocab, val_target_vocab = dataloader(val_data_path, 
                                                            batch_size=cfg.TRAIN.BATCH_SIZE) #\TODO add k and statistics
    
    val_input_vocab_size, val_target_vocab_size = len(val_input_vocab.itos), len(val_target_vocab.itos)
    # val_set = GroundedScanDataset(data_path, data_directory, split="dev",  # TODO: use dev set here
    #                                input_vocabulary_file=input_vocab_path,
    #                                target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0)
    # val_set.read_dataset(max_examples=None,
    #                       simple_situation_representation=simple_situation_representation)

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.
    
    #val_set.shuffle_data()
    logger.info("Done Loading Dev. set.")

    model = GSCAN_model(pad_idx, eos_idx, train_input_vocab_size, train_target_vocab_size)

    # model = GSCAN_model(input_vocabulary_size=training_set.input_vocabulary_size,
    #               target_vocabulary_size=training_set.target_vocabulary_size,
    #               num_cnn_channels=training_set.image_channels,
    #               input_padding_idx=training_set.input_vocabulary.pad_idx,
    #               target_pad_idx=training_set.target_vocabulary.pad_idx,
    #               target_eos_idx=training_set.target_vocabulary.eos_idx,
    #               **cfg)


    model = model.cuda() if use_cuda else model

    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.TRAIN.SOLVER.LR, betas=(cfg.TRAIN.SOLVER.ADAM_BETA1, cfg.TRAIN.SOLVER.ADAM_BETA2))
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: cfg.TRAIN.SOLVER.LR_DECAY ** (t / cfg.TRAIN.SOLVER.LR_DECAY_STEP))

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_accuracy = 0
    best_exact_match = 0
    best_loss = float('inf')

    if cfg.RESUME_FROM_FILE:
        assert os.path.isfile(cfg.RESUME_FROM_FILE), "No checkpoint found at {}".format(cfg.RESUME_FROM_FILE)
        logger.info("Loading checkpoint from file at '{}'".format(cfg.RESUME_FROM_FILE))
        optimizer_state_dict = model.load_model(cfg.RESUME_FROM_FILE)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(cfg.RESUME_FROM_FILE, start_iteration))

    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < cfg.TRAIN.MAX_EPOCH: # iterations here actually means "epoch"

        # Shuffle the dataset and loop over it.
        # training_set.shuffle_data() \TODO add reshuffle option in iterator
        for x in train_iter:
            is_best = False
            model.train()
            target_scores, target_position_scores = model(x.input, x.situation, x.target) #\TODO: model does not output target position prediction
            
            loss = model.get_loss(target_scores, x.target[0])


            target_loss = 0
            if cfg.AUXILIARY_TASK:
                target_loss = model.get_auxiliary_loss(target_position_scores, x.target)#\TODO x.target currently does not include ground truth target position
            loss += cfg.TRAIN.WEIGHT_TARGET_LOSS * target_loss

        # for (input_batch, input_lengths, _, situation_batch, _, target_batch,
        #      target_lengths, agent_positions, target_positions) in training_set.get_data_iterator(
        #         batch_size=training_batch_size):
            # is_best = False
            # model.train()

            # Forward pass.
            # target_scores, target_position_scores = model(commands_input=input_batch, commands_lengths=input_lengths,
            #                                               situations_input=situation_batch, target_batch=target_batch,
            #                                               target_lengths=target_lengths)
            # loss = model.get_loss(target_scores, target_batch)
            # if auxiliary_task:
            #     target_loss = model.get_auxiliary_loss(target_position_scores, target_positions)
            # else:
            #     target_loss = 0
            # loss += weight_target_loss * target_loss

            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.update_state(is_best=is_best)

            # Print current metrics.
            if training_iteration % cfg.PRINT_EVERY == 0:
                accuracy, exact_match = model.get_metrics(target_scores, x.target[0])
                if cfg.AUXILIARY_TASK:
                    auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, x.target)#\TODO add ground truth target position into x.target
                else:
                    auxiliary_accuracy_target = 0.
                learning_rate = scheduler.get_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, accuracy %5.2f, exact match %5.2f, learning_rate %.5f,"
                            " aux. accuracy target pos %5.2f" % (training_iteration, loss, accuracy, exact_match,
                                                                 learning_rate, auxiliary_accuracy_target))
            
            raise NotImplementedError
            # Evaluate on test set.
            if training_iteration % flags.evaluate_every == 0:
                with torch.no_grad():
                    model.eval()
                    logger.info("Evaluating..")
                    accuracy, exact_match, target_accuracy = evaluate()

                    # accuracy, exact_match, target_accuracy = evaluate(
                    #     test_set.get_data_iterator(batch_size=1), model=model,
                    #     max_decoding_steps=max_decoding_steps, pad_idx=test_set.target_vocabulary.pad_idx,
                    #     sos_idx=test_set.target_vocabulary.sos_idx,
                    #     eos_idx=test_set.target_vocabulary.eos_idx,
                    #     max_examples_to_evaluate=kwargs["max_testing_examples"])


                    logger.info("  Evaluation Accuracy: %5.2f Exact Match: %5.2f "
                                " Target Accuracy: %5.2f" % (accuracy, exact_match, target_accuracy))

                    if exact_match > best_exact_match:
                        is_best = True
                        best_accuracy = accuracy
                        best_exact_match = exact_match
                        model.update_state(accuracy=accuracy, exact_match=exact_match, is_best=is_best)
                    file_name = "checkpoint.{}th.tar".format(str(training_iteration))
                    if is_best:
                        model.save_checkpoint(file_name=file_name, is_best=is_best,
                                              optimizer_state_dict=optimizer.state_dict())

            training_iteration += 1
            if training_iteration > flags.max_training_iterations:
                break
    logger.info("Finished training.")


    

def main(flags):
    #print(flags)

    # for arg, val in flags.items():
    #     logger.info("{} : {}".format(arg, val))
    #\TODO enable argparser
    
    
    
    if not os.path.exists(cfg.OUTPUT_DIRECTORY):
        os.mkdir(os.path.join(os.getcwd(), cfg.OUTPUT_DIRECTORY))

    
    
    # Some checks on the flags
    if cfg.GENERATE_VOCABULARIES:
        assert cfg.INPUT_VOCAB_PATH and cfg.TARGET_VOCAB_PATH, "Please specify paths to vocabularies to save."



    train_data_path = os.path.join(cfg.DATA_DIRECTORY, "dev.json")
    val_data_path = os.path.join(cfg.DATA_DIRECTORY, "dev.json") #\TODO val dataset not exist

    if cfg.MODE == "train":
        train(train_data_path=train_data_path, val_data_path=val_data_path)


    # elif flags["mode"] == "test":
    #     assert os.path.exists(os.path.join(flags["data_directory"], flags["input_vocab_path"])) and os.path.exists(
    #         os.path.join(flags["data_directory"], flags["target_vocab_path"])), \
    #         "No vocabs found at {} and {}".format(flags["input_vocab_path"], flags["target_vocab_path"])
    #     splits = flags["splits"].split(",")
    #     for split in splits:
    #         logger.info("Loading {} dataset split...".format(split))
    #         test_set = GroundedScanDataset(data_path, flags["data_directory"], split=split,
    #                                        input_vocabulary_file=flags["input_vocab_path"],
    #                                        target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False,
    #                                        k=flags["k"])
    #         test_set.read_dataset(max_examples=None,
    #                               simple_situation_representation=flags["simple_situation_representation"])
    #         logger.info("Done Loading {} dataset split.".format(flags["split"]))
    #         logger.info("  Loaded {} examples.".format(test_set.num_examples))
    #         logger.info("  Input vocabulary size: {}".format(test_set.input_vocabulary_size))
    #         logger.info("  Most common input words: {}".format(test_set.input_vocabulary.most_common(5)))
    #         logger.info("  Output vocabulary size: {}".format(test_set.target_vocabulary_size))
    #         logger.info("  Most common target words: {}".format(test_set.target_vocabulary.most_common(5)))

    #         model = Model(input_vocabulary_size=test_set.input_vocabulary_size,
    #                       target_vocabulary_size=test_set.target_vocabulary_size,
    #                       num_cnn_channels=test_set.image_channels,
    #                       input_padding_idx=test_set.input_vocabulary.pad_idx,
    #                       target_pad_idx=test_set.target_vocabulary.pad_idx,
    #                       target_eos_idx=test_set.target_vocabulary.eos_idx,
    #                       **flags)
    #         model = model.cuda() if use_cuda else model

    #         # Load model and vocabularies if resuming.
    #         assert os.path.isfile(flags["resume_from_file"]), "No checkpoint found at {}".format(flags["resume_from_file"])
    #         logger.info("Loading checkpoint from file at '{}'".format(flags["resume_from_file"]))
    #         model.load_model(flags["resume_from_file"])
    #         start_iteration = model.trained_iterations
    #         logger.info("Loaded checkpoint '{}' (iter {})".format(flags["resume_from_file"], start_iteration))
    #         output_file_name = "_".join([split, flags["output_file_name"]])
    #         output_file_path = os.path.join(flags["output_directory"], output_file_name)
    #         output_file = predict_and_save(dataset=test_set, model=model, output_file_path=output_file_path, **flags)
    #         logger.info("Saved predictions to {}".format(output_file))
    
    
    elif cfg.MODE == "predict":
        raise NotImplementedError()
    
    
    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(cfg.MODE))

    



if __name__ == "__main__":
    FORMAT = "%(asctime)-15s %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                        datefmt="%Y-%m-%d %H:%M")
    logger = logging.getLogger(__name__)
    use_cuda = True if torch.cuda.is_available() else False
    logger.info("Initialize logger")

    if use_cuda:
        logger.info("Using CUDA.")
        logger.info("Cuda version: {}".format(torch.version.cuda))

    


    parser = argparse.ArgumentParser(description="LGCN models for GSCAN")
    args = parser.parse_args()

    

    main(args)

'''
Based on the lasagne lstm text generation example.
Original written by @keskarnitish
Pre-processing of text uses snippets of Karpathy's code (BSD License)

Code for deep2ch lstm training
@libfun
'''
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import categorical_crossentropy
import lasagne
import pickle
from lasagne.nonlinearities import softmax

with open('ncollect.txt', 'r') as f:
    in_text = f.read()
in_text = in_text.replace('\n\n', '\n')

generation_phrase = "Я считаю что всё что написано выше"

chars = list(set(in_text))
data_size, vocab_size = len(in_text), len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

with open('chars.pkl', 'wb') as f:
    pickle.dump(chars, f)

lasagne.random.set_rng(np.random.RandomState(1))

SEQ_LENGTH = 30
N_HIDDEN = 512
LEARNING_RATE = .01
GRAD_CLIP = 100
PRINT_FREQ = 500
NUM_EPOCHS = 50
BATCH_SIZE = 4096
SEQ_OUT_LEN = 500


def gen_data(p, batch_size=BATCH_SIZE, data=in_text, return_target=True):
    """
    Generates input for network from data. Starting point is the position p.
    """
    x = np.zeros((batch_size, SEQ_LENGTH, vocab_size))
    y = np.zeros(batch_size)

    for n in range(batch_size):
        ptr = n
        for i in range(SEQ_LENGTH):
            x[n, i, char_to_ix[data[p + ptr + i]]] = 1.
        if (return_target):
            y[n] = char_to_ix[data[p + ptr + SEQ_LENGTH]]
    return x, np.array(y, dtype='int32')


def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")

    l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size))

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True)

    l_out = lasagne.layers.DenseLayer(l_forward_2, num_units=vocab_size,
                                      W=lasagne.init.Normal(),
                                      nonlinearity=softmax)
    target_values = T.ivector('target_output')
    network_output = lasagne.layers.get_output(l_out)
    cost = categorical_crossentropy(network_output, target_values).mean()
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost,
                            updates=updates, allow_input_downcast=True)

    probs = theano.function([l_in.input_var], network_output,
                            allow_input_downcast=True)

    def try_it_out(N=SEQ_OUT_LEN):
        """
        Generates output for the predefined generation_phrase.
        More general example of generation function is in the application
        script.
        """
        assert (len(generation_phrase) >= SEQ_LENGTH)
        sample_ix = []
        x, _ = gen_data(len(generation_phrase) - SEQ_LENGTH,
                        1, generation_phrase, False)

        for i in range(N):
            ix = np.argmax(probs(x).ravel())
            sample_ix.append(ix)
            x[:, 0:SEQ_LENGTH - 1, :] = x[:, 1:, :]
            x[:, SEQ_LENGTH - 1, :] = 0
            x[0, SEQ_LENGTH - 1, sample_ix[-1]] = 1.

        random_snippet = generation_phrase + ''.join(ix_to_char[ix]
                                                     for ix in sample_ix)
        print("----\n %s \n----" % random_snippet)

    print("Training ...")
    print("Seed used for text generation is: " + generation_phrase)
    p = 0
    try:
        for it in range(int(data_size * num_epochs / BATCH_SIZE)):
            try_it_out()
            avg_cost = 0
            for _ in range(PRINT_FREQ):
                x, y = gen_data(p)
                p += SEQ_LENGTH + BATCH_SIZE - 1
                if (p + BATCH_SIZE + SEQ_LENGTH >= data_size):
                    print('Carriage Return')
                    p = 0

                avg_cost += train(x, y)
            print("Epoch {} average loss = {}"
                  .format(it * 1.0 * PRINT_FREQ / data_size * BATCH_SIZE,
                          avg_cost / PRINT_FREQ))
            netname = 'epoch-{:.5f}.pkl' \
                .format(it * 1.0 * PRINT_FREQ / data_size * BATCH_SIZE)
            with open('nets/' + netname, 'wb') as f:
                pickle.dump(lasagne.layers.get_all_param_values(l_out), f)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

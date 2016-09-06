from flask import Flask, request, abort
from flask.json import dumps
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle
import re


def remove_html_tag(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

app = Flask(__name__)

with open('/home/libfun/chars.pkl', 'rb') as f:
    chars = pickle.load(f)

vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

SEQ_LENGTH = 30
N_HIDDEN = 512
LEARNING_RATE = .01
GRAD_CLIP = 100
PRINT_FREQ = 500
NUM_EPOCHS = 50
BATCH_SIZE = 4096
SEQ_OUT_LEN = 500
TEMP = 0.4

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
                                  nonlinearity=lasagne.nonlinearities.softmax)

target_values = T.ivector('target_output')
network_output = lasagne.layers.get_output(l_out)
probs = theano.function([l_in.input_var], network_output,
                        allow_input_downcast=True)

with open('./nets/12-08-66.72314.pkl', 'rb') as f:
    weights = pickle.load(f)
    lasagne.layers.set_all_param_values(l_out, weights)


def gen_data(p, data, batch_size=BATCH_SIZE, return_target=True):
    x = np.zeros((batch_size, SEQ_LENGTH, vocab_size))
    y = np.zeros(batch_size)

    for n in range(batch_size):
        ptr = n
        for i in range(SEQ_LENGTH):
            x[n, i, char_to_ix[data[p+ptr+i]]] = 1.
        if(return_target):
            y[n] = char_to_ix[data[p+ptr+SEQ_LENGTH]]
    return x, np.array(y, dtype='int32')


def generate(generation_phrase, N=SEQ_OUT_LEN, wclim=2):
    """
    Generation function. Takes string for generation as input, outputs
    generated string.
    """
    if len(generation_phrase) < SEQ_LENGTH:
        newgen = ' '*(SEQ_LENGTH-len(generation_phrase))+generation_phrase
        generation_phrase = newgen
    assert(len(generation_phrase) >= SEQ_LENGTH)
    # Ensure that the network would not try to continue the last word of the
    # input and instead generates new word.
    generation_phrase = generation_phrase + ' '
    sample_ix = []
    x, _ = gen_data(len(generation_phrase)-SEQ_LENGTH, generation_phrase, 1, 0)

    wc = -1
    lastword_len = 0

    for i in range(N):
        # Check if we reached the wordcount limit
        if wc > wclim and (sample_ix[-1] == char_to_ix[' ']):
            print('wclim')
            break
        # Check if the network generated sentence or line
        if len(sample_ix) > 10 and (sample_ix[-1] == char_to_ix['.']
           or sample_ix[-1] == char_to_ix['\n']) and wc > 2:
            print('dot or linebreak')
            break
        ps = probs(x).ravel()
        # If this is the first char of the word - sample output with
        # temperature. Otherwise just take argmax so that the words are not
        # gibberish.
        if len(sample_ix) > 0 and (sample_ix[-1] == char_to_ix[' ']):
            dist = np.exp(np.log(ps) / TEMP)
            ix = np.random.choice(np.arange(len(chars)), p=dist/dist.sum())
            wc += 1
            lastword_len = 0
        else:
            ix = np.argmax(ps)
        sample_ix.append(ix)
        x[:, 0:SEQ_LENGTH-1, :] = x[:, 1:, :]
        x[:, SEQ_LENGTH-1, :] = 0
        x[0, SEQ_LENGTH-1, sample_ix[-1]] = 1.
        lastword_len += 1

    random_snippet = ''.join(ix_to_char[ix] for ix in sample_ix[:-1])
    return random_snippet.replace('&quot;', '')


@app.route('/deep2ch/reply', methods=['POST'])
def create_task():
    """
    Accepts POST with json of form {"q": "message text"}
    Returns json of form {"a": "answer text"}
    """
    print(request.json)
    if not request.json or 'q' not in request.json:
        abort(400)

    # Parsing input string to clean it of html tags
    inreq = remove_html_tag(request.json['q'])
    # Additional check to be sure that all the chars are from our set
    req = ''
    for symbol in inreq:
        if symbol in chars:
            req = req + symbol

    generated = generate(req)

    # Generation may return empty sting or string consisting of dots and
    # linebreaks. If this is the case we regenerate with given seed.
    if generated.replace(' ', '').replace('.', '').replace('\n', '') == '':
        generated = generate('Ты тварь тупая, умри.')

    print({'a': generated})
    return dumps({'a': generated}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

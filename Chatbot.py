from datetime import datetime
from tkinter import *
from enum import Enum
from simpleeval import simple_eval

import aiml
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import nltk.tag.brill
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from nltk import load_parser, Valuation, evaluate_sents
import tensorflow as tf

tf.random.set_seed(1234)

import tensorflow_datasets as tfds
import csv
import os
import re
import numpy as np

# Maximum number of samples to preprocess
MAX_SAMPLES = 10000


# Note the following code is adapted from a tutorial
# Adapted from https://github.com/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    # adding a start and an end token to the sentence
    return sentence


def load_conversations():
    inputs, outputs = [], []
    with open('./friends-final.txt', encoding='utf8') as friends:
        lines = friends.readlines()
        prev_line = ''
        scene = ''
        for line in lines:
            parts = line.split('\t')
            if scene == '':
                scene = parts[5]
                prev_line = parts[5]
                continue
            if scene != parts[1]:
                scene = parts[1]
                prev_line = parts[5]
                continue
            inputs.append(prev_line)
            outputs.append(parts[5])
            prev_line = parts[5]
            if len(inputs) > MAX_SAMPLES:
                return inputs, outputs

    return inputs, outputs


questions, answers = load_conversations()
print(len(questions))

# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2 ** 13)

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

# Maximum sentence length
MAX_LENGTH = 40


# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


questions, answers = tokenize_and_filter(questions, answers)

BATCH_SIZE = 64
BUFFER_SIZE = 20000

# decoder inputs use the previous target as input
# remove START_TOKEN from targets
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask
    })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
        'query': attention1,
        'key': enc_outputs,
        'value': enc_outputs,
        'mask': padding_mask
    })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


tf.keras.backend.clear_session()

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
# Lowered this value it seemed to help
DROPOUT = 0.001

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=8000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


def accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

model.load_weights('mod')
# Training code
# EPOCHS = 100
#
# model.fit(dataset, epochs=EPOCHS)
# model.save_weights('mod')

def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence


# End of adapted code
class ModulePriority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class CalculationChatbotModule:
    def priority(self):
        return ModulePriority.HIGH

    def validator(self, text):
        try:
            self.response = eval(text)
        except:
            return False
        return True

    def get_response(self, text):
        return text + " is " + str(self.response)


class WikipediaCommand:
    def execute(self, params):
        print(params)
        try:
            self.response = wikipedia.summary(wikipedia.search(params[1])[0], sentences=2)
        except wikipedia.DisambiguationError as e:
            randomPage = wikipedia.random(e.options)
            self.response = wikipedia.summary(randomPage)

        return self.response


class ExitCommand:
    def execute(self, params):
        exit(0)


class TimeCommand:
    def execute(self, params):
        print(params)
        now = datetime.now()
        isLate = False
        if now.hour < 7 or now.hour > 20:
            isLate = True
        return "The time is " + now.strftime("%I:%M %p") + (" excellent time for studying", " you should probably get "
                                                                                            "some rest")[isLate]


class DateCommand:
    def execute(self, params):
        now = datetime.now()
        return "The date is " + now.strftime("%B %d, %Y")


# Adapted from https://github.com/jankrepl/symbolic-regression-genetic-programming/blob/master/main.ipynb
class SymbolicRegressionCommand:
    def execute(self, params):
        listString = params[1].split(',')
        xRaw = []
        yRaw = []
        i = 1
        for number in listString:
            yRaw.append(float(number))
            xRaw.append(i)
            i += 1
        X = pd.DataFrame({"x": xRaw})
        y = pd.DataFrame({"y": yRaw})

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = SymbolicRegressor(generations=15, verbose=4, max_samples=0.8, random_state=42)

        model.fit(X_train, y_train)
        p = pd.DataFrame({"x": [len(xRaw) + 1]})
        str1 = " "

        # return string
        pred = model.predict(p)
        a_str = ','.join(str(round(x, 2)) for x in pred)
        return a_str


class AtomModuleCommand:
    def valuation_string(self):
        data = ""
        with open('model.mod', 'r') as myfile:
            data = myfile.read()

        return data

    def __init__(self):
        self.val = nltk.Valuation.fromstring(self.valuation_string())
        self.grammar_file = 'atom.fcfg'
        self.g = nltk.Assignment(self.val.domain)
        self.m = nltk.Model(self.val.domain, self.val)

    def execute(self, params):
        if params[1] == "inquisition":
            param1 = params[2].replace(' ', '_')
            if param1[len(param1) - 1] == 's' and param1 != 'nucleus':
                param1 = param1[0: len(param1) - 1]

            sent = ["I like a " + param1]
            try:
                results = nltk.evaluate_sents(sent, self.grammar_file, self.m, self.g)
                for result in results:
                    for (syntree, semrep, value) in result:
                        if value:
                            return "Yes you do"
                        else:
                            return "No you don't"
            except:
                return "Sorry Invalid syntax"

        if params[1] == "dislike":
            param1 = params[2].replace(' ', '_')
            if param1[len(param1) - 1] == 's' and param1 != 'nucleus':
                param1 = param1[0: len(param1) - 1]
            self.val['likes'].remove(('i', param1))
            return "Noted"
        if params[1] == "like":
            param1 = params[2].replace(' ', '_')
            if param1[len(param1) - 1] == 's' and param1 != 'nucleus':
                param1 = param1[0: len(param1) - 1]
            self.val['likes'].add(('i', param1))
            return "Noted"
        if params[1] == "contain":
            param1 = params[2].replace(' ', '_')
            if param1[len(param1) - 1] == 's' and param1 != 'nucleus':
                param1 = param1[0: len(param1) - 1]

            param2 = params[3].replace(' ', '_')
            if param2[len(param2) - 1] == 's' and param2 != 'nucleus':
                param2 = param2[0: len(param2) - 1]

            sent = ["does a " + param1 + " contain a " + param2]
            try:
                results = nltk.evaluate_sents(sent, self.grammar_file, self.m, self.g)
                for result in results:
                    for (syntree, semrep, value) in result:
                        if value:
                            return "A " + param1.replace('_', ' ') + " does contain a " + param2.replace('_', ' ')
                        else:
                            return "A " + param1.replace('_', ' ') + " does not contain a " + param2.replace('_', ' ')
            except:
                return "Sorry Invalid syntax"

        return "Sorry I can't understand that"


class Commands:
    def __init__(self):
        self.commands = {}

    def add_command(self, key, value):
        self.commands[key] = value

    def get_response(self, key, parameters):
        return self.commands[key].execute(parameters)

    def get_keys(self):
        return list(self.commands.keys())


class SimilarityChatboxModule:
    def get_similarity(self, query):
        vectorized = TfidfVectorizer(stop_words='english')
        docs_tfidf = vectorized.fit_transform(self.documents)
        query_tfidf = vectorized.fit(self.documents)
        query_tfidf = query_tfidf.transform([query])

        similarity_score = cosine_similarity(query_tfidf, docs_tfidf).flatten()
        return similarity_score

    def __init__(self):
        self.commands = Commands()
        self.setup_commands()
        self.documents = self.commands.get_keys()

    def priority(self):
        return ModulePriority.HIGH

    def validator(self, text):
        similarity = self.get_similarity(text)
        if similarity[similarity.argmax()] < 0.7:
            return False
        else:
            self.response = self.commands.get_response(self.documents[similarity.argmax()], text)
            return True

    def get_response(self, text):
        return self.response

    def setup_commands(self):
        self.commands.add_command("What is the time", TimeCommand())
        self.commands.add_command("What is the date", DateCommand())


class AIMLChatbotModule:

    def __init__(self):
        self.kernel = aiml.Kernel()
        self.kernel.learn("chatbot.xml")
        self.commands = Commands()
        self.commands.add_command(1, WikipediaCommand())
        self.commands.add_command(5, SymbolicRegressionCommand())
        self.commands.add_command(2, AtomModuleCommand())
        self.commands.add_command(0, ExitCommand())

    def priority(self):
        return ModulePriority.LOW

    def validator(self, text):
        text = text.replace('.', 'DOT')
        self.response = self.kernel.respond(text)
        self.response = self.response.replace('DOT', '.')
        if self.response[0] == '#':

            params = self.response[1:].split('$')
            cmd = int(params[0])
            if cmd == 99:
                return False
            else:
                self.response = self.commands.get_response(cmd, params)
        return True

    def get_response(self, text):
        return self.response


class ChatbotResponse:
    def __init__(self):
        self.modules = [

        ]

    def add_module(self, module):
        self.modules.append(module)
        self.modules.sort(key=lambda mod: mod.priority().value, reverse=False)

    def get_response(self, text):
        for module in self.modules:
            if module.validator(text):
                return module.get_response(text)
        return predict(text)


class ChatbotController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.model.add_module(AIMLChatbotModule())
        self.model.add_module(CalculationChatbotModule())
        self.model.add_module(SimilarityChatboxModule())

    def chat(self, input):
        self.view.chat(self.model.get_response(input))


class ChatbotView(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.controller = ChatbotController(self, ChatbotResponse())
        self.parent = master
        self.init()
        self.chat("Welcome to the educational chatbot, I can help you with your studying!")

    def init(self):
        """Configure Window"""
        self.parent.wm_title("Chatbot")
        self.parent.resizable(False, False)
        self.parent.geometry('400x300')
        self.parent.configure(background='white')
        """Chat History Box"""
        self.chatBox = Text(self.parent, bg='white', height=16, fg='black', relief=SUNKEN, yscrollcommand='FALSE', )
        self.chatBox.config(font=(None, 8))
        self.chatBox.pack(fill='both', expand=True)
        self.chatBox.configure(state="disabled")
        """Chat Input Box"""
        self.input = Text(self.parent, width=30, bg='white')
        self.input.pack(side=LEFT, padx=5, pady=5)

        self.input.bind('<Return>', self.input_sent)
        """Send Button"""
        self.send = Button(self.parent, text='Send', width=25, bg='white', fg='green', activebackground='black',
                           activeforeground='green', command=self.input_sent)
        self.send.pack(side=RIGHT, padx=5, pady=5)

    def chat(self, text):
        self.chatBox.configure(state="normal")

        if len(self.input.get("1.0", END)) > 1:
            self.chatBox.insert(END, "You: " + self.input.get("1.0", END))
        self.chatBox.insert(END, "BOT: " + text + '\n')

        self.chatBox.configure(state="disabled")

        self.input.delete("0.0", END)
        self.chatBox.see(END)

    def input_sent(self, event=None):
        self.controller.chat(self.input.get("1.0", END)[:-1])
        return 'break'


def main():
    root = Tk()
    view = ChatbotView(root)
    view.mainloop()


main()

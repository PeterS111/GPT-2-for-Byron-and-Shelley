#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
from datetime import datetime

import model, sample, encoder

def interact_model(
    model_descr='ABC',
    model_name='774M',
    seed=12,
    nsamples=10,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
    raw_text="today"
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    if batch_size is None:
        batch_size = 1
    # assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
   
        context_tokens = enc.encode(raw_text)
        

        if not os.path.exists("outputs"):
            os.mkdir("outputs")

        write_path = "outputs/"

        gen_file = '{:%Y_%m_%d_%H_%M_%S}_{}_{}_len_{}_seed_{}.txt'.format(datetime.utcnow(), model_descr, model_name, str(length), str(seed) )
        complete_path = write_path + gen_file

        details_string = (model_descr + "_model_name: " + model_name + "," + " seed:" + str(seed) + "," + " samples:" + str(nsamples) + "," +
        " length:" + str(length) + "," + " temperature:" + str(temperature) + "," + " top_k:" + str(top_k) + "," + " top_p:" + 
        str(top_p) + "," + " raw_text: " + raw_text + "\n")
        
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]

            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])                                                            

                f = open(complete_path, "a", encoding="utf-8")

                f.write( "$$$$$ " + details_string + "\n")
                f.write(raw_text + text + "\n")   

                print(raw_text + text + "\n")

                f.close()


if __name__ == '__main__':
    fire.Fire(interact_model)

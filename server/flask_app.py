from flask import Flask, request

from run import tensorflow

time_str = "18-08-14_23-58-25"
sess, pred, target_vocab, input_vocab, model = tensorflow.get_server_sess(time_str)

app = Flask(__name__)


@app.route("/")
def index():
    return "Please use /text_classification."


@app.route('/text_classification', methods=['GET', 'POST'])
def text_classification():
    if request.method == 'POST':
        text = request.form["text"]
        word_list = text[:1000]
        input_id_list = []
        for word in word_list:
            if word in input_vocab:
                word_id = input_vocab.index(word)
            else:
                word_id = input_vocab.index("UNK")
            input_id_list.append(word_id)

        pred_target_arr = sess.run(pred, feed_dict={
            model.input_holder: [input_id_list],
            model.target_holder: [[1, 0, 0, 0, 0]]
        })

        class_name = target_vocab[pred_target_arr[0]]
        return class_name
    else:
        return "Please use POST method."


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)

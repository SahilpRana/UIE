# source ~/.bashrc; conda activate myenv
# python3 main1.py --prefix fresh_run --data-dir "../Experimental results" --max-step 100 --batch-size 1 --test-step 100 --vgg16-path "./vgg16.npy"

import threading
import numpy as np
import random, sys
import shutil
from PIL import Image, ImageEnhance, ImageFile
from skimage import transform, io, img_as_float, color
import cv2
import glob, pickle
from random import shuffle
from model_vgg import model_vgg
import time
from tqdm import tqdm
from utils import *
import os
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from q_network import q_network, q_network_shallow, q_network_large
from get_hist import *
from get_global_feature import get_global_feature
from action import take_action, action_size

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}, linewidth=200, threshold=np.inf)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def imread(path, mode='RGB'):
    """Read an image file and return a uint8 numpy array in the given mode."""
    img = Image.open(path).convert(mode)
    return np.array(img)


def imresize(arr, size):
    """Resize a uint8 numpy array to (H, W) using Lanczos; mirrors scipy.misc.imresize."""
    h, w = size
    img = Image.fromarray(arr)
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img)


class Agent:
    def __init__(self, prefix, args):
        """ init the class """
        self.save_raw = True
        self.finetune = False
        self.logging = False
        self.test_logging = True
        self.use_history = True
        self.use_batch_norm = False
        self.deep_feature_model_path = args.vgg16_path

        self.gpu_num = args.gpu

        self.enqueue_repeat = 3
        self.prep = None
        self.double_q = True
        self.duel = False
        self.use_deep_feature = True
        self.use_color_feature = True
        self.deep_feature_len = 0
        if self.use_deep_feature:
            self.deep_feature_len = 4096
        if not self.use_color_feature:
            self.color_type = 'None'
            self.color_feature_len = 0
        else:
            self.color_type = 'lab_8k'
            self.color_feature_len = 8000

        self.w = {}
        self.batch_size = args.batch_size
        self.max_step = args.max_step
        self.seq_len = args.seq_len
        self.feature_length = self.deep_feature_len + self.color_feature_len + self.use_history*self.seq_len*action_size
        
        self.q = None
        self.q_w = None
        self.target_q = None
        self.target_q_w = None
        self.target_q_w_input = None
        self.target_q_w_assign_op = None
        self.delta = None
        self.min_delta = -1.0
        self.max_delta = 1.0
        self.action_size = action_size
        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)
        self.max_reward = 0.5
        self.min_reward = -0.5
        self.memory_size = 5000
        self.target_q_update_step = 1000
        self.test_step = args.test_step
        self.learning_rate_minimum = 0.00000001
        self.learning_rate = 0.00001
        self.learning_rate_decay_step = 5000
        self.learning_rate_decay = 0.96
        self.learn_start = 100
        self.train_frequency = 1
        self.discount = 0.95
        self.save_interval = self.test_step
        self.prefix = prefix

        # build output directory from prefix (no --output-dir CLI arg required)
        self.output_dir = 'test'
        self.output_prefix_dir = os.path.join(self.output_dir, self.prefix)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')

        self.data_dir = args.data_dir
        self.test_count = 60

        self.train_dir = os.path.join(self.data_dir, 'train')
        self.test_dir = os.path.join(self.data_dir, 'test')

        self.num_thread = 5
        self.config = {
            "learning_rate": self.learning_rate,
            "learning_rate_decay_rate": self.learning_rate_decay,
            "learning_rate_minimum": self.learning_rate_minimum,
            "target_q_update_step": self.target_q_update_step,
            "discount": self.discount,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "min_delta": self.min_delta,
            "max_delta": self.max_delta,
            "feature_type": self.color_type,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "prefix": self.prefix,
            "memory_size": self.memory_size,
            "action_size": action_size,
        }

    def init_img_list(self, img_list_path=None, test=False):
        if img_list_path:
            with open(img_list_path, 'rb') as f:
                self.img_list = pickle.load(f)
            self.train_img_list = self.img_list['train_img_list']
            self.test_img_list = self.img_list['test_img_list']
        else:
            train_raw_dir = self.train_dir
            test_raw_dir = self.test_dir
            self.train_img_list = [os.path.join(train_raw_dir, x) for x in os.listdir(train_raw_dir) if x.endswith(('.jpg', '.png'))]
            self.test_img_list = [os.path.join(test_raw_dir, x) for x in os.listdir(test_raw_dir) if x.endswith(('.jpg', '.png'))]
            print('train img list:', self.train_img_list[:10])
            print('test img list:', self.test_img_list[:10])
            self.img_list = {'train_img_list': self.train_img_list, 'test_img_list': self.test_img_list}

            if not os.path.exists(self.output_prefix_dir):
                os.makedirs(self.output_prefix_dir)
            with open(os.path.join(self.output_prefix_dir, 'img_list.pkl'), 'wb') as f:
                pickle.dump(self.img_list, f)
        if not os.path.exists(self.output_prefix_dir):
            os.makedirs(self.output_prefix_dir)
        with open(os.path.join(self.output_prefix_dir, 'config'), 'wb') as f:
            pickle.dump(self.config, f)
        self.train_img_count = len(self.train_img_list)
        self.test_img_count = len(self.test_img_list)

    def save_model(self, sess, step=None):
        print(" [*] Saving checkpoints...")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(sess, os.path.join(self.checkpoint_dir, self.prefix + ".ckpt"), global_step=step)

    def load_model(self, sess, model_path):
        print('Restoring model ...')
        self.saver = tf.train.import_meta_graph(model_path + '.meta')
        self.saver.restore(sess, model_path)
        print(" [*] Load success: %s" % model_path)
        return True

    def train_with_queue(self, sess):
        self.step = 0
        self.coord = tf.train.Coordinator()
        for i in range(self.num_thread):
            print('[Thread %d/%d] enqueue samples' % (i, self.num_thread))
            t = threading.Thread(target=enqueue_samples, args=(self, self.coord, sess))
            t.start()

        for self.step in tqdm(range(0, self.max_step), ncols=70, initial=0):
            current_step = self.step + 1
            if self.step == 0:
                print("############## do dry test ###############")
                self.test(sess, idx=0)
                print("############## dry test finished #############")
            if current_step % 1000 == 0:
                print('############## Now is step %d ###############' % current_step)
                print('Starting enqueue...')

            self.q_learning_minibatch(sess)
            if current_step % self.target_q_update_step == 0:
                print("Step %d: Target Q updating ... " % current_step)
                print(f'Step {current_step}: q_learning_minibatch...')
                self.q_learning_minibatch(sess)
                print(f'Step {current_step}: done')

            if current_step % self.save_interval == 0:
                print("Step %d: Saving model ... " % current_step)
                self.save_model(sess, current_step)

            if current_step % self.test_step == 0:
                init_scores = []
                final_scores = []
                print("Step %d: ############ Testing ... #############" % current_step)
                for i in range(self.test_count):
                    init_score, final_score = self.test(
                        sess, in_order=True, idx=i,
                        save_raw=(current_step % (self.test_step * 5) == 0)
                    )
                    print('[Testing %d/%d] init score = %.5f; final score = %.5f'
                          % (i, self.test_count, init_score, final_score))
                    init_scores.append(init_score)
                    final_scores.append(final_score)
                print('Average init scores = %.6f; average final scores = %.6f'
                      % (np.mean(init_scores), np.mean(final_scores)))
                init_scores = [str(v) for v in init_scores]
                final_scores = [str(v) for v in final_scores]

                with open('./test/' + self.prefix + '/test.txt', 'a') as f:
                    f.write("step %d\n" % current_step)
                    f.write(" ".join(init_scores) + "\n")
                    f.write(" ".join(final_scores) + "\n")

    def q_learning_minibatch(self, sess, sample_terminal=False):
        state, reward, action, next_state, terminal = sess.run(
            [self.s_t_many, self.reward_many, self.action_many,
             self.s_t_plus_1_many, self.terminal_many]
        )

        if self.double_q:
            if self.use_batch_norm:
                q, q_actions = sess.run(
                    [self.q, self.q_action],
                    feed_dict={self.s_t: state, self.phase_train: True}
                )
            else:
                q, q_actions = sess.run(
                    [self.q, self.q_action],
                    feed_dict={self.s_t: state}
                )
            q_t_plus_1 = self.target_q.eval({self.target_s_t: next_state}, session=sess)
            pred_action = q_actions
            max_q_t_plus_1 = []
            for i in range(self.batch_size):
                max_q_t_plus_1.append(q_t_plus_1[i][pred_action[i]])
            max_q_t_plus_1 = np.reshape(np.array(max_q_t_plus_1), [self.batch_size, 1])

            terminal = np.array(terminal) + 0.
            target_q_t = (1 - terminal) * self.discount * max_q_t_plus_1 + reward
            action = np.reshape(action, target_q_t.shape)
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: next_state}, session=sess)
            terminal_np = np.array(terminal) + 0.
            max_q_t_plus_1 = np.reshape(np.max(q_t_plus_1, axis=1), terminal_np.shape)
            target_q_t = (1 - terminal_np) * self.discount * max_q_t_plus_1 + reward
            action = np.reshape(action, target_q_t.shape)

        action = np.reshape(action, (self.batch_size))
        target_q_t = np.reshape(target_q_t, (self.batch_size))
        if self.use_batch_norm:
            _, q_t, loss, delta, one_hot = sess.run(
                [self.optim, self.q, self.loss, self.delta, self.action_one_hot],
                feed_dict={
                    self.target_q_t: target_q_t,
                    self.action: action,
                    self.s_t: state,
                    self.phase_train: True,
                }
            )
        else:
            _, q_t, loss, delta, one_hot = sess.run(
                [self.optim, self.q, self.loss, self.delta, self.action_one_hot],
                feed_dict={
                    self.target_q_t: target_q_t,
                    self.action: action,
                    self.s_t: state,
                }
            )
        if self.logging:
            print("q_t"); print(q_t)
            print("delta"); print(delta)
            print("terminal"); print(terminal)

    def init_model(self, sess, model_path=None):
        self.s_t_single = tf.placeholder('float32', [self.batch_size, self.feature_length], name="s_t_single")
        self.s_t_plus_1_single = tf.placeholder('float32', [self.batch_size, self.feature_length], name="s_t_plus_1_single")
        self.action_single = tf.placeholder('int64', [self.batch_size, 1], name='action_single')
        self.terminal_single = tf.placeholder('int64', [self.batch_size, 1], name='terminal_single')
        self.reward_single = tf.placeholder('float32', [self.batch_size, 1], name='reward_single')

        self.s_t = tf.placeholder('float32', [self.batch_size, self.feature_length], name="s_t")
        self.target_q_t = tf.placeholder('float32', [self.batch_size], name='target_q_t')
        self.action = tf.placeholder('int64', [self.batch_size], name='action')

        self.queue = tf.queue.RandomShuffleQueue(
            self.memory_size, 1000,
            [tf.float32, tf.float32, tf.float32, tf.int64, tf.int64],
            [[self.feature_length], [self.feature_length], 1, 1, 1]
        )
        self.enqueue_op = self.queue.enqueue_many(
            [self.s_t_single, self.s_t_plus_1_single,
             self.reward_single, self.action_single, self.terminal_single]
        )
        self.s_t_many, self.s_t_plus_1_many, self.reward_many, self.action_many, self.terminal_many = \
            self.queue.dequeue_many(self.batch_size)

        self.target_s_t = tf.placeholder('float32', [self.batch_size, self.feature_length], name="target_s_t")

        if self.use_batch_norm:
            self.phase_train = tf.placeholder(tf.bool, name='phase_train')
            self.q, self.q_w = q_network(
                self.s_t, 'pred', input_length=self.feature_length,
                num_action=self.action_size, duel=self.duel,
                batch_norm=self.use_batch_norm, phase_train=self.phase_train
            )
            self.target_q, self.target_q_w = q_network(
                self.target_s_t, 'target', input_length=self.feature_length,
                num_action=self.action_size, duel=self.duel,
                batch_norm=self.use_batch_norm, phase_train=self.phase_train
            )
        else:
            self.q, self.q_w = q_network(
                self.s_t, 'pred', input_length=self.feature_length,
                num_action=self.action_size, duel=self.duel
            )
            self.target_q, self.target_q_w = q_network(
                self.target_s_t, 'target', input_length=self.feature_length,
                num_action=self.action_size, duel=self.duel
            )

        self.q_action = tf.argmax(self.q, axis=1)
        self.target_q_action = tf.argmax(self.target_q, axis=1)

        with tf.variable_scope('optimizer'):
            self.action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
            print("self.q shape"); print(self.q.get_shape().as_list())
            print("self.action_one_hot shape"); print(self.action_one_hot.get_shape().as_list())
            q_acted = tf.reduce_sum(self.q * self.action_one_hot, axis=1, name='q_acted')
            print("q_acted shape"); print(q_acted.get_shape().as_list())
            print("target_q_t shape"); print(self.target_q_t.get_shape().as_list())
            self.delta = self.target_q_t - q_acted
            print("delta shape"); print(self.delta.get_shape().as_list())
            self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')
            for weight in self.q_w.values():
                self.loss += 1e-4 * tf.nn.l2_loss(weight)
            self.learning_rate_op = tf.maximum(
                self.learning_rate_minimum,
                tf.train.exponential_decay(
                    self.learning_rate, self.global_step,
                    self.learning_rate_decay_step, self.learning_rate_decay, staircase=True
                )
            )
            self.optim = tf.train.AdamOptimizer(self.learning_rate_op).minimize(
                self.loss, var_list=list(self.q_w.values()), global_step=self.global_step
            )

        with tf.variable_scope('pred_to_target'):
            self.target_q_w_input = {}
            self.target_q_w_assign_op = {}
            for name in self.q_w.keys():
                self.target_q_w_input[name] = tf.placeholder(
                    'float32', self.target_q_w[name].get_shape().as_list(), name=name
                )
                self.target_q_w_assign_op[name] = self.target_q_w[name].assign(self.target_q_w_input[name])

        # tf.initialize_all_variables() is removed in TF2 compat.v1  use tf.global_variables_initializer()
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(list(self.q_w.values()) + [self.step_op], max_to_keep=30)

    def init_preprocessor(self, sess):
        self.prep = DeepFeatureNetwork(
            [self.batch_size, 224, 224, 3], model_vgg,
            self.deep_feature_model_path, self.gpu_num
        )
        self.prep.init_model(sess)

    def predict(self, sess, state, is_training=True, use_target_q=True):
        if is_training:
            if self.step < 10000:
                ep = 1.0
            elif self.step < 40000:
                ep = 0.8
            elif self.step < 80000:
                ep = 0.6
            elif self.step < 160000:
                ep = 0.4
            elif self.step < 320000:
                ep = 0.2
            else:
                ep = 0.1
        if use_target_q:
            if self.use_batch_norm:
                q, q_actions = sess.run(
                    [self.target_q, self.target_q_action],
                    feed_dict={self.target_s_t: state, self.phase_train: is_training}
                )
            else:
                q, q_actions = sess.run(
                    [self.target_q, self.target_q_action],
                    feed_dict={self.target_s_t: state}
                )
        else:
            if self.use_batch_norm:
                q, q_actions = sess.run(
                    [self.q, self.q_action],
                    feed_dict={self.s_t: state, self.phase_train: is_training}
                )
            else:
                q, q_actions = sess.run(
                    [self.q, self.q_action],
                    feed_dict={self.s_t: state}
                )

        if self.finetune:
            ep = 0.1
        qs = []
        actions = []
        for i in range(self.batch_size):
            if is_training and random.random() < ep:
                action_idx = random.randrange(self.action_size)
                actions.append(action_idx)
                qs.append(q[i])
            else:
                actions.append(q_actions[i])
                qs.append(q[i])
        return actions, np.array(qs)

    def update_target_q_network(self, sess):
        for name in self.q_w.keys():
            self.target_q_w_assign_op[name].eval(
                {self.target_q_w_input[name]: self.q_w[name].eval(session=sess)},
                session=sess
            )

    def get_hist(self, image_data):
        color_hist = []
        for i in range(image_data.shape[0]):
            if self.color_type == 'rgbl':
                color_hist.append(rgbl_hist(image_data[i]))
            elif self.color_type == 'lab':
                color_hist.append(lab_hist(image_data[i]))
            elif self.color_type == 'lab_8k':
                color_hist.append(lab_hist_8k(image_data[i]))
            elif self.color_type == 'tiny':
                color_hist.append(tiny_hist(image_data[i]))
            elif self.color_type == 'tiny28':
                color_hist.append(tiny_hist_28(image_data[i]))
            elif self.color_type == 'VladB':
                color_hist.append(get_global_feature(image_data[i]))
        return np.stack(np.array(color_hist))

    def get_state(self, sess, raw_data, target_data, history=None):
        scores = []
        for i in range(len(target_data)):
            target_lab = color.rgb2lab(target_data[i] + 0.5)
            data_lab = color.rgb2lab(raw_data[i] + 0.5)
            mse = np.sqrt(np.sum((target_lab - data_lab) ** 2, axis=2)).mean() / 10.0
            scores.append([-mse])

        if self.use_deep_feature:
            if self.use_color_feature:
                color_features = self.get_hist(raw_data)
                features = []
                deep_features = self.prep.get_feature(sess, raw_data)
                for i in range(self.batch_size):
                    if self.use_history:
                        features.append(np.concatenate((deep_features[i], color_features[i], history[i]), axis=0))
                    else:
                        features.append(np.concatenate((deep_features[i], color_features[i]), axis=0))
                return np.stack(np.array(features)), np.array(scores)
            else:
                deep_features = self.prep.get_feature(sess, raw_data)
                return deep_features, np.array(scores)
        else:
            color_features = self.get_hist(raw_data)
            return np.stack(np.array(color_features)), np.array(scores)

    def get_new_state(self, sess, is_training=True, in_order=False, idx=-1, get_raw_images=False):
        if in_order:
            state_raw, target_state_raw, op_str, fn, raw_images = self._load_images(
                idx, is_training, in_order=in_order, get_raw_images=get_raw_images
            )
        else:
            state_raw, target_state_raw, op_str, fn, raw_images = self._load_images(
                idx, is_training, get_raw_images=get_raw_images
            )

        history = None
        if self.use_history:
            history = np.zeros([self.batch_size, self.seq_len * action_size])
            state, score = self.get_state(sess, state_raw, target_state_raw, history=history)
        else:
            state, score = self.get_state(sess, state_raw, target_state_raw)
        return state, state_raw, score, target_state_raw, op_str, fn, raw_images, history

    def _load_images(self, offset, is_training, in_order=False, get_raw_images=False):
        if is_training:
            max_offset = max(0, self.train_img_count - self.batch_size)
            offset = random.randint(0, max_offset) if max_offset > 0 else 0
            img_list = self.train_img_list[offset:offset + self.batch_size]
        else:
            if offset >= 0:
                offset = offset * self.batch_size
            else:
                offset = random.randint(0, self.test_img_count - self.batch_size)
            if offset + self.batch_size > len(self.test_img_list):
                offset = len(self.test_img_list) - self.batch_size
            img_list = self.test_img_list[offset:offset + self.batch_size]

        imgs = []
        target_imgs = []
        op_str = []
        raw_imgs = []
        raw_imgs_raw = []
        raw_imgs_target = []

        for img_path in img_list:
            # imread + imresize defined at top of file (replaces deprecated scipy.misc versions)
            imgs.append(imresize(imread(img_path, mode='RGB'), (224, 224)) / 255.0 - 0.5)
            target_path = os.path.join(
                os.path.join(os.path.dirname(os.path.dirname(img_path)), "target"),
                os.path.basename(img_path).split("__")[0]
            )
            if "__" in os.path.basename(img_path):
                op_str.append(os.path.basename(img_path).split("__")[1])
            else:
                op_str.append("_")
            target_imgs.append(imresize(imread(target_path, mode='RGB'), (224, 224)) / 255.0 - 0.5)

            if get_raw_images:
                raw_imgs_raw.append(imread(img_path, mode='RGB') / 255.0 - 0.5)
                raw_imgs_target.append(imread(target_path, mode='RGB') / 255.0 - 0.5)

        if len(raw_imgs_raw) > 0:
            if self.logging:
                for raw in raw_imgs_raw:
                    print(raw.shape)
            raw_imgs.append(raw_imgs_raw)
        if len(raw_imgs_target) > 0:
            raw_imgs.append(raw_imgs_target)

        fns = [os.path.basename(path) for path in img_list]
        return np.array(np.stack(imgs, axis=0)), np.array(np.stack(target_imgs, axis=0)), op_str, fns, raw_imgs

    def act(self, actions, state_raw, target_state_raw, prev_score, sess,
            is_training=True, step_count=0, history=None):
        images_after = []
        for i, action_idx in enumerate(actions):
            if action_idx == -1:
                images_after.append(state_raw[i])
            else:
                images_after.append(take_action(state_raw[i], action_idx))

        images_after_np = np.stack(images_after, axis=0)

        if self.use_history:
            for idx in range(self.batch_size):
                history[idx][step_count * action_size + actions[idx]] = 1
            new_state, new_score = self.get_state(sess, images_after_np, target_state_raw, history=history)
        else:
            new_state, new_score = self.get_state(sess, images_after_np, target_state_raw)

        if not is_training:
            return new_state, images_after_np, None, None, history

        reward = new_score - prev_score
        return new_state, images_after_np, reward, new_score, history

    def act_on_raw_img(self, actions, state_raw):
        images_after = []
        for i, action_idx in enumerate(actions):
            if action_idx == -1:
                images_after.append(state_raw[i])
            else:
                images_after.append(take_action(state_raw[i], action_idx))
        return images_after

    def test(self, sess, is_batch_test=False, use_target_q=True,
             in_order=False, idx=-1, save_raw=False):
        if is_batch_test:
            test_result_dir = "test/" + self.prefix + "/batch_test_%s" % self.prefix
        else:
            test_result_dir = "test/" + self.prefix + "/step_%010d" % self.step
        if not os.path.exists(test_result_dir):
            os.mkdir(test_result_dir)
        print('[Testing] directory = %s' % test_result_dir)

        state, state_raw, score_initial, target_state_raw, op_strs, fn, raw_images, history = \
            self.get_new_state(sess, is_training=False, in_order=in_order, idx=idx, get_raw_images=True)
        score = score_initial.copy()
        state_raw_init = state_raw.copy()
        raw_images_raw = raw_images[0]
        raw_images_target = raw_images[1]
        retouched_raw_images = [item.copy() for item in raw_images_raw]
        actions = []

        print('[Testing] Retouching images ... #####################')
        for i in range(self.seq_len):
            action, q_val = self.predict(sess, state, is_training=False, use_target_q=use_target_q)
            if self.logging or self.test_logging:
                print("[Testing] Retouch step %d" % i)
            all_stop = True
            for j in range(self.batch_size):
                if q_val[j][action[j]] <= 0:
                    all_stop = all_stop and True
                    action[j] = -1
                else:
                    all_stop = all_stop and False
            if all_stop:
                print("all negative, stop. %d retouch step finished" % i)
                break
            next_state, next_state_raw, r, new_score, history = self.act(
                action, state_raw, target_state_raw, score, sess,
                is_training=False, history=history
            )
            state = next_state
            state_raw = next_state_raw
            if self.logging:
                print(""); print("step", i); print(q_val); print(q_val.argsort())
            actions.append(action)
            score = new_score

            if self.save_raw:
                if i < 4:
                    retouched_raw_images = self.act_on_raw_img(action, retouched_raw_images)

        if self.logging:
            print(actions)

        state, final_score = self.get_state(sess, state_raw, target_state_raw, history=history)
        score_diff = final_score - score_initial
        if self.logging or self.test_logging:
            print(""); print("final_score:\t", final_score); print("original_score:\t", score_initial)

        raw_dir = os.path.join(test_result_dir, 'raw')
        if not os.path.exists(raw_dir):
            os.mkdir(raw_dir)

        initial_score = 0
        retouched_score = 0
        print('[Testing] Storing retouched images ... #################')
        for i in range(state_raw.shape[0]):
            try:
                actions_str = []
                for j in range(len(actions)):
                    if actions[j][i] != -1:
                        actions_str.append(str(actions[j][i]))
                actions_desc = "_".join(actions_str)
                if idx == -1:
                    random_id = random.randrange(999999999)
                else:
                    random_id = idx
                with open(os.path.join(test_result_dir, '%s.log' % fn[i]), 'w') as f:
                    f.write("%s_%f_%s.png\n" % (fn[i], score_diff[i][0], actions_desc))
                image = Image.fromarray(np.uint8(np.clip((state_raw[i] + 0.5) * 255, 0, 255)))
                image.save(os.path.join(test_result_dir, "%s_%f_retouched.png" % (fn[i], score_diff[i][0])))
                image_target = Image.fromarray(np.uint8(np.clip((target_state_raw[i] + 0.5) * 255, 0, 255)))
                image_target.save(os.path.join(test_result_dir, "%s_target.png" % fn[i]))
                image_init = Image.fromarray(np.uint8(np.clip((state_raw_init[i] + 0.5) * 255, 0, 255)))
                image_init.save(os.path.join(test_result_dir, "%s_raw_%s.png" % (fn[i], op_strs[i])))

                raw_dir_dir = os.path.join(raw_dir, fn[i].split("__")[0])
                if not os.path.exists(raw_dir_dir):
                    os.mkdir(raw_dir_dir)

                if save_raw and i < 4:
                    raw_image_raw = Image.fromarray(np.uint8(np.clip((raw_images_raw[i] + 0.5) * 255, 0, 255)))
                    raw_image_retouched = Image.fromarray(np.uint8(np.clip((retouched_raw_images[i] + 0.5) * 255, 0, 255)))
                    raw_image_target = Image.fromarray(np.uint8(np.clip((raw_images_target[i] + 0.5) * 255, 0, 255)))

                    target_lab = color.rgb2lab(raw_images_target[i] + 0.5)
                    retouched_lab = color.rgb2lab(retouched_raw_images[i] + 0.5)
                    initial_lab = color.rgb2lab(raw_images_raw[i] + 0.5)
                    initial_score += -np.sqrt(np.sum((target_lab - initial_lab) ** 2, axis=2)).mean() / 10.0
                    retouched_score += -np.sqrt(np.sum((target_lab - retouched_lab) ** 2, axis=2)).mean() / 10.0
                    try:
                        raw_image_raw.save(os.path.join(raw_dir_dir, "%s_raw.png" % fn[i]))
                        raw_image_retouched.save(os.path.join(raw_dir_dir, "%s_retouched_%f_%s.png" % (fn[i], score_diff[i][0], actions_desc)))
                        raw_image_target.save(os.path.join(raw_dir_dir, "%s_target.png" % fn[i]))
                    except Exception as e:
                        print("[!!!! TESTING ERROR] raw image saving problem"); print(str(e))
                else:
                    target_lab = color.rgb2lab(target_state_raw[i] + 0.5)
                    retouched_lab = color.rgb2lab(state_raw[i] + 0.5)
                    initial_lab = color.rgb2lab(state_raw_init[i] + 0.5)
                    initial_score += -np.sqrt(np.sum((target_lab - initial_lab) ** 2, axis=2)).mean() / 10.0
                    retouched_score += -np.sqrt(np.sum((target_lab - retouched_lab) ** 2, axis=2)).mean() / 10.0
            except Exception as e:
                print('[!!!! TESTING ERROR] Exception: %s' % str(e))
        return initial_score, retouched_score


class DeepFeatureNetwork:
    def __init__(self, input_size, model_func, model_path, gpu_num):
        self.input_tensor = tf.placeholder(tf.float32, shape=input_size)
        self.feature, self.weights = model_func(self.input_tensor, model_path, gpu_num)

    def init_model(self, sess):
        sess.run(tf.variables_initializer(self.weights))

    def get_feature(self, sess, in_data):
        return sess.run(self.feature, feed_dict={self.input_tensor: in_data + 0.5})


def enqueue_samples(agent, coord, sess):
    """Worker thread: continuously enqueue (s, s', r, a, terminal) tuples."""
    repeat = 0
    state, state_raw, score, target_state_raw, _, _, _, history = agent.get_new_state(sess)

    step_count = 0
    while not coord.should_stop():
        if repeat > agent.seq_len:
            repeat = 0
            state, state_raw, score, target_state_raw, _, _, _, history = agent.get_new_state(sess)
            terminal = False
            step_count = 0
            if agent.use_history:
                del history
                history = np.zeros([agent.batch_size, agent.seq_len * action_size])

        action, q_val = agent.predict(sess, state)
        step_count += 1

        if agent.use_history:
            next_state, next_state_raw, reward, new_score, history = agent.act(
                action, state_raw, target_state_raw, score, sess, history=history
            )
        else:
            next_state, next_state_raw, reward, new_score, _ = agent.act(
                action, state_raw, target_state_raw, score, sess
            )

        terminal = reward <= 0
        if np.sum(terminal) > agent.batch_size / 2:
            repeat += 5
        else:
            repeat += 1
        terminal = np.reshape(terminal, (agent.batch_size, -1))
        action = np.reshape(action, (agent.batch_size, -1))

        for _ in range(agent.enqueue_repeat):
            sess.run(
                agent.enqueue_op,
                feed_dict={
                    agent.s_t_single: state,
                    agent.s_t_plus_1_single: next_state,
                    agent.reward_single: reward,
                    agent.action_single: action,
                    agent.terminal_single: terminal,
                }
            )

        state_raw = next_state_raw
        state = next_state
        score = new_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prefix")
    parser.add_argument("--gpu", type=str, default='1', help="assign a gpu")
    parser.add_argument("--data-dir", type=str, default='/home/sunshixin/2021/joe/data', help="Data directory")
    parser.add_argument("--vgg16-path", type=str, default="./vgg16.npy")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--seq-len", type=int, default=20, help="The number of retouch actions to take.")
    parser.add_argument("--test-step", type=int, default=160000, help="Test Step.")
    parser.add_argument("--max-step", type=int, default=6000000, help="Maximum training steps.")
    parser.add_argument('--test', action="store_true", default=False)
    args = parser.parse_args()
    model_path = args.model_path
    prefix = args.prefix

    if prefix is None:
        print("please provide a valid prefix")
        sys.exit(1)
    elif os.path.exists('./test/' + prefix):
        print("duplicated prefix")
        shutil.rmtree('./test/' + prefix)
        os.makedirs('./test/' + prefix)
    else:
        os.makedirs('./test/' + prefix)

    agent = Agent(prefix, args)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        agent.init_preprocessor(sess)
        agent.init_model(sess)
        agent.init_img_list()
        if args.test:
            if model_path is None:
                print('[Error] Please give the model path!!!')
                sys.exit(1)
            agent.load_model(sess, model_path)
            print("run test with model {}".format(model_path))
            agent.step = 0
            for i in range(agent.test_count):
                init_score, final_score = agent.test(sess, in_order=True, idx=i)
        else:
            if model_path is not None:
                agent.load_model(sess, model_path)
            print("start training with prefix {}".format(prefix))
            agent.train_with_queue(sess)
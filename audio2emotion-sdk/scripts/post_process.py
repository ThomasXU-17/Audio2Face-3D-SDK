# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import numpy as np


def default_post_processing(a2e_emotion_vector):
    # default params
    _emotion_contrast = 1.0
    _emotion_strength = 0.6
    _live_blend_coef = 0.7
    _preferred_emotion_strength = 0.0
    _max_emotions = 6
    _a2f_emotion_dim = 10
    _live_prev_emo = np.zeros((_a2f_emotion_dim), dtype=np.float32)
    _preferred_emotion = np.zeros((_a2f_emotion_dim), dtype=np.float32)

    def _sigmoid(data):
        return 1 / (1 + np.exp(-data))

    def _softmax(x):
        exp_shifted = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

    # EmotionContrast
    a2e_emotion_vector = _softmax(a2e_emotion_vector * _emotion_contrast)
    a2e_emotion_vector[4] = 0

    # MaxEmotions
    zero_emotion_idxes = np.argsort(a2e_emotion_vector, axis=-1)[:-_max_emotions]
    a2e_emotion_vector[zero_emotion_idxes] = 0.0

    # MapToA2FEmotionIndex
    a2f_emotion_vector = np.zeros((_a2f_emotion_dim), dtype=np.float32)
    emo2id = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5}
    a2f_emotion_vector[1] = a2e_emotion_vector[emo2id["angry"]]
    a2f_emotion_vector[3] = a2e_emotion_vector[emo2id["disgust"]]
    a2f_emotion_vector[4] = a2e_emotion_vector[emo2id["fear"]]
    a2f_emotion_vector[6] = a2e_emotion_vector[emo2id["happy"]]
    a2f_emotion_vector[9] = a2e_emotion_vector[emo2id["sad"]]

    # Smoothing
    a2f_emotion_vector = (1 - _live_blend_coef) * a2f_emotion_vector + _live_blend_coef * _live_prev_emo

    # BlendPreferredEmotion
    a2f_emotion_vector = (
        _preferred_emotion_strength * _preferred_emotion + (1.0 - _preferred_emotion_strength) * a2f_emotion_vector
    )

    # EmotionStrength
    a2f_emotion_vector = _emotion_strength * a2f_emotion_vector
    return a2f_emotion_vector

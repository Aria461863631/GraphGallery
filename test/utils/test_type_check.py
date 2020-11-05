# from pytest import raises as assert_raises
from graphgallery.utils.type_check import *
from graphgallery import set_backend
from graphgallery import intx, floatx

import numpy as np
import tensorflow as tf
import torch
from scipy.sparse import csr_matrix

from pytest import raises as assert_raises

def test_is_list_like():
    assert is_list_like([1, 2])
    assert is_list_like((1, 2))
    assert is_list_like(1) == False

def test_is_scalar_like():
    assert is_scalar_like(3)
    assert is_scalar_like(np.ndarray(()))
    assert is_scalar_like(Exception) == False
    assert is_scalar_like(np.ndarray([4,5])) == False
    # assert is_scalar_like(csr_matrix((3, 4)))

def test_is_integer_scalar():
    assert is_interger_scalar(4)
    assert is_interger_scalar(np.int64(4))
    assert is_interger_scalar("str") == False

def test_infer_type():
    # torch
    x = torch.tensor([1], dtype=torch.int64)
    assert infer_type(x) == intx()
    x = torch.tensor([1], dtype=torch.short)
    assert infer_type(x) == intx()
    x = torch.tensor([2.5], dtype=torch.float64)
    assert infer_type(x) == floatx() 
    x = torch.tensor([False], dtype=torch.bool)
    assert infer_type(x) == 'bool'
    x = torch.tensor([1.0], dtype=torch.complex64)
    assert_raises(RuntimeError, infer_type, x)

    # tf
    x = tf.constant(3)
    assert infer_type(x) == intx()
    x = tf.constant(3.0)
    assert infer_type(x) == floatx()
    x = tf.constant(True)
    assert infer_type(x) == 'bool'
    x = tf.constant("type_string")
    assert_raises(RuntimeError, infer_type, x)

    # others
    assert infer_type([1, 2]) == intx()
    assert infer_type([1.0]) == floatx()
    assert infer_type(True) == 'bool'
    assert_raises(RuntimeError, infer_type, 'string')

def test_is_tensor():
    set_backend('T')
    assert is_tensor(tf.constant([1, 2]))
    set_backend('P')
    assert is_tensor(torch.tensor([1]))
    assert not is_tensor("string")




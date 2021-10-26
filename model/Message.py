"""Built-in message function."""
from __future__ import absolute_import

import sys
from itertools import product

from dgl.function.base import BuiltinFunction, TargetCode
from dgl._deprecate.runtime import ir
from dgl._deprecate.runtime.ir import var
import torch
import os
import numpy as np
import random
from torch import optim
from torch import nn
from torch.nn import functional as F

class Message(BuiltinFunction):

    def _invoke(self, graph, src_frame, dst_frame, edge_frame, out_size,
                src_map, dst_map, edge_map, out_map, reducer="none"):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError


class BinaryMessage(Message):

    def __init__(self, binary_op, lhs, rhs, lhs_field, rhs_field, out_field):
        self.binary_op = binary_op
        self.lhs = lhs
        self.rhs = rhs
        self.lhs_field = lhs_field
        self.rhs_field = rhs_field
        self.out_field = out_field

    def _invoke(self, graph, src_frame, dst_frame, edge_frame, out_size,
                src_map, dst_map, edge_map, out_map, reducer="none"):

        graph = var.GRAPH(graph)
        in_frames = (src_frame, dst_frame, edge_frame)
        in_maps = (src_map, dst_map, edge_map)
        lhs_data = ir.READ_COL(in_frames[self.lhs], var.STR(self.lhs_field))
        rhs_data = ir.READ_COL(in_frames[self.rhs], var.STR(self.rhs_field))
        lhs_map = var.MAP(in_maps[self.lhs])
        rhs_map = var.MAP(in_maps[self.rhs])
        out_map = var.MAP(out_map)
        return ir.BINARY_REDUCE(reducer, self.binary_op, graph, self.lhs,
                                self.rhs, lhs_data, rhs_data, out_size,
                                lhs_map, rhs_map, out_map)

    @property
    def name(self):
        lhs = TargetCode.CODE2STR[self.lhs]
        rhs = TargetCode.CODE2STR[self.rhs]
        return "{}_{}_{}".format(lhs, self.binary_op, rhs)


class CopyMessage(Message):

    def __init__(self, target, in_field, out_field):
        self.target = target
        self.in_field = in_field
        self.out_field = out_field

    def _invoke(self, graph, src_frame, dst_frame, edge_frame, out_size,
                src_map, dst_map, edge_map, out_map, reducer="none"):

        graph = var.GRAPH(graph)
        in_frames = (src_frame, dst_frame, edge_frame)
        in_maps = (src_map, dst_map, edge_map)
        in_data = ir.READ_COL(in_frames[self.target], var.STR(self.in_field))
        in_map = var.MAP(in_maps[self.target])
        out_map = var.MAP(out_map)
        return ir.COPY_REDUCE(reducer, graph, self.target, in_data, out_size,
                              in_map, out_map)

    @property
    def name(self):
        return "copy_{}".format(TargetCode.CODE2STR[self.target])

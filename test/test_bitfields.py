# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest

import auspex.config as config
config.auspex_dummy_mode = True

from auspex.instruments.binutils import BitField, BitFieldUnion

class TestBitFieldUnion(BitFieldUnion):
    num_words  = BitField(16)
    mode       = BitField(4)
    read_write = BitField(1)
    ack        = BitField(1)
    reserved   = BitField(10)

class BitFieldTestCase(unittest.TestCase):
    """
    Tests bitfield union setting/getting
    """
    def test_setter(self):
        test_union = TestBitFieldUnion()
        #Set a bitfield
        test_union.ack = 1
        #Set another field
        self.assertEqual(test_union.packed, 1 << 21)
        test_union.mode = 0xa
        self.assertEqual(test_union.packed, (1 << 21) | (0xa << 16))
        #clear the first field
        test_union.ack = 0
        self.assertEqual(test_union.packed, 0xa << 16)

    def test_getter(self):
        test_union = TestBitFieldUnion()
        test_union.packed = 1 << 21
        self.assertEqual(test_union.ack, 1)

    def test_setter_raise(self):
        """setting too large a value for the field should raise"""
        test_union = TestBitFieldUnion()
        with self.assertRaises(ValueError):
            test_union.mode = 0xab

    def test_constructor_raise(self):
        #trying to set both packed an another field should raise
        with self.assertRaises(AttributeError):
            test_union = TestBitFieldUnion(packed=3, mode=3)

    def test_constructor(self):
        test_union = TestBitFieldUnion(packed=0xbaad)
        self.assertEqual(test_union.packed, 0xbaad)
        test_union = TestBitFieldUnion(ack=1)
        self.assertEqual(test_union.packed, 1 << 21)

if __name__ == '__main__':
    unittest.main()

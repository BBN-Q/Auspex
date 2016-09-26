# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import collections

class BitField(object):
    """Bit field in a bit field union"""
    def __init__(self, width):
        super(BitField, self).__init__()
        self.width = width

class BitFieldUnionMeta(type):
    """Metaclass for injecting bitfield descriptors"""
    @classmethod
    def __prepare__(metacls, name, bases):
        return collections.OrderedDict()

    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)
        self.packed = 0
        init_offset = 0
        for k,v in dct.items():
            if isinstance(v, BitField):
                def fget(self, offset=init_offset, width=v.width):
                    return (self.packed >> offset) & (2**width-1)
                def fset(self, val, offset=init_offset, width=v.width):
                    #check we don't exceed the width of the field
                    if (val & (2**width-1)) != val:
                        err_msg = "attempted to assign value that does not fit in bit field width {:d}".format(width)
                        raise ValueError(err_msg)
                    self.packed &= ~((2**width-1) << offset) #clear the field
                    self.packed |= (val & (2**width-1)) << offset #set the field
                setattr(self, k, property(fget, fset, None, None))
                init_offset += v.width

class BitFieldUnion(metaclass=BitFieldUnionMeta):
    """Painless bit packing"""
    def __init__(self, **kwargs):
        super(BitFieldUnion, self).__init__()
        if "packed" in kwargs and len(kwargs) > 1:
            raise AttributeError("unable to set both `packed` and another bit field")
        for k,v in kwargs.items():
            setattr(self, k, v)

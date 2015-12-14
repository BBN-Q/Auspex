import collections

class BitField(object):
    """Bit field"""
    def __init__(self, width):
        super(BitField, self).__init__()
        self.width = width

class BitFieldUnion(type):
    """Painless bit packing"""
    @classmethod
    def __prepare__(self, name, bases):
        return collections.OrderedDict()

    def __new__(cls, name, bases, classdict):
        classdict['__ordered__'] = [key for key in classdict.keys()
                if key not in ('__module__', '__qualname__')]
        return type.__new__(cls, name, bases, classdict)

    def __init__(self, name, bases, dct):
        type.__init__(self, name, bases, dct)
        self.packed = 0
        init_offset = 0
        for k,v in dct.items():
            if isinstance(v, BitField):
                def fget(self, offset=init_offset, width=v.width):
                    return (self.packed >> offset) & (2**width-1)
                def fset(self, val, offset=init_offset, width=v.width):
                    self.packed &= ~((2**width-1) << offset) #clear the field
                    self.packed |= (val & (2**width-1)) << offset #set the field
                setattr(self, k, property(fget, fset, None, None))
                init_offset += v.width

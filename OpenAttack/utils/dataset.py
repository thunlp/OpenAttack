import json
import random
from ..exceptions import DuplicatedParameterException, UnknownDataLabelException, UnknownDataFormatException

class Dataset(object):
    """

    ``Dataset`` is an useful module that helps you to manage your data. It has a similar
    interface to list, supports "append", "extend", "slice" and "merge" methods.

    """
    def __init__(self, data_list=None, copy=True):
        """
        :param list data_list: This can be a list of :py:class:`.DataInstance`, a list of dict, a list of sentences or a list of tuples of type `(sentence, target)`. **Default:** None
        :param bool copy: Create a copy of :py:class:`.DataInstance` s in data_list. **Default:** True
        """
        self.__data = {}
        self.__next_idx = 0
        if data_list is not None:
            unsolved = []
            for data in data_list:
                if isinstance(data, dict) and ("idx" in data):
                    self.__data[ data["idx"] ] = DataInstance(**data)
                    self.__next_idx = max(self.__next_idx, data["idx"] + 1)
                elif isinstance(data, DataInstance) and data.id is not None:
                    self.__data[ data.id ] = data.copy() if copy else data
                    self.__next_idx = max(self.__next_idx, data.id + 1)
                else:
                    unsolved.append(data)

            for data in unsolved:
                if isinstance(data, tuple):
                    sent, target = data
                    self.__data[self.__next_idx] = DataInstance(
                        sent, target=target, idx=self.__next_idx
                    )
                elif type(data).__name__ == "str":
                    self.__data[self.__next_idx] = DataInstance(
                        data, idx=self.__next_idx
                    )
                elif isinstance(data, dict):
                    data["idx"] = self.__next_idx
                    self.__data[self.__next_idx] = DataInstance(
                        ** data
                    )
                elif isinstance(data, DataInstance):
                    data.id = self.__next_idx
                    self.__data[self.__next_idx] = data.copy() if copy else data
                else:
                    raise UnknownDataFormatException(data)
                self.__next_idx += 1
        else:
            pass

    def append(self, data_instance):
        """
        :param DataInstance data_instance: The :py:class:`.DataInstance` which you want to append.
        :return: Index of this new data instance.
        :rtype: int
        """
        ret_id = self.__next_idx
        data_instance = data_instance.copy()

        data_instance.index = ret_id
        self.__data[ret_id] = data_instance
        self.__next_idx += 1
        return ret_id
    
    def remove(self, index):
        """
        :param int index: The index of data instance that you want to remove.
        :return: Returns False if index is not found, otherwise True.
        :rtype: bool
        """
        if isinstance(index, DataInstance):
            data = index
            index, data.index = data.index, None
        if index in self.__data:
            del self.__data[index]
            return True
        else:
            return False
        
    def shuffle(self, inplace=False, copy=True):
        """
        :param bool inplace: If true, randomly shuffle dataset and returns itself.
        :param bool copy: If true, works on the copy of this dataset. This option would be ignored if ``inplace=True``.
        :rtype: Dataset

        This method would randomly shuffle dataset and reset indexes.
        """
        keys = list(self.__data.keys())
        random.shuffle(keys)
        if inplace:
            nw_dict = {}
            for idx, old_id in enumerate(keys):
                nw_dict[idx] = self.__data[old_id]
                nw_dict[idx].id = idx
            self.__data = nw_dict
            return self
        else:
            ret = [ self.__data[idx].copy() if copy else self.__data[idx] for idx in keys]
            return Dataset(ret, copy=False)
    
    def iter(self, shuffle=False, copy=True):
        """
        :param bool shuffle: Shuffle dataset. **Default:** False
        :param bool copy: Generates copy of data instances. **Default:** True
        :return: Returns iterator of all data instances.
        :rtype: generator
        """
        keys = list(self.__data.keys())
        if shuffle:
            random.shuffle(keys)
        else:
            keys = sorted(keys)
        def generator():
            for kw in keys:
                inst = self.__data[kw]
                yield inst.copy() if copy else inst
        return generator()
    
    def __iter__(self):
        return self.iter(shuffle=False, copy=False)
    
    def eval(self, clsf, batch_size=1, copy=True, ignore_known=True, ignore_meta=False):
        """
        :param Classifier clsf: classifier.
        :param int batch_size: Number of instances in a batch. **Default:** 1
        :param bool copy: Generate a copy for result. **Default:** True
        :param bool ignore_known: Ignore the data instances which already has a predicted label. **Default:** True
        :param bool ignore_meta: Ignore meta of data instances when evaluating. **Default:** False
        :return: Results with predictions.
        :rtype: Dataset

        This method evaluates your classifier on the dataset and generates a result.
        """
        ret = []
        def update(batch):
            batch_x = [ inst.x for inst in batch ]
            res = clsf.get_pred(batch_x)
            for i, inst in enumerate(batch):
                if copy:
                    inst = inst.copy()
                    ret.append(inst)
                inst.pred = res[i]
            return
        
        batch = []
        for kw, val in self.__data.items():
            if ignore_known and val.pred is not None:
                if copy:
                    ret.append(val.copy())
                continue

            if (not ignore_meta) and val.meta != {}:  # for instance with meta
                inst = val.copy() if copy else val
                inst.pred = clsf.get_pred([inst.x], inst.meta)[0]
                if copy:
                    ret.append(inst)
                continue

            batch.append(val)
            if len(batch) < batch_size:
                continue
            update(batch)
            batch = []
        if len(batch) > 0:
            update(batch)
            batch = []
        if copy:
            return Dataset(ret, copy=False)
        else:
            return self
    
    def __check(self, equal, ignore_unknown, keep_ids, copy):
        ret = []
        for kw, val in self.__data.items():
            if val.pred is None or val.y is None:
                if ignore_unknown:
                    continue
                raise UnknownDataLabelException(val)
            if (val.pred != val.y) == equal:
                continue
            inst = val.copy() if copy else val
            if copy and not keep_ids:
                inst.id = None
            ret.append(inst)
        return Dataset(ret, copy=False)
    
    def correct(self, ignore_unknown=True, keep_ids=True, copy=True):
        """
        :param bool ignore_unknown: Ignore the data instances which has no predicted label.
        :param bool keep_ids: Keep index of instances in the result. **Default:** True
        :param bool copy: If true, returns a copy of dataset. **Default:** True
        :rtype: Dataset

        This method returns a subset of the dataset which predicted label is equal to true label.
        """
        return self.__check(True, ignore_unknown, keep_ids, copy)
    
    def wrong(self, ignore_unknown=True, keep_ids=True, copy=True):
        """
        :param bool ignore_unknown: Ignore the data instances which has no predicted label.
        :param bool keep_ids: Keep index of instances in the result. **Default:** True
        :param bool copy: If true, returns a copy of dataset. **Default:** True
        :rtype: Dataset

        This method returns a subset of the dataset which predicted label is not equal to true label.
        """
        return self.__check(False, ignore_unknown, keep_ids, copy)
    
    def sample(self, num, keep_ids=True, copy=True):
        """
        :param int num: Number of sampling results.
        :param bool keep_ids: Keep index of instances in the result. **Default:** True
        :param bool copy: If true, returns a copy of dataset. **Default:** True
        :rtype: Dataset

        This method randomly samples ``num`` instances from the dataset.
        """
        keys = list(self.__data.keys())
        random.shuffle(keys)
        keys = keys[:num]
        keys = sorted(keys)

        ret = []
        for kw in keys:
            inst = self.__data[kw]
            if copy:
                inst = inst.copy()
                if not keep_ids:
                    inst.id = None
            ret.append(inst)
        return Dataset(ret, copy=False)
    
    def filter_label(self, label, keep_ids=True, copy=True):
        """
        :param int label: The ground truth label which you want to keep.
        :param bool keep_ids: Keep index of instances in the result. **Default:** True
        :param bool copy: If true, returns a copy of dataset. **Default:** True
        :return: Returns a subset of the dataset, consists of instances that ground truth label is equal to ``label``.
        :rtype: Dataset
        """
        ret = []
        for kw, val in self.__data.items():
            if val.y == label:
                inst = val.copy() if copy else val
                if copy and not keep_ids:
                    inst.id = None
                ret.append(inst)
        return Dataset(ret, copy=False)
    
    def filter_pred(self, label, keep_ids=True, copy=True):
        """
        :param int label: The predicted label which you want to keep.
        :param bool keep_ids: Keep index of instances in the result. **Default:** True
        :param bool copy: If true, returns a copy of dataset. **Default:** True
        :return: Returns a subset of the dataset, consists of instances that predicted label is equal to ``label``.
        :rtype: Dataset
        """
        ret = []
        for kw, val in self.__data.items():
            if val.pred == label:
                inst = val.copy() if copy else val
                if copy and not keep_ids:
                    inst.id = None
                ret.append(inst)
        return Dataset(ret, copy=False)
    
    def __setitem__(self, index, val):
        if not isinstance(index, int):
            raise TypeError("Key '%s' is not supported." % repr(index))
        if isinstance(val, DataInstance):
            val = val.copy()
            val.id = index
            self.__data[index] = val
        else:
            raise TypeError("Object '%s' is not allowd." % repr(val))

    def __getitem__(self, index):
        if isinstance(index, int) and not isinstance(index, bool):
            if index in self.__data:
                return self.__data[index]
            raise KeyError(index)
        elif isinstance(index, slice):
            st = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else self.__next_idx
            step = index.step if index.step is not None else 1
            if st < 0 or stop < 0:
                return Dataset()
            if st >= stop or step <= 0:
                return Dataset()

            if step > 0:
                ret = []
                while st < stop:
                    if st in self.__data:
                        ret.append(self.__data[st])
                    st += step
            
            return Dataset(ret, copy=False)
        elif isinstance(index, bool):
            if index:
                return self.correct(ignore_unknown=True, keep_ids=True, copy=False)
            else:
                return self.wrong(ignore_unknown=True, keep_ids=True, copy=False)
        elif index is None:
            return self.filter_pred(None, keep_ids=True, copy=False)
        else:
            raise KeyError(index)
    
    def extend(self, dataset_b, copy=True, inplace=True):
        """
        :param Dataset dataset_b: Another dataset.
        :param bool copy: Use the copy of ``dataset_b``. **Default:** True
        :param bool inplace: If true, returns self, otherwise returns a new :py:class:`.Dataset` **Default:** True
        :return: Returns a dataset of result.
        :rtype: Dataset

        This method would reset indexes of ``dataset_b``.
        """
        assert isinstance(dataset_b, Dataset)
        ret = []
        for inst in dataset_b:
            if copy:
                inst = inst.copy()
            if inplace:
                inst.id = self.__next_idx
                self.__data[self.__next_idx] = inst
                self.__next_idx += 1
            else:
                ret.append(inst)
                inst.id = self.__next_idx + len(ret)
        if inplace:
            return self
        else:
            for kw, val in self.__data.items():
                inst = val.copy() if copy else val
                ret.append(inst)
            return Dataset(ret, copy=False)
    
    def merge(self, dataset_b, copy=True, inplace=True):
        """
        :param Dataset dataset_b: Another dataset.
        :param bool copy: Use the copy of ``dataset_b``. **Default:** True
        :param bool inplace: If true, returns self, otherwise returns a new :py:class:`.Dataset` **Default:** True
        :return: Returns a dataset of result.
        :rtype: Dataset

        This method merges two dataset and keeps indexes of all instances. If indexes are conflicted, the instance in ``dataset_b`` would be ignored.
        """
        assert isinstance(dataset_b, Dataset)
        ret = []
        for inst in dataset_b:
            if inst.id not in self.__data:
                if copy:
                    inst = inst.copy()
                if inplace:
                    self.__data[inst.id] = inst
                else:
                    ret.append(inst)
        if not inplace:
            for kw, val in self.__data.items():
                if copy:
                    inst = val.copy()
                else:
                    inst = val
                ret.append(inst)
            return Dataset(ret, copy=False)
        else:
            return self

    def __len__(self):
        return len(self.__data)     
    
    def __contains__(self, key):
        return key in self.__data
    
    def __iadd__(self, val):
        return self.extend(val, copy=True, inplace=True)

    def __delitem__(self, index):
        if isinstance(index, int):
            if index in self.__data:
                del self.__data[index]
            raise KeyError(index)
        elif isinstance(index, slice):
            st = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else self.__next_idx
            step = index.step if index.step is not None else 1
            if st < 0 or stop < 0:
                return
            if st >= stop or step <= 0:
                return
            while st < stop:
                if st in self.__data:
                    del self.__data[st]
                st += step
        elif isinstance(index, bool):
            for inst in self.__check(index, ignore_unknown=True, keep_ids=True, copy=False):
                del self.__data[inst.id]
        else:
            raise KeyError(index)
    
    def __add__(self, dataset_b):
        assert isinstance(dataset_b, Dataset)
        ret = []
        for kw, val in self.__data.items():
            inst = val.copy()
            inst.id = None
            ret.append(inst)
        for inst in dataset_b:
            inst = inst.copy()
            inst.id = None
            ret.append(inst)
        return Dataset(ret, copy=False)
    
    def data(self, copy=True):
        """
        :param bool copy: Use the copy of dataset. **Default:** True
        :return: A list contains all instances of this dataset.
        :rtype: list

        This method returns a dict which is JSON-serializable, and can be used to initialize :py:class:`.Dataset`. See :py:meth:`.Dataset.__init__`.
        """
        ret = []
        for kw, val in self.__data.items():
            ret.append(val.data(copy=copy))
        return ret
    
    def clear_pred(self, copy=False):
        """
        :param bool copy: Use the copy of dataset. If false, it would modify in place. **Default:** False
        :rtype: Dataset

        This method clears the predicted label for all instances (sets it to None), and returns the result dataset.
        """
        ret = []
        for kw, val in self.__data.items():
            if copy:
                inst = val.copy()
                ret.append(inst)
            else:
                inst = val
            inst.pred = None
        if copy:
            return Dataset(ret, copy=False)
        else:
            return self
    
    def clear_label(self, copy=False):
        """
        :param bool copy: Use the copy of dataset. If false, it would modify in place. **Default:** False
        :rtype: Dataset

        This method clears the ground truth label for all instances (sets it to None), and returns the result dataset.
        """
        ret = []
        for kw, val in self.__data.items():
            if copy:
                inst = val.copy()
                ret.append(inst)
            else:
                inst = val
            inst.y = None
        if copy:
            return Dataset(ret, copy=False)
        else:
            return self
    
    def copy(self):
        """
        :return: Returns a copy of this dataset.
        :rtype: Dataset
        """
        return Dataset( self.data() )
    
    def reset_index(self, inplace=False):
        """
        :param bool inplace: If true, resets in place, otherwise works on a copy of dataset. **Default:** False
        :return: A result dataset.
        :rtype: Dataset

        This method resets indexes of all instances to ``0 ... len(dataset) - 1``
        """
        ret = []
        for kw, val in self.__data.items():
            if inplace:
                inst = val
            else:
                inst = val.copy()
            ret.append(inst)
            inst.id = None
        
        if inplace:
            self.__data = {}
            self.__next_idx = 0
            for inst in ret:
                self.append(inst)
            return self
        else:
            return Dataset(ret)


class DataInstance(object):
    __KEY_MAP = {
        "id": ["id", "idx", "index"],
        "x": ["x", "x_orig", "sent", "sentence"],
        "y": ["y", "y_orig", "label"],
        "pred": ["pred", "y_pred"],
        "target": ["target"],
        "meta": ["meta"]
    }
    __KEY_ORDER = [
        "x", "y", "pred", "target", "meta", "id"
    ]
    def __find_key(self, kwargs, keys, default=None):
        ret = None
        okw = None
        for kw in keys:
            if kw in kwargs:
                if ret is None:
                    ret = kwargs[kw]
                    okw = kw
                else:
                    if kwargs[kw] != ret:
                        raise DuplicatedParameterException("%s = %d; %s = %d" % (okw, ret, kw, kwargs[kw]))
        if ret is None:
            ret = default
        return ret

    def __init__(self, *args, **kwargs):
        """
        :param str x: Original input sentence.
        :param int y: Ground truth label of input x. **Default:** None
        :param int pred: Predicted label. **Default:** None
        :param int target: Target label in attacker, leaves None for untargeted attack. **Default:** None
        :param int id: Index of data instance in dataset, leaves None for standalone data instance. **Default:** None
        :param dict meta: Meta data of this instance.
        """
        for i, val in enumerate(args):
            if i >= len(self.__KEY_ORDER):
                raise TypeError("__init__() takes %d positional argument but %d were given" % (len(self.__KEY_ORDER), len(args)))
            kwargs[self.__KEY_ORDER[i]] = val
        self.__x_orig = self.__find_key(kwargs, self.__KEY_MAP["x"], None)
        assert self.__x_orig is not None, "'X' shouldn't be None."
        self.__y_orig = self.__find_key(kwargs, self.__KEY_MAP["y"], None)
        self.__pred = self.__find_key(kwargs, self.__KEY_MAP["pred"], None)
        self.__meta = self.__find_key(kwargs, self.__KEY_MAP["meta"], {})
        self.__target = self.__find_key(kwargs, self.__KEY_MAP["target"], None)
        self.__id = self.__find_key(kwargs, self.__KEY_MAP["id"], None)
        

    def __getattr__(self, name):
        if name in self.__KEY_MAP["id"]:
            return self.__id
        elif name in self.__KEY_MAP["x"]:
            return self.__x_orig
        elif name in self.__KEY_MAP["y"]:
            return self.__y_orig
        elif name in self.__KEY_MAP["pred"]:
            return self.__pred
        elif name in self.__KEY_MAP["target"]:
            return self.__target
        elif name in self.__KEY_MAP["meta"]:
            return self.__meta
        elif name.startswith("_"):
            return super().__getattr__(name)
        elif name in self.__meta:
            return self.__meta[name]
        raise AttributeError(
            "'%s' object has no attribute '%s'\nmeta: %s"
            % (self.__class__.__name__, name, self.__meta.__repr__())
        )

    def __setattr__(self, name, value):
        if name in self.__KEY_MAP["id"]:
            self.__id = value
        elif name in self.__KEY_MAP["x"]:
            assert value is not None
            self.__x_orig = value
        elif name in self.__KEY_MAP["y"]:
            self.__y_orig = value
        elif name in self.__KEY_MAP["pred"]:
            self.__pred = value
        elif name in self.__KEY_MAP["target"]:
            self.__target = value
        elif name in self.__KEY_MAP["meta"]:
            assert isinstance(value, dict)
            self.__meta = value
        elif name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self.__meta[name] = value

    def __delattr__(self, name):
        if name in self.__KEY_MAP["id"]:
            self.__id = None
        elif name in self.__KEY_MAP["x"]:
            raise AttributeError("'%s' doesn't support attribute deletion" % name)
        elif name in self.__KEY_MAP["y"]:
            self.__y_orig = None
        elif name in self.__KEY_MAP["pred"]:
            self.__pred = None
        elif name in self.__KEY_MAP["target"]:
            self.__target = None
        elif name in self.__KEY_MAP["meta"]:
            self.__meta = {}
        elif name.startswith("_"):
            super().__delattr__(name)
        elif name in self.__meta:
            del self.__meta[name]
        else:
            raise AttributeError(
                "'%s' object has no attribute '%s'\nmeta: %s"
                % (self.__class__.__name__, name, self.__meta.__repr__())
            )

    def __repr__(self):
        ret = "<%s" % self.__class__.__name__
        if self.__id is not None:
            ret += " index=%d" % self.__id
        if len(self.__x_orig) > 20:
            ret += " x=%s..." % self.__x_orig[:18]
        else:
            ret += " x=%s" % self.__x_orig

        if self.__y_orig is not None:
            ret += " y=%d" % self.__y_orig
        if self.__pred is not None:
            ret += " pred=%d" % self.__pred
        if self.__meta is not None:
            ret += " meta=%s" % self.__meta.__repr__() 
        ret += ">"
        return ret

    def __str__(self):
        ret = "%s" % self.__class__.__name__
        if self.__id is not None:
            ret += " %d\n" % self.__id
        else:
            ret += "\n"

        ret += "x:\t%s\n" % self.__x_orig
        if self.__y_orig is not None:
            ret += "y:\t%d\n" % self.__y_orig
        else:
            ret += "y:\tunknown\n"
        if self.__pred is not None:
            ret += "pred:\t%d\n" % self.__pred
        else:
            ret += "pred:\tunknown\n"
        if self.__target is not None:
            ret += "target:\t%d\n" % self.__target
        ret += "meta:\t%s" % self.__meta.__repr__()
        return ret
    
    def __contains__(self, name):
        try:
            val = self.__getattr__(name)
        except AttributeError:
            return False
        return val is not None

    def copy(self):
        """
        :rtype: DataInstance

        This method returns a copy of itself.
        """
        return DataInstance(
            x_orig=self.__x_orig,
            y_orig=self.__y_orig,
            pred=self.__pred,
            target=self.__target,
            idx=self.__id,
            meta=self.__meta.copy(),
        )
    
    def data(self, copy=True):
        """
        :rtype: dict

        This method returns a dict contains all information of this instance.
        """
        ret = { "x_orig": self.__x_orig }
        if self.__y_orig is not None:
            ret["y_orig"] = self.__y_orig
        if self.__pred is not None:
            ret["prd"] = self.__pred
        if self.__target is not None:
            ret["target"] = self.__target
        if self.__id is not None:
            ret["idx"] = self.__id
        ret["meta"] = self.__meta.copy() if copy else self.__meta
        return ret

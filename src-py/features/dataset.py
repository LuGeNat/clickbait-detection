#!/usr/bin/python3
import json
import numpy as np


class ClickbaitDataset(object):
    # TODO switch to pandas
    def __init__(self, instances_path=None, truth_path=None):
        self.dataset_dict = {}
        if instances_path is not None and truth_path is not None:
            self.from_file(instances_path, truth_path)
        elif truth_path is None and instances_path is not None:
            with open(instances_path, "r") as inf:
                _instances = [json.loads(x) for x in inf.readlines()]
            for i in _instances:
                if i['text'] is not []:
                    self.dataset_dict[i['id']] = {'text': i['text'],
                                                  'id': i['id']}

    def from_file(self, instances_path, truth_path):
        with open(instances_path, "r") as inf:
            _instances = [json.loads(x) for x in inf.readlines()]
        with open(truth_path, "r") as inf:
            _truth = [json.loads(x) for x in inf.readlines()]

        for i in _instances:
            if i['text'] is not []:
                self.dataset_dict[i['id']] = {'text': i['text']}
            else:
                pass

        for t in _truth:
            if t['id'] in self.dataset_dict.keys():
                if t['gender'] == 'male':
                    self.dataset_dict[t['id']]['trueGender'] = 0
                elif t['gender'] == 'female':
                    self.dataset_dict[t['id']]['trueGender'] = 1
                else:
                    self.dataset_dict[t['id']]['trueGender'] = 2

        # self.id_index = {index: key for index, key in enumerate(self.dataset_dict.keys())}

    def add_feed(self, feet_id, text=''):
        self.dataset_dict[feet_id] = {'text': text,
                                       'id': feet_id}
        return self

    def get_y(self):
        return np.asarray([self.dataset_dict[key]['trueGender'] for key in sorted(self.dataset_dict.keys())])

    def get_y_class(self):
        class_list = [self.dataset_dict[key]['trueGender'] for key in sorted(self.dataset_dict.keys())]
        return np.asarray([0 if t == 'male' or t == 'female' else 1 for t in class_list])

    def get_x(self, field_name):
        _result = []
        for key in sorted(self.dataset_dict.keys()):
            _result.append(''.join(self.dataset_dict[key][field_name]))
        return _result

    def get_x_annotated(self):
        _result = []
        for value in self.dataset_dict.values():
            _result.append(value)
        return _result

    def size(self):
        return len(self.dataset_dict.keys())


if __name__ == "__main__":
    pass

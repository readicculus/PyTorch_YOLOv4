import os
from decimal import Decimal

import mlflow
import numpy as np

class DNConfig():
    def __init__(self, cfg_path):
        self.cfg = self.parse_model_cfg(cfg_path)
        self.net_cfg = self.cfg[0]
        self.cfg_path = cfg_path

    def is_decimal(self, v):
        try:
            float(v)
            return True
        except: return False

    def is_int(self, v):
        try:
            int(v)
            return True
        except: return False

    def parse_model_cfg(self, path):
        # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
        if not path.endswith('.cfg'):  # add .cfg suffix if omitted
            path += '.cfg'
        if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):  # add cfg/ prefix if omitted
            path = 'cfg' + os.sep + path

        with open(path, 'r') as f:
            lines = f.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        mdefs = []  # module definitions
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                mdefs.append({})
                mdefs[-1]['type'] = line[1:-1].rstrip()
                if mdefs[-1]['type'] == 'convolutional':
                    mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
            else:
                key, val = line.split("=")
                key = key.rstrip()

                if key == 'anchors':  # return nparray
                    mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
                elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                    mdefs[-1][key] = [int(x) for x in val.split(',')]
                else:
                    val = val.strip()
                    if self.is_int(val):  # return int or float
                        mdefs[-1][key] = int(val)
                    elif self.is_decimal(val):
                        mdefs[-1][key] = Decimal(val)
                    else:
                        mdefs[-1][key] = val  # return string

        # Check all fields are supported
        supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                     'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                     'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                     'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'conv_lstm', 'Gaussian_yolo']

        f = []  # fields
        for x in mdefs[1:]:
            [f.append(k) for k in x if k not in f]
        u = [x for x in f if x not in supported]  # unsupported fields
        assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

        return mdefs

    def __repr__(self):
        return self.cfg_path

    def __str__(self):
        lines = []
        for module in self.cfg:
            lines.append('[%s]' % module['type'])
            for k in module:
                val = module[k]
                if k == 'type':
                    continue
                elif k == 'anchors':
                    int_to_str = [str(x) for x in val.reshape(-1).astype(np.int)]
                    anchor_str = ', '.join(int_to_str)
                    lines.append('%s = %s' % (k, anchor_str))
                elif (k in ['from', 'layers', 'mask']) or (k == 'size' and ',' in str(val)):
                    val_str = ', '.join([str(v) for v in val])
                    lines.append('%s = %s' % (k, str(val_str)))
                else:
                    lines.append('%s = %s' % (k, str(val)))
            # append a blank line between all modules
            lines.append('')
        return '\n'.join(lines)

    def mlflow_log_params(self , prefix = ''):
        for k in self.net_cfg:
            mlflow.log_param(''+k, self.net_cfg[k])
class DNData(object):
    def __init__(self, n_classes=None, f_train=None, f_valid=None, f_test=None, f_names=None, dir_backup=None):
        self.n_classes = n_classes
        self.f_train = f_train
        self.f_test = f_test
        self.f_valid = f_valid
        self.f_names = f_names
        self.dir_backup = dir_backup

    def load(self, file):
        with open(file) as f:
            vars = {}
            for line in f:
                name, var = line.split('=')
                vars[name.strip()] = var.strip()

            self.n_classes = int(vars['classes'])
            self.f_train = vars['train']
            self.f_test = vars['test']
            self.f_valid = vars['valid']
            self.f_names = vars['names']
            self.dir_backup = vars['backup']

    def get_names(self):
        names = []
        with open(self.f_names) as f:
            for line in f:
                names.append(line.strip())
        return names

    def get_train_images(self):
        return open(self.f_train).read().split('\n')

    def get_test_images(self):
        return open(self.f_test).read().split('\n')

    def save(self, file):
        try:
            with open(file, "w") as f:
                f.write('classes = %d\n' % self.n_classes)
                f.write('train = %s\n' % self.f_train)
                f.write('test = %s\n' % self.f_test)
                f.write('valid = %s\n' % self.f_valid)
                f.write('names = %s\n' % self.f_names)
                f.write('backup = %s' % self.dir_backup)
        except:
            return False
        return True
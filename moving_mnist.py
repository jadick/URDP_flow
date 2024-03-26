import socket
import numpy as np
from torchvision import datasets, transforms

class MovingMNIST(object):

    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=False):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 28 #int(image_size/2)
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                dx = np.random.randint(-4, 5)
                dy = np.random.randint(-4, 5)
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-self.digit_size:
                    sy = image_size-self.digit_size-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-self.digit_size:
                    sx = image_size-self.digit_size-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)

                x[t, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x
    
class MovingMNIST_unidir_test(object):

    """Data Handler that creates uniderectional MNIST dataset on the fly."""
    '''         Each video contains motion in only one direction         '''

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=False):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 28 
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N
    def set_directions(self, direction, step = 5):
        if direction == 'up':
            dy = np.random.randint(1, step)
            dx = 0
        elif direction == 'down' :
            dy = np.random.randint(-step+1, 0)
            dx = 0
        elif direction == 'left':
            dx = np.random.randint(-step+1, 0)
            dy = 0
        elif direction == 'right':
            dx = np.random.randint(1, step)
            dy = 0

        return dx, dy

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)
        directions = ['up', 'down', 'left', 'right']
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]
            direction = directions[np.random.randint(0, 4)]
            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)

            for t in range(self.seq_len):
                dx, dy = self.set_directions(direction)
                if sy < 0:
                    sy = 0
                    if direction == 'down':
                        direction = 'up'
                    if self.deterministic:
                        dy = -dy
                    else:
                        dx, dy = self.set_directions(direction)
                elif sy >= image_size-self.digit_size:
                    sy = image_size-self.digit_size-1
                    if direction == 'up':
                        direction = 'down'
                    if self.deterministic:
                        dy = -dy
                    else:
                        dx, dy = self.set_directions(direction)

                if sx < 0:
                    sx = 0
                    if direction == 'left':
                        direction = 'right'
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx, dy = self.set_directions(direction)
                elif sx >= image_size-self.digit_size:
                    sx = image_size-self.digit_size-1
                    if direction == 'right':
                        direction = 'left'
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx, dy = self.set_directions(direction)

                x[t, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x
    
class MovingMNIST_custom_step(object):

    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=1, image_size=64, step=10, deterministic=False):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 28 #int(image_size/2)
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1
        self.step = step

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)
        step = self.step
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-step+1, step)
            dy = np.random.randint(-step+1, step)
            for t in range(self.seq_len):
                dx = np.random.randint(-step+1, step)
                dy = np.random.randint(-step+1, step)
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, step)
                        dx = np.random.randint(-step+1, step)
                elif sy >= image_size-self.digit_size:
                    sy = image_size-self.digit_size-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-step+1, 0)
                        dx = np.random.randint(-step+1, step)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, step)
                        dy = np.random.randint(-step+1, step)
                elif sx >= image_size-self.digit_size:
                    sx = image_size-self.digit_size-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-step+1, 0)
                        dy = np.random.randint(-step+1, step)

                x[t, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x
    

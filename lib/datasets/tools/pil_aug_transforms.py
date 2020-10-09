#!/usr/bin/env python
# -*- coding:utf-8 -*-


import collections
import random
import math
import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from lib.utils.tools.logger import Logger as Log


class RandomPad(object):
    """ Padding the Image to proper size.
            Args:
                stride: the stride of the network.
                pad_value: the value that pad to the image border.
                img: Image object as input.
            Returns::
                img: Image object.
    """
    def __init__(self, target_size=None, ratio=0.5, mean=(104, 117, 123)):
        self.target_size = target_size
        self.ratio = ratio
        self.mean = tuple(mean)

    def __call__(self, img):
        assert isinstance(img, (Image.Image, list))

        if random.random() > self.ratio:
            return img

        width, height = img.size
        pad_width = self.target_size[0] - width if self.target_size[0] > width else width - self.target_size[0]
        pad_height = self.target_size[1] - height if self.target_size[1] > height else height - self.target_size[1]
        left_pad = random.randint(0, pad_width) if self.target_size[0] > width else -random.randint(0, pad_width)
        up_pad = random.randint(0, pad_height) if self.target_size[1] > height else -random.randint(0, pad_height)
        right_pad = pad_width - left_pad if self.target_size[0] > width else -pad_width - left_pad
        down_pad = pad_height - up_pad if self.target_size[0] > height else -pad_height - up_pad
        img = ImageOps.expand(img, border=(left_pad, up_pad, right_pad, down_pad), fill=tuple(self.mean))
        return img


class RandomFlip(object):
    def __init__(self, flip90=False, ratio=0.5):
        self.flip90 = flip90
        self.ratio = ratio

    def __call__(self, img):
        assert isinstance(img, Image.Image)

        if random.random() > self.ratio:
            return img

        if not self.flip90:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            method = [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90, Image.ROTATE_270]
            img = img.transpose(method[random.randint(0, len(method) - 1)])

        return img


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = ratio
        assert self.upper >= self.lower, "saturation upper must be >= lower."
        assert self.lower >= 0, "saturation lower must be non-negative."

    def __call__(self, img):
        assert isinstance(img, Image.Image)

        if random.random() > self.ratio:
            return img

        img = np.array(img).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        img[:, :, 1] *= random.uniform(self.lower, self.upper)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = np.clip(img, 0, 255)
        return Image.fromarray(img.astype(np.uint8))


class RandomHue(object):
    def __init__(self, delta=18, ratio=0.5):
        assert 0 <= delta <= 360
        self.delta = delta
        self.ratio = ratio

    def __call__(self, img):
        assert isinstance(img, Image.Image)

        if random.random() > self.ratio:
            return img

        img = np.array(img).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        img[:, :, 0] += random.uniform(-self.delta, self.delta)
        img[:, :, 0][img[:, :, 0] > 360] -= 360
        img[:, :, 0][img[:, :, 0] < 0] += 360
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = np.clip(img, 0, 255)
        return Image.fromarray(img.astype(np.uint8))


class RandomPerm(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img):
        assert isinstance(img, Image.Image)

        if random.random() > self.ratio:
            return img

        swap = self.perms[random.randint(0, len(self.perms)-1)]
        img = np.array(img)
        img = img[:, :, swap]
        return Image.fromarray(img.astype(np.uint8))


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = ratio
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img):
        assert isinstance(img, Image.Image)

        if random.random() > self.ratio:
            return img

        img = np.array(img).astype(np.float32)
        img *= random.uniform(self.lower, self.upper)
        img = np.clip(img, 0, 255)

        return Image.fromarray(img.astype(np.uint8))


class RandomBrightness(object):
    def __init__(self, shift_value=30, ratio=0.5):
        self.shift_value = shift_value
        self.ratio = ratio

    def __call__(self, img):
        assert isinstance(img, Image.Image)

        if random.random() > self.ratio:
            return img

        shift = np.random.uniform(-self.shift_value, self.shift_value, size=1)
        image = np.array(img).astype(np.float32)
        image[:, :, :] += shift
        image = np.around(image)
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)

        return image


class RandomGaussBlur(object):
    def __init__(self, max_blur=4, ratio=0.5):
        self.max_blur = max_blur
        self.ratio = ratio

    def __call__(self, img):
        assert isinstance(img, Image.Image)

        if random.random() > self.ratio:
            return img

        blur_value = np.random.uniform(0, self.max_blur)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_value))
        return img


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
    """

    def __init__(self, h_range, s_range, v_range, ratio=0.5):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range
        self.ratio = ratio

    def __call__(self, img):
        assert isinstance(img, Image.Image)

        if random.random() > self.ratio:
            return img

        img = np.array(img)
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v * v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return Image.fromarray(img_new.astype(np.uint8))


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale_range=(0.08, 1.0), aspect_range=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale_range
        self.ratio = aspect_range

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = img.crop((j, i, j + w, i + h))
        img = img.resize(self.size, Image.BILINEAR)
        return img


class RandomResize(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, scale_range=(0.75, 1.25), aspect_range=(0.9, 1.1),
                 target_size=None, resize_bound=None, method='random', ratio=0.5):
        self.scale_range = scale_range
        self.aspect_range = aspect_range
        self.resize_bound = resize_bound
        self.method = method
        self.ratio = ratio

        if target_size is not None:
            if isinstance(target_size, int):
                self.input_size = (target_size, target_size)
            elif isinstance(target_size, (list, tuple)) and len(target_size) == 2:
                self.input_size = target_size
            else:
                raise TypeError('Got inappropriate size arg: {}'.format(target_size))
        else:
            self.input_size = None

    def get_scale(self, img_size):
        if self.method == 'random':
            scale_ratio = random.uniform(self.scale_range[0], self.scale_range[1])
            return scale_ratio

        elif self.method == 'bound':
            scale1 = self.resize_bound[0] / min(img_size)
            scale2 = self.resize_bound[1] / max(img_size)
            scale = min(scale1, scale2)
            return scale

        else:
            Log.error('Resize method {} is invalid.'.format(self.method))
            exit(1)

    def __call__(self, img):
        """
        Args:
            img     (Image):   Image to be resized.

        Returns:
            Image:  Randomly resize image.
        """
        assert isinstance(img, Image.Image)

        width, height = img.size
        if random.random() < self.ratio:
            scale_ratio = self.get_scale([width, height])
            aspect_ratio = random.uniform(*self.aspect_range)
            w_scale_ratio = math.sqrt(aspect_ratio) * scale_ratio
            h_scale_ratio = math.sqrt(1.0 / aspect_ratio) * scale_ratio
        else:
            w_scale_ratio, h_scale_ratio = 1.0, 1.0

        converted_size = (int(width * w_scale_ratio), int(height * h_scale_ratio))
        img = img.resize(converted_size, Image.BILINEAR)
        return img


class RandomRotate(object):
    """Rotate the input numpy.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """

    def __init__(self, max_degree, ratio=0.5, mean=(104, 117, 123)):
        assert isinstance(max_degree, int)
        self.max_degree = max_degree
        self.ratio = ratio
        self.mean = tuple(mean)

    def __call__(self, img):
        """
        Args:
            img    (Image):     Image to be rotated.

        Returns:
            Image:     Rotated image.
        """
        assert isinstance(img, Image.Image)

        if random.random() < self.ratio:
            rotate_degree = random.uniform(-self.max_degree, self.max_degree)
        else:
            return img

        img = np.array(img)
        height, width, _ = img.shape

        img_center = (width / 2.0, height / 2.0)

        rotate_mat = cv2.getRotationMatrix2D(img_center, rotate_degree, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - img_center[0]
        rotate_mat[1, 2] += (new_height / 2.) - img_center[1]
        img = cv2.warpAffine(img, rotate_mat, (new_width, new_height), borderValue=self.mean)
        img = Image.fromarray(img.astype(np.uint8))
        return img


class RandomCrop(object):
    """Crop the given numpy.ndarray and  at a random location.

    Args:
        size (int or tuple): Desired output size of the crop.(w, h)
    """

    def __init__(self, crop_size, ratio=0.5, method='focus', grid=None):
        self.ratio = ratio
        self.method = method
        self.grid = grid

        if isinstance(crop_size, float):
            self.size = (crop_size, crop_size)
        elif isinstance(crop_size, collections.Iterable) and len(crop_size) == 2:
            self.size = crop_size
        else:
            raise TypeError('Got inappropriate size arg: {}'.format(crop_size))

    def get_lefttop(self, crop_size, img_size):
        if self.method == 'center':
            return [(img_size[0] - crop_size[0]) // 2, (img_size[1] - crop_size[1]) // 2]

        elif self.method == 'random':
            x = random.randint(0, img_size[0] - crop_size[0])
            y = random.randint(0, img_size[1] - crop_size[1])
            return [x, y]

        elif self.method == 'grid':
            grid_x = random.randint(0, self.grid[0] - 1)
            grid_y = random.randint(0, self.grid[1] - 1)
            x = grid_x * ((img_size[0] - crop_size[0]) // (self.grid[0] - 1))
            y = grid_y * ((img_size[1] - crop_size[1]) // (self.grid[1] - 1))
            return [x, y]

        else:
            Log.error('Crop method {} is invalid.'.format(self.method))
            exit(1)

    def __call__(self, img):
        """
        Args:
            img (Image):   Image to be cropped.

        Returns:
            Image:  Cropped image.
        """
        assert isinstance(img, Image.Image)

        if random.random() > self.ratio:
            return img

        target_size = (min(self.size[0], img.size[0]), min(self.size[1], img.size[1]))
        offset_left, offset_up = self.get_lefttop(target_size, img.size)
        img = img.crop((offset_left, offset_up, offset_left + target_size[0], offset_up + target_size[1]))
        return img


class Resize(object):
    def __init__(self, target_size=None, min_side_length=None, max_side_length=None):
        self.target_size = target_size
        self.min_side_length = min_side_length
        self.max_side_length = max_side_length

    def __call__(self, img):
        assert isinstance(img, Image.Image)

        width, height = img.size
        if self.target_size is not None:
            target_size = self.target_size

        elif self.min_side_length is not None:
            scale_ratio = self.min_side_length / min(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            target_size = [int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))]

        else:
            scale_ratio = self.max_side_length / max(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            target_size = [int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))]

        img = img.resize(target_size, Image.BILINEAR)
        return img

class JigSaw(object):
    def __init__(self,n=7,ratio=0.5):
        self.n = n
        self.ratio = ratio

    def crop_image(self,image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list

    def __call__(self,img):
        assert isinstance(img, Image.Image)

        if random.random() > self.ratio:
            return img

        widthcut, highcut = img.size
        # img = img.crop((10, 10, widthcut-10, highcut-10))
        crop=[self.n,self.n]
        images = self.crop_image(img, crop)
        pro = 5
        if pro >= 5:          
            tmpx = []
            tmpy = []
            count_x = 0
            count_y = 0
            k = 1
            RAN = 2
            for i in range(crop[1] * crop[0]):
                tmpx.append(images[i])
                count_x += 1
                if len(tmpx) >= k:
                    tmp = tmpx[count_x - RAN:count_x]
                    random.shuffle(tmp)
                    tmpx[count_x - RAN:count_x] = tmp
                if count_x == crop[0]:
                    tmpy.append(tmpx)
                    count_x = 0
                    count_y += 1
                    tmpx = []
                if len(tmpy) >= k:
                    tmp2 = tmpy[count_y - RAN:count_y]
                    random.shuffle(tmp2)
                    tmpy[count_y - RAN:count_y] = tmp2
            random_im = []
            for line in tmpy:
                random_im.extend(line)
            
            # random.shuffle(images)
            width, high = img.size
            iw = int(width / crop[0])
            ih = int(high / crop[1])
            toImage = Image.new('RGB', (iw * crop[0], ih * crop[1]))
            x = 0
            y = 0
            for i in random_im:
                i = i.resize((iw, ih), Image.ANTIALIAS)
                toImage.paste(i, (x * iw, y * ih))
                x += 1
                if x == crop[0]:
                    x = 0
                    y += 1
        else:
            toImage = img
        toImage = toImage.resize((widthcut, highcut))
        return toImage

PIL_AUGMENTATIONS_DICT = {
    'random_saturation': RandomSaturation,
    'random_hue': RandomHue,
    'random_perm': RandomPerm,
    'random_contrast': RandomContrast,
    'random_brightness': RandomBrightness,
    'random_gauss_blur': RandomGaussBlur,
    'random_hsv': RandomHSV,
    'random_pad': RandomPad,
    'random_flip': RandomFlip,
    'random_hflip': RandomFlip,
    'random_resize': RandomResize,
    'random_crop': RandomCrop,
    'random_resized_crop': RandomResizedCrop,
    'random_rotate': RandomRotate,
    'resize': Resize,
    'jigsaw':JigSaw
}


class PILAugCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> PILAugCompose([
        >>>     RandomCrop(),
        >>> ])
    """

    def __init__(self, configer, split='train'):
        self.configer = configer
        self.transforms = dict()
        self.split = split
        aug_trans = self.configer.get(split, 'aug_trans')
        shuffle_train_trans = []
        if 'shuffle_trans_seq' in aug_trans:
            if isinstance(aug_trans['shuffle_trans_seq'][0], list):
                train_trans_seq_list = aug_trans['shuffle_trans_seq']
                for train_trans_seq in train_trans_seq_list:
                    shuffle_train_trans += train_trans_seq

            else:
                shuffle_train_trans = aug_trans['shuffle_trans_seq']

        for trans in aug_trans['trans_seq'] + shuffle_train_trans:
            if trans in PIL_AUGMENTATIONS_DICT:
                self.transforms[trans] = PIL_AUGMENTATIONS_DICT[trans](**aug_trans[trans])
            else:
                Log.info("{} not implemented in PIL".format(trans))

    def __call__(self, img):
        aug_trans = self.configer.get(self.split, 'aug_trans')
        shuffle_trans_seq = []
        if 'shuffle_trans_seq' in aug_trans:
            if isinstance(aug_trans['shuffle_trans_seq'][0], list):
                shuffle_trans_seq_list = aug_trans['shuffle_trans_seq']
                shuffle_trans_seq = shuffle_trans_seq_list[random.randint(0, len(shuffle_trans_seq_list))]
            else:
                shuffle_trans_seq = aug_trans['shuffle_trans_seq']
                random.shuffle(shuffle_trans_seq)

        for trans_key in (shuffle_trans_seq + aug_trans['trans_seq']):
            if trans_key in self.transforms:
                img = self.transforms[trans_key](img)

        return img

#!/usr/bin/env python
# -*- coding:utf-8 -*-


import collections
import random
import math
import cv2
import numpy as np

import imgaug.augmenters as iaa
from lib.utils.tools.logger import Logger as Log


class RandomErase(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, ratio=0.5, erase_range=(0.02, 0.4), aspect=0.3, mean=[104, 117, 123]):
        self.ratio = ratio
        self.mean = mean
        self.erase_range = erase_range
        self.aspect = aspect

    def __call__(self, img):
        if random.uniform(0, 1) > self.ratio:
            return img

        height, width, channels = img.shape
        for _ in range(100):
            area = height * width
            target_area = random.uniform(self.erase_range[0], self.erase_range[1]) * area
            aspect_ratio = random.uniform(self.aspect, 1 / self.aspect)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                x1 = random.randint(0, height - h)
                y1 = random.randint(0, width - w)
                if channels == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                else:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                return img

        return img


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
        self.mean = mean

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        height, width, channels = img.shape
        pad_width = self.target_size[0] - width if self.target_size[0] > width else width - self.target_size[0]
        pad_height = self.target_size[1] - height if self.target_size[1] > height else height - self.target_size[1]
        left_pad = random.randint(0, pad_width)
        up_pad = random.randint(0, pad_height)
        offset_left = -left_pad if self.target_size[0] > width else left_pad
        offset_up = -up_pad if self.target_size[1] > height else up_pad

        expand_image = np.zeros((max(height, self.target_size[1]) + abs(offset_up),
                                 max(width, self.target_size[0]) + abs(offset_left), channels), dtype=img.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
                     abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = img
        img = expand_image[max(offset_up, 0):max(offset_up, 0) + self.target_size[1],
                           max(offset_left, 0):max(offset_left, 0) + self.target_size[0]]
        return img


class RandomFlip(object):
    def __init__(self, flip90=False, ratio=0.5):
        self.flip90 = flip90
        self.ratio = ratio

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        height, width, _ = img.shape
        if not self.flip90:
            img = cv2.flip(img, 1)
        else:
            method = random.randint(0, 2)
            if method == 0:
                img = cv2.flip(img, 1)

            if method == 1:
                img = cv2.rotate(img, 0)

            if method == 2:
                img = cv2.rotate(img, 2)

        return img


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = ratio
        assert self.upper >= self.lower, "saturation upper must be >= lower."
        assert self.lower >= 0, "saturation lower must be non-negative."

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] *= random.uniform(self.lower, self.upper)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class RandomHue(object):
    def __init__(self, delta=18, ratio=0.5):
        assert 0 <= delta <= 360
        self.delta = delta
        self.ratio = ratio

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 0] += random.uniform(-self.delta, self.delta)
        img[:, :, 0][img[:, :, 0] > 360] -= 360
        img[:, :, 0][img[:, :, 0] < 0] += 360
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class RandomPerm(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        swap = self.perms[random.randint(0, len(self.perms) - 1)]
        img = img[:, :, swap].astype(np.uint8)
        return img


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = ratio
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        img = img.astype(np.float32)
        img *= random.uniform(self.lower, self.upper)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class RandomBrightness(object):
    def __init__(self, shift_value=30, ratio=0.5):
        self.shift_value = shift_value
        self.ratio = ratio

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class RandomBlur(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.blur_list = [
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AverageBlur(k=(2, 11)),
            iaa.AverageBlur(k=((5, 11), (1, 3))),
            iaa.MedianBlur(k=(3, 11)),
        ]

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        method = random.randint(0, len(self.blur_list)-1)
        img = self.blur_list[method].augment_image(img)
        return img


class RandomNoise(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.noise_list = [
            iaa.AddElementwise((-40, 40)),
            iaa.AddElementwise((-40, 40), per_channel=0.5),
            iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
            iaa.Dropout(p=(0, 0.2)),
            iaa.Dropout(p=(0, 0.2), per_channel=0.5),
            iaa.CoarseDropout(0.02, size_percent=0.5),
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5)
        ]

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        method = random.randint(0, len(self.noise_list)-1)
        img = self.noise_list[method].augment_image(img)
        return img


class RandomAffine(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.affine_list = [
            iaa.Affine(scale=(0.7, 1.3)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.01, 0.05)),
            iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
        ]

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        method = random.randint(0, len(self.affine_list)-1)
        img = self.affine_list[method].augment_image(img)
        return img


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
        self.size = tuple(size)
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
        height, width, _ = img.shape
        for attempt in range(10):
            area = width * height
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= width and h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback
        w = min(height, width)
        i = (height - w) // 2
        j = (width - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (Numpy Image): Image to be cropped and resized.

        Returns:
            Numpy Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = img[i:i+h, j:j+w]
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        return img


class RandomResize(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, scale_range=(0.75, 1.25), aspect_range=(0.9, 1.1), target_size=None,
                 resize_bound=None, method='random', ratio=0.5):
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
        assert isinstance(img, np.ndarray)

        height, width, _ = img.shape
        if random.random() < self.ratio:
            scale_ratio = self.get_scale([width, height])
            aspect_ratio = random.uniform(*self.aspect_range)
            w_scale_ratio = math.sqrt(aspect_ratio) * scale_ratio
            h_scale_ratio = math.sqrt(1.0 / aspect_ratio) * scale_ratio
        else:
            w_scale_ratio, h_scale_ratio = 1.0, 1.0

        converted_size = (int(width * w_scale_ratio), int(height * h_scale_ratio))
        img = cv2.resize(img, converted_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
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
        self.mean = mean

    def __call__(self, img):
        """
        Args:
            img    (Image):     Image to be rotated.

        Returns:
            Image:     Rotated image.
        """
        assert isinstance(img, np.ndarray)
        if random.random() < self.ratio:
            rotate_degree = random.uniform(-self.max_degree, self.max_degree)
        else:
            return img

        height, width, _ = img.shape

        img_center = (width / 2.0, height / 2.0)

        rotate_mat = cv2.getRotationMatrix2D(img_center, rotate_degree, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - img_center[0]
        rotate_mat[1, 2] += (new_height / 2.) - img_center[1]
        img = cv2.warpAffine(img, rotate_mat, (new_width, new_height), borderValue=self.mean).astype(np.uint8)
        return img


class RandomCrop(object):
    """Crop the given numpy.ndarray and  at a random location.

    Args:
        size (int or tuple): Desired output size of the crop.(w, h)
    """

    def __init__(self, crop_size, ratio=0.5, method='random', grid=None):
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
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        height, width, _ = img.shape
        target_size = [min(self.size[0], width), min(self.size[1], height)]

        offset_left, offset_up = self.get_lefttop(target_size, [width, height])

        # img = ImageHelper.draw_box(img, bboxes[index])
        img = img[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]]
        return img


class Resize(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.
    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, target_size=None, min_side_length=None, max_side_length=None):
        self.target_size = target_size
        self.min_side_length = min_side_length
        self.max_side_length = max_side_length

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        height, width, _ = img.shape
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

        target_size = tuple(target_size)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        return img

class GridMask(object):
    """ drops out rectangular regions of an image and the corresponding mask in a grid fashion.
    Args:
        d1: the min length of one unit.
        d2: the max length of one unit.
        r: the ratio of the shorter gray edge in a unit.
    """

    def __init__(self, d1, d2, r=0.5, ratio=0.5):
        self.d1 = d1
        self.d2 = d2
        self.r = r
        self.ratio = ratio

    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        h, w, _ = img.shape

        hh = math.ceil((math.sqrt(h*h + w*w)))
        d = np.random.randint(self.d1, self.d2)
        l = math.ceil(d*self.r)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh//d+1):
            s = d*i + st_h
            t = s + l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t,:] *= 0
        for i in range(-1, hh//d+1):
            s = d*i + st_w
            t = s + l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:,s:t] *= 0
        mask = 1 - mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]
        mask = np.expand_dims(mask, 2).repeat(3, axis=2)
        return mask * img


class JigSaw(object):
    """ get the jigsaw image of the input image
    Args:
        n: the spilt part of w and h
    """

    def __init__(self,n=7,ratio=0.5):
        self.n = n
        self.ratio = ratio

    def __call__(self,img):
        assert isinstance(img, np.ndarray)
        if random.random() > self.ratio:
            return img
        l = []
        for a in range(self.n):
            for b in range(self.n):
                l.append([a, b])
        h,w,_ = img.shape
        block_size = [h // self.n,w//self.n]
        rounds = self.n ** 2
        random.shuffle(l)
        jigsaws = img.copy()
        for i in range(rounds):
            x, y = l[i]
            temp = jigsaws[0:block_size[0], 0:block_size[1],:].copy()
            jigsaws[0:block_size[0], 0:block_size[1],:] = jigsaws[x * block_size[0]:(x + 1) * block_size[0],
                                                    y * block_size[1]:(y + 1) * block_size[1],:].copy()
            
            jigsaws[x * block_size[0]:(x + 1) * block_size[0], y * block_size[1]:(y + 1) * block_size[1],:] = temp

        jig = jigsaws[0:block_size[0]*self.n,0:block_size[1]*self.n,:]
        jigsaws = cv2.resize(jig,(w,h))
        return jigsaws

class RandomCamCrop(object):
    """ 
    Args:
        d1: the min length of one unit.
        d2: the max length of one unit.
        r: the ratio of the shorter gray edge in a unit.
    """

    def __init__(self, crop_ratio_range, ratio=0.5):
        self.crop_ratio_range = crop_ratio_range
        self.ratio = ratio

    def __call__(self, img, label):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img

        h, w, _ = img.shape
        kp_h, kp_w = int(h*label[1]/100), int(w*label[2]/100)
        h_crop_ratio = random.uniform(self.crop_ratio_range[0], self.crop_ratio_range[1])
        w_crop_ratio = random.uniform(self.crop_ratio_range[0], self.crop_ratio_range[1])

        t, b = kp_h, h-kp_h
        l, r = kp_w, w-kp_w

        top = kp_h - int(t*h_crop_ratio)
        buttom = kp_h + int(b*h_crop_ratio)
        left = kp_w - int(l*w_crop_ratio)
        right = kp_w + int(r*w_crop_ratio)

        crop_img = img[top:buttom, left:right]

        return crop_img

CV2_AUGMENTATIONS_DICT = {
    'random_erase': RandomErase,
    'random_saturation': RandomSaturation,
    'random_hue': RandomHue,
    'random_perm': RandomPerm,
    'random_contrast': RandomContrast,
    'random_brightness': RandomBrightness,
    'random_pad': RandomPad,
    'random_flip': RandomFlip,
    'random_resize': RandomResize,
    'random_crop': RandomCrop,
    'random_resized_crop': RandomResizedCrop,
    'random_rotate': RandomRotate,
    'random_blur': RandomBlur,
    'random_noise': RandomNoise,
    'random_affine': RandomAffine,
    'resize': Resize,
    'gridmask': GridMask,
    'jigsaw':JigSaw,
    'random_cam_crop': RandomCamCrop
}


class CV2AugCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> CV2AugCompose([
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
                self.transforms[trans] = CV2_AUGMENTATIONS_DICT[trans](**aug_trans[trans])

    def __call__(self, img, label=None):

        if self.configer.get('data', 'input_mode') == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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
            if trans_key == 'random_cam_crop':
                img = self.transforms[trans_key](img, label)
            else:
                img = self.transforms[trans_key](img)

        if self.configer.get('data', 'input_mode') == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

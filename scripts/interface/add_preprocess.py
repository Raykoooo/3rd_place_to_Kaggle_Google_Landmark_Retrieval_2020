import numpy as np
import os
import cv2
import glob
import sys
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.compat.v2 as tf_v2


pb_path = sys.argv[1]+'.pb'
saved_model_path = sys.argv[1]
size = sys.argv[2]

sess = tf.Session()

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph
graph_model = load_graph(pb_path)
graph_model_def = graph_model.as_graph_def()

# sess = tf.Session()
# graph_model = tf.saved_model.load(sess, export_dir=saved_model_path, tags=['serve'])
# graph_model_def = graph_model.as_graph_def()

# here is the important step, create a new graph and DON'T create a new session explicity
graph_model_new = tf.Graph()
graph_model_new.as_default()

#mean = np.asarray([[[0.485, 0.456, 0.406]]], dtype=np.float32).transpose((2,0,1))
#mean = np.asarray([[[104, 117, 123]]], dtype=np.float32).transpose((2,0,1))
#std = np.asarray([[[0.229, 0.224, 0.225]]], dtype=np.float32).transpose((2,0,1))

input_image = tf.placeholder(tf.uint8, shape=(None, None, 3), name='input_image')
# imgs_map.set_shape((None, None, None, 3))
#imgs = tf.image.resize(input_image, [299, 299], method=tf.image.ResizeMethod.BILINEAR)
if int(size) == 299:
    imgs = tf_v2.image.resize(input_image, [299, 299], method='bilinear',antialias=False)
elif int(size) == 448:
    imgs = tf_v2.image.resize(input_image, [448, 448], method='bilinear',antialias=True)
elif int(size) == 512:
    imgs = tf_v2.image.resize(input_image, [512, 512], method='bilinear',antialias=True)
elif int(size) == 640:
    imgs = tf_v2.image.resize(input_image, [640, 640], method='bilinear',antialias=True)
else:
    raise Exception("input size???")
# imgs = tf.reshape(imgs, (299, 299, 3))
#imgs = tf.convert_to_tensor(imgs)
#imgs = tf.image.resize(input_image, [299, 299], method=tf.image.ResizeMethod.BILINEAR)
#imgs = tf.transpose(imgs, [2, 0, 1])
#imgs = tf.reverse(imgs, axis=[-1]) #convert rgb to bgr
#imgs = input_image[..., ::-1]
imgs = tf.transpose(imgs, [2, 0, 1])
#imgs = imgs / 255
#imgs = imgs - mean
#imgs = imgs / std
# imgs = tf.reshape(imgs, (-1, 3, 299, 299))
imgs = tf.expand_dims(imgs, 0)
# imgs = tf.transpose(imgs, [2, 0, 1])
#img_float32 = tf.image.convert_image_dtype(imgs, dtype=tf.float32, saturate=False)
#img_float32 = tf.cast(imgs, dtype=tf.float32)
img_float32 = imgs

# import the model graph with the new input
tf.import_graph_def(graph_model_def, name='', input_map={"input:0": img_float32})

builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
inp = sess.graph.get_tensor_by_name('input_image:0')
out = sess.graph.get_tensor_by_name('global_descriptor:0')
signature = tf.saved_model.signature_def_utils.predict_signature_def(
    inputs={'input_image': input_image}, outputs={'global_descriptor': out})

builder.add_meta_graph_and_variables(
    sess=sess,
    tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            signature
    })
builder.save()

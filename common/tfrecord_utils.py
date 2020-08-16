import tensorflow as tf
import io

def read_tfrecord(example):
    tfrec_format = {
        'image/height': tf.io.FixedLenFeature([ ], tf.int64),
        'image/width': tf.io.FixedLenFeature([ ], tf.int64),
        'image/source_id': tf.io.FixedLenFeature([ ], tf.string),
        'image/encoded': tf.io.FixedLenFeature([ ], tf.string),
        'image/encoded_mask': tf.io.FixedLenFeature([ ], tf.string),
        'image/format': tf.io.FixedLenFeature([ ], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)}
    example = tf.io.parse_single_example(example, tfrec_format)

    mask = example[ 'image/encoded_mask' ]
    mask = tf.image.decode_png(mask, channels=1)
    
    width = tf.cast(example[ 'image/width' ],tf.float32)
    height = tf.cast(example[ 'image/height' ],tf.float32)

    xmin = example[ 'image/object/bbox/xmin' ]
    ymin = example[ 'image/object/bbox/ymin' ]
    xmax = example[ 'image/object/bbox/xmax' ]
    ymax = example[ 'image/object/bbox/ymax' ]
    label = example[ 'image/object/class/label' ]

    xmin = tf.cast(xmin, tf.float32)
    xmin = tf.sparse.to_dense(xmin)
    xmin = tf.cast(xmin*width, tf.int32)

    ymin = tf.cast(ymin, tf.float32)
    ymin = tf.sparse.to_dense(ymin)
    ymin = tf.cast(ymin*height, tf.int32)

    xmax = tf.cast(xmax, tf.float32)
    xmax = tf.sparse.to_dense(xmax)
    xmax = tf.cast(xmax*width, tf.int32)

    ymax = tf.cast(ymax, tf.float32)
    ymax = tf.sparse.to_dense(ymax)
    ymax = tf.cast(ymax*height, tf.int32)

    label = tf.cast(label, tf.int32)
    label = tf.sparse.to_dense(label)-1

    image = example[ 'image/encoded' ]
    #image = tf.image.decode_png(image, channels=3)
    #encoded_jpg_io = io.Bytes(encoded)

    bbox = tf.stack([xmin, ymin, xmax, ymax, label], axis=-1)

    return image, bbox


def get_tfrecord_dataset(files, batch_size=8, train=False):
    autotune = tf.data.experimental.AUTOTUNE

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=autotune)
    ds = ds.cache()

    ds = ds.map(read_tfrecord, num_parallel_calls=autotune)
    #ds = ds.map(process_data, num_parallel_calls=autotune)

    if train:
        REPLICAS = None
        ds = ds.batch(batch_size, drop_remainder=True).repeat(REPLICAS).shuffle(8 * batch_size,
                                                                                reshuffle_each_iteration=True)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    else:
        REPLICAS = 1
        ds = ds.batch(batch_size, drop_remainder=True)

    #ds = ds.repeat(REPLICAS)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.prefetch(autotune)
    return ds
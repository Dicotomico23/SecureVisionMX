# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
import scipy.misc 
from PIL import Image

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images using TensorFlow 2.x's tf.summary.image."""
        with self.writer.as_default():
            for i, img in enumerate(images):
                img = np.clip(img, 0, 1)
                # Convert [0,1] to uint8
                img_uint8 = (img * 255).astype(np.uint8)
                # If image is [C, H, W] with C in [1,3], transpose to [H, W, C]
                if img_uint8.ndim == 3 and img_uint8.shape[0] in [1, 3]:
                    img_uint8 = np.transpose(img_uint8, (1, 2, 0))
                # Add a batch dimension and log the image
                img_uint8 = np.expand_dims(img_uint8, axis=0)  # shape [1, H, W, C]
                tf.summary.image(f"{tag}/{i}", img_uint8, step=step)
            self.writer.flush()
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from image_batch import ImageBatcher, ImageBatcherType

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


# Good example for TensorFlow:
#  https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#static-dynamic-mode
# COuld take some tips from there

# Another repo about 'How to map yolov7 to TensorRT':
#  https://github.com/triple-Mu/YOLO-TensorRT8
# From this repo were taken dynamic batch-size code and thanks to it were added code for dynamic img
# Additional links about dynamic shapes for TensorRT:
#  https://forums.developer.nvidia.com/t/tensorrt-use-dynamic-batch-or-specified-batch/232835

class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = cuda.mem_alloc(size)
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """
    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        # self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30))
        # self.config.max_workspace_size = workspace * (2 ** 30)  # Deprecation

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path, end2end, conf_thres, iou_thres, max_det, batch_size, possible_inputs, **kwargs):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        v8 = kwargs['v8']
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        print("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
            
        if batch_size is not None or possible_inputs is not None:
            wandh0 = wandh1 = wandh2 = self.network.get_input(0).shape[-2:]
            
            if possible_inputs is not None:
                wandh0 = possible_inputs[0]
                wandh1 = possible_inputs[1]
                wandh2 = possible_inputs[2]
            
            name = self.network.get_input(0).name
            profile = self.builder.create_optimization_profile()
            print(f'\ndynamic batch profile is\n\
                {(batch_size[0], 3, *wandh0)}\n\
                {(batch_size[1], 3, *wandh1)}\n\
                {(batch_size[2], 3, *wandh2)}'
            )
            profile.set_shape(name, (batch_size[0], 3, *wandh0),
                              (batch_size[1], 3, *wandh1),
                              (batch_size[2], 3, *wandh2))
            self.config.add_optimization_profile(profile)
        
        # self.builder.max_batch_size = self.batch_size  # This no effect for networks created with explicit batch dimension mode. Also DEPRECATED.

        if end2end:
            previous_output = self.network.get_output(0)
            self.network.unmark_output(previous_output)
            if not v8: 
                # output [1, 8400, 85]
                # slice boxes, obj_score, class_scores
                strides = trt.Dims([1,1,1])
                starts = trt.Dims([0,0,0])
                bs, num_boxes, temp = previous_output.shape
                shapes = trt.Dims([bs, num_boxes, 4])
                # [0, 0, 0] [1, 8400, 4] [1, 1, 1]
                boxes = self.network.add_slice(previous_output, starts, shapes, strides)
                num_classes = temp -5 
                starts[2] = 4
                shapes[2] = 1
                # [0, 0, 4] [1, 8400, 1] [1, 1, 1]
                obj_score = self.network.add_slice(previous_output, starts, shapes, strides)
                starts[2] = 5
                shapes[2] = num_classes
                # [0, 0, 5] [1, 8400, 80] [1, 1, 1]
                scores = self.network.add_slice(previous_output, starts, shapes, strides)
                # scores = obj_score * class_scores => [bs, num_boxes, nc]
                scores = self.network.add_elementwise(obj_score.get_output(0), scores.get_output(0), trt.ElementWiseOperation.PROD)
            else:
                strides = trt.Dims([1,1,1])
                starts = trt.Dims([0,0,0])
                previous_output = self.network.add_shuffle(previous_output)
                previous_output.second_transpose    = (0, 2, 1)
                print(previous_output.get_output(0).shape)
                bs, num_boxes, temp = previous_output.get_output(0).shape
                shapes = trt.Dims([bs, num_boxes, 4])
                # [0, 0, 0] [1, 8400, 4] [1, 1, 1]
                boxes = self.network.add_slice(previous_output.get_output(0), starts, shapes, strides)
                num_classes = temp -4 
                starts[2] = 4
                shapes[2] = num_classes
                # [0, 0, 4] [1, 8400, 80] [1, 1, 1]
                scores = self.network.add_slice(previous_output.get_output(0), starts, shapes, strides)
            '''
            "plugin_version": "1",
            "background_class": -1,  # no background class
            "max_output_boxes": detections_per_img,
            "score_threshold": score_thresh,
            "iou_threshold": nms_thresh,
            "score_activation": False,
            "box_coding": 1,
            '''
            registry = trt.get_plugin_registry()
            assert(registry)
            creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
            assert(creator)
            fc = []
            fc.append(trt.PluginField("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("max_output_boxes", np.array([max_det], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("score_threshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            fc.append(trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            fc.append(trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("score_activation", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32))
            
            fc = trt.PluginFieldCollection(fc) 
            nms_layer = creator.create_plugin("nms_layer", fc)

            layer = self.network.add_plugin_v2([boxes.get_output(0), scores.get_output(0)], nms_layer)
            layer.get_output(0).name = "num"
            layer.get_output(1).name = "boxes"
            layer.get_output(2).name = "scores"
            layer.get_output(3).name = "classes"
            for i in range(4):
                self.network.mark_output(layer.get_output(i))


    def create_engine(self, engine_path, precision, calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8, calib_preprocessor=ImageBatcherType.LETTERBOX_YOLO, possible_inputs=None):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        print("Building {} Engine in {}".format(precision, engine_path))
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        # TODO: Strict type is only needed If the per-layer precision overrides are used
        # If a better method is found to deal with that issue, this flag can be removed.
        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                print("INT8 is not supported natively on this platform/device")
            else:
                if self.builder.platform_has_fast_fp16:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    self.config.set_flag(trt.BuilderFlag.FP16)
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                    if -1 in list(inputs[0].shape[1:]):
                        raise NotImplementedError('TensorRT will be not successfully created with such input data for now.')
                        if possible_inputs is None:
                            raise Exception('Model have dynamic input, but possible inputs do not provided')
                        if calib_shape[1] == 3:
                            self.format = "NCHW"
                            calib_shape[2] = possible_inputs[1][0]
                            calib_shape[3] = possible_inputs[1][1]
                        elif calib_shape[3] == 3:
                            self.format = "NHWC"
                            calib_shape[1] = possible_inputs[1][0]
                            calib_shape[2] = possible_inputs[1][1]
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    self.config.int8_calibrator.set_image_batcher(
                        ImageBatcher(calib_input, calib_shape, calib_dtype, max_num_images=calib_num_images,
                                     exact_batches=True, preprocessor=calib_preprocessor))

        # with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
        with self.builder.build_serialized_network(self.network, self.config) as engine, open(engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(engine_path))
            f.write(engine)  # .serialize()

def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx, args.end2end, args.conf_thres, args.iou_thres, args.max_det, v8=args.v8, batch_size=args.batch_size, possible_inputs=args.possible_inputs)
    builder.create_engine(args.engine, args.precision, args.calib_input, args.calib_cache, args.calib_num_images,
                          args.calib_batch_size, args.calib_preprocessor, args.possible_inputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("-w", "--workspace", default=1, type=int, help="The max memory workspace size to allow in Gb, "
                                                                       "default: 1")
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument("--calib_cache", default="./calibration.cache",
                        help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    parser.add_argument("--calib_num_images", default=5000, type=int,
                        help="The maximum number of images to use for calibration, default: 5000")
    parser.add_argument("--calib_batch_size", default=8, type=int,
                        help="The batch size for the calibration process, default: 8")
    parser.add_argument('--calib_preprocessor',
                        default=ImageBatcherType.LETTERBOX_YOLO,
                        type=str,
                        help='Preprocess type for calibration (int8). Supported types are: '
                            f'{ImageBatcherType.ONLY_NORMALIZE}, {ImageBatcherType.FIXED_SHAPE_RESIZER}, '
                            f'{ImageBatcherType.KEEP_ASPECT_RATIO_RESIZER}, {ImageBatcherType.LETTERBOX_YOLO}.')
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="export the engine include nms plugin, default: False")
    parser.add_argument("--conf_thres", default=0.4, type=float,
                        help="The conf threshold for the nms, default: 0.4")
    parser.add_argument("--iou_thres", default=0.5, type=float,
                        help="The iou threshold for the nms, default: 0.5")
    parser.add_argument("--max_det", default=100, type=int,
                        help="The total num for results, default: 100")
    parser.add_argument("--v8", default=False, action="store_true",
                        help="use yolov8 model, default: False")
    parser.add_argument('--batch-size',
                        nargs='+',
                        type=int,
                        default=None, #
                        help='Sequence of batch sized for tensorrt engine. Example: 1 2 3, '
                             'in this sequence maximum/between/minimum will be taken as corrisponding for TensorRT (max/mid/min). '
                             'Also single value is possible, aka: 4, means that batch-size will be freezed for this value')
    parser.add_argument('--possible-inputs',
                        nargs='+',
                        type=int,
                        default=None, #
                        help='Sequence of image sizes for tensorrt engine. Example: 320 480 640, '
                             'in this sequence input will be square (!) and maximum/between/minimum will be taken as corrisponding for TensorRT (max/mid/min). '
                             'Also single value is possible, aka: 640, means that input size will be freezed for this value. '
                             'Example with rectangular input: 320 340 480 500 640 700, where each two values are corrisponding Height and Width of the input, '
                             'also notice that for this example order is metter (order: minimum, mid, maximum)')
    args = parser.parse_args()
    print(args)
    if args.batch_size is not None and isinstance(args.batch_size, list) and len(args.batch_size) > 0:
        args.batch_size = args.batch_size if len(
            args.batch_size) == 3 else args.batch_size[-1:] * 3
        args.batch_size.sort()
        
    if args.possible_inputs is not None and isinstance(args.possible_inputs, list) and len(args.possible_inputs) > 0:
        if len(args.possible_inputs) == 1:
            args.possible_inputs = [[args.possible_inputs[-1:], args.possible_inputs[-1:]]] * 3
        elif len(args.possible_inputs) == 2:
            args.possible_inputs = [args.possible_inputs[-2:]] * 3
        elif len(args.possible_inputs) == 3:
            args.possible_inputs = [ 
                [single_possible_inputs, single_possible_inputs] 
                for single_possible_inputs in args.possible_inputs
            ]
        elif len(args.possible_inputs) == 6:
            args.possible_inputs = list(map(list, 
                zip(args.possible_inputs[::2], args.possible_inputs[1::2])
            ))
        else:
            raise Exception(f"Uknown size of possible inputs ({len(args.possible_inputs)}) = {args.possible_inputs}")
        # TODO: Check order min/opt/max
        
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision == "int8" and not (args.calib_input or os.path.exists(args.calib_cache)):
        parser.print_help()
        log.error("When building in int8 precision, --calib_input or an existing --calib_cache file is required")
        sys.exit(1)
    
    main(args)


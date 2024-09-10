from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="/workspace/RDD_yolo/v8l_no_pre/0802_3classes_640/weights/best.onnx", data="/workspace/RDD_yolo/data/rdd_3.yaml", imgsz=640, half=False, device=0)
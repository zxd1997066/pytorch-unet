import time
import os
import numpy as np
import simulation
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pytorch_unet


parser = argparse.ArgumentParser(description='PyTorch UNet evaluation')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--n_classes', default=6, type=int,
                    metavar='N', help='the number of class')
parser.add_argument('-i', '--iterations', default=100, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('-w', '--warmup-iterations', default=10, type=int, metavar='N',
                    help='number of warmup iterations to run')
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use ipex')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable Intel_PyTorch_Extension JIT path')
parser.add_argument('--precision', type=str, default="float32",
                    help='precision, float32, int8, bfloat16')
parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
parser.add_argument('--profile', action='store_true', help='Trigger profile on current topology.')
parser.add_argument('--arch', type=str, help='model name', default="U-Net")
parser.add_argument('--config_file', type=str, default="./conf.yaml", help='config file for int8 tuning')
parser.add_argument("--quantized_engine", type=str, default=None, help="torch backend quantized engine.")
parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")
parser.add_argument("--device", type=str, default='cpu',
                    help="cpu or cuda")

args = parser.parse_args()

# set quantized engine
if args.quantized_engine is not None:
    torch.backends.quantized.engine = args.quantized_engine
else:
    args.quantized_engine = torch.backends.quantized.engine
print("backends quantized engine is {}".format(torch.backends.quantized.engine))

def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)        
        self.transform = transform
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)
        
        return [image, mask]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate(args, device, model, dataloader):
    batch_time = AverageMeter()
    batch_time_list = []
    if args.compile:
        model = torch.compile(model, backend=args.backend, options={"freezing": True})
    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.iterations)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.iterations):
                inputs, labels = next(iter(dataloader))
                if args.channels_last:
                    inputs_oob, labels_oob = inputs, labels
                    inputs_oob = inputs_oob.to(memory_format=torch.channels_last)
                    labels_oob = labels_oob.to(memory_format=torch.channels_last)
                    inputs, labels = inputs_oob, labels_oob
                else:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                start = time.time()
                pred = model(inputs)
                p.step()
                end = time.time()
                print("Iteration: {}, inference time: {} sec.".format(i, end - start), flush=True)
                if i >= args.warmup_iterations:
                    batch_time_list.append((end - start) * 1000)
                    batch_time.update(end - start)
    else:
        for i in range(args.iterations):
            inputs, labels = next(iter(dataloader))
            if args.channels_last:
                inputs_oob, labels_oob = inputs, labels
                inputs_oob = inputs_oob.to(memory_format=torch.channels_last)
                labels_oob = labels_oob.to(memory_format=torch.channels_last)
                inputs, labels = inputs_oob, labels_oob
            else:
                inputs = inputs.to(device)
                labels = labels.to(device)

            start = time.time()
            pred = model(inputs)
            end = time.time()
            print("Iteration: {}, inference time: {} sec.".format(i, end - start), flush=True)
            if i >= args.warmup_iterations:
                batch_time_list.append((end - start) * 1000)
                batch_time.update(end - start)

    latency = batch_time.avg / args.batch_size * 1000
    throughput = args.batch_size/batch_time.avg
    print("\n", "-"*20, "Summary", "-"*20)
    print("inference latency:\t {:.3f} ms".format(latency))
    print("inference Throughput:\t {:.2f} samples/s".format(throughput))
    # P50
    batch_time_list.sort()
    p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
    p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
    p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
    print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
            % (p50_latency, p90_latency, p99_latency))

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
                '-unet-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

def main():
    print(args)

    # use same transform for train/val for this example
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_set = SimDataset(200, transform = trans)
    dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("----device:", device)

    model = pytorch_unet.UNet(args.n_classes).eval()
    if args.channels_last:
        try:
            model_oob = model
            model_oob = model_oob.to(memory_format=torch.channels_last)
            print("[INFO] Use NHWC model")
            model = model_oob
        except:
            print("[WARN] Failed to use NHWC model")
    else:
        model.to(device)

    if args.ipex:
        import intel_extension_for_pytorch as ipex
        print("import IPEX **************")
        if args.precision == "bfloat16":
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
            print("Running with bfloat16...")
        else:
            model = ipex.optimize(model, dtype=torch.float32, inplace=True)
            print("Running with float32...")

    if args.precision == 'inc_int8':
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization(args.config_file)
        quantizer.calib_dataloader = dataloader
        quantizer.model = common.Model(model)
        q_model = quantizer()
        model = q_model.model

    if args.precision == "fx_int8":
        print('Converting int8 model...')
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        qconfig_mapping = get_default_qconfig_mapping(args.quantized_engine)
        with torch.no_grad():
            for i in range(args.warmup_iterations):
                inputs, labels = next(iter(dataloader))
                if args.channels_last:
                    inputs_oob, labels_oob = inputs, labels
                    inputs_oob = inputs_oob.to(memory_format=torch.channels_last)
                    labels_oob = labels_oob.to(memory_format=torch.channels_last)
                    inputs, labels = inputs_oob, labels_oob
                else:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                prepared_model = prepare_fx(model, qconfig_mapping, inputs)
                prepared_model(inputs)
        model = convert_fx(prepared_model)
        print('Convert int8 model done...')

    if args.jit:
        model = torch.jit.script(model)
        print("---- Use script model.")
        if args.ipex:
            model = torch.jit.freeze(model)

    # evaluate
    with torch.no_grad():
        if args.precision == 'bfloat16':
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                evaluate(args, device, model, dataloader)
        elif args.precision == 'float16':
            if device == "cpu":
              print('---- Enable CPU AMP float16')
              with torch.cpu.amp.autocast(enabled=True, dtype=torch.half):
                  evaluate(args, device, model, dataloader)
            elif device == "cuda":
              print('---- Enable CUDA AMP float16')
              with torch.cuda.amp.autocast(enabled=True, dtype=torch.half):
                  evaluate(args, device, model, dataloader)
        else:
            evaluate(args, device, model, dataloader)


if __name__ == '__main__':
    main()

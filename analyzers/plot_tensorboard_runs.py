import sys
sys.path.append("../GOProFormer")

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_tensorboard_runs(paths, model_names, x_label, y_label, tag, img_name, img_format="png"):
    plt.cla()
    for i, path in enumerate(paths):
        ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
        _absorb_print = ea.Reload()

        x, y=[],[]
        for j, event in enumerate(ea.Scalars(tag)):
            # print(event.step, event.value)
            x.append(event.step)
            y.append(event.value) 
            if j==40: break
        if "Loss" in tag: plt.plot(x, y, label=model_names[i])
        elif "max" in tag: plt.plot(x, y, label=model_names[i])
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.show()
    plt.savefig(f"outputs/images/loss_and_perf_vs_epochs/{img_name}.{img_format}", dpi=300, format=img_format, bbox_inches='tight', pad_inches=0.0)

dir="outputs/tensorboard_runs/"

# GOname = "BP"
# paths = [dir +"Modelv3.2_yeast_BP_0.0001_32_500_271_512_256_5_1024_3_2_0.5_False_False_cuda/events.out.tfevents.1660410557.NODE076.orc.gmu.edu.120293.0",
#          dir +"Modelv3.3_yeast_BP_0.0001_32_500_287_512_256_5_1024_3_2_0.5_False_False_cuda/events.out.tfevents.1660502695.NODE050.orc.gmu.edu.19550.0",
#          dir +"Modelv3.4_yeast_BP_0.0001_32_500_287_512_256_5_1024_3_2_0.5_False_False_cuda/events.out.tfevents.1660503149.node056.orc.gmu.edu.24054.0"]

# GOname = "CC"
# paths = [dir +"Modelv3.2_yeast_CC_0.0001_32_500_244_512_256_5_1024_3_2_0.5_False_False_cuda/events.out.tfevents.1660410591.NODE076.orc.gmu.edu.120414.0",
#          dir +"Modelv3.3_yeast_CC_0.0001_32_500_246_512_256_5_1024_3_2_0.5_False_False_cuda/events.out.tfevents.1660502755.NODE050.orc.gmu.edu.19906.0",
#          dir +"Modelv3.4_yeast_CC_0.0001_32_500_246_512_256_5_1024_3_2_0.5_False_False_cuda/events.out.tfevents.1660503179.node056.orc.gmu.edu.24131.0"]

GOname = "MF"
paths = [dir + "Modelv3.2_yeast_MF_0.0001_32_500_370_512_256_5_1024_3_2_0.5_False_False_cuda/events.out.tfevents.1660428555.NODE050.orc.gmu.edu.17721.0",
         dir + "Modelv3.3_yeast_MF_0.0001_32_500_433_512_256_5_1024_3_2_0.5_False_False_cuda/events.out.tfevents.1660502546.NODE050.orc.gmu.edu.19115.0",
         dir + "Modelv3.4_yeast_MF_0.0001_32_500_432_512_256_5_1024_3_2_0.5_False_False_cuda/events.out.tfevents.1660503123.node056.orc.gmu.edu.23908.0"]

models = ["GOProFormer-TDNK", "GOProFormer-RS", "GOProFormer-TSNK"]
img_names = [f"{GOname}_comparison_train_loss", f"{GOname}_comparison_val_loss", f"{GOname}_comparison_val_fmax"]


# plotting using above set of configurations
tags = ["TrainLoss", "ValLoss", "ValFmax"]
y_labels = ["Binary cross-entropy", "Binary cross-entropy", "Fmax"]
x_label = "Number of epochs"

for i, tag in enumerate(tags):
    plot_tensorboard_runs(paths, models, x_label, y_label=y_labels[i], tag=tags[i], img_name=img_names[i], img_format="png")


# ProToFormer (256-SEQ+IDM): dir+"Model_distmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda/events.out.tfevents.1652807314.node056.orc.gmu.edu.27957.0"
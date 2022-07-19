---
layout: editorial
---

# torch model 2 onnx

当torch模型用来进行推理的时候，一般需要两个步骤：1）重新生成模型对象；2）导入训练好的权重。在同时训练了多个参数不同的模型时，往往需要管理多个模型的定义，当这些模型相互依赖时，代码将变得难以维护。

将torch模型转换为onnx模型，由onnx runtime进行推理，可以省去对模型文件的管理。

### 转换

使用`torch.onnx.export`可以容易地将模型转换为onnx格式，其官方说明文档如下

```
torch.onnx.export(
    model, 
    args, 
    f, 
    export_params=True, 
    verbose=False, 
    training=<TrainingMode.EVAL: 0>, 
    input_names=None, 
    output_names=None, 
    operator_export_type=<OperatorExportTypes.ONNX: 0>, 
    opset_version=None, 
    do_constant_folding=True, 
    dynamic_axes=None, 
    keep_initializers_as_inputs=None, 
    custom_opsets=None, 
    export_modules_as_functions=False
)
```

其中model即是我们要导出的模型实例，args则是model的forward方法所需要的必要参数，多个参数以元组的形式传递，f是要导出的文件。

**由于train和eval模式中，model的行为有所不同，在导出模型之前，需要先将model设置为需要的模式**。这里需要说明的是，onnx实际上记录的是数据的变换过程，而不关心模型的实现以及torch的协议，在导出过程中，该函数需要以输入的args实际运行一遍模型，并在运行过程中追踪模型的行为，从而决定构建onnx可理解的数据流变换。显然为了追踪数据变换，这里的args只要形状满足要求即可，并不一定需要用真实的数据来作为输入。

### 加载

假设我们已经得到了某个名为`diffusion.onnx`的模型，为了加载该模型，需要安装onnx运行时，用pip指令可以直接进行安装。示例加载代码如下：

```
import onnxruntime
import numpy as np

sess=onnxruntime.InferenceSession('diffusion.onnx')
print('inputs:',[(x.name,x.shape) for x in sess.get_inputs()])
print('outputs:',[(x.name,x.shape) for x in sess.get_outputs()])

frames=1024
spec=np.random.rand(1,1,128,frames).astype(np.float32)
diffusion_steps=[51,]
cond=np.random.rand(1,384,frames).astype(np.float32)

outputs=sess.run(None,{'spec':spec,'diffusion_step':diffusion_steps,'cond':cond})
print(outputs[0].shape)
```

运行上述代码的输出如下

```
inputs: [('spec', [1, 1, 128, 'frames']), ('diffusion_step', [1]), ('cond', [1, 384, 'frames'])]
outputs: [('output', [1, 1, 128, 'frames'])]
(1, 1, 128, 1024)
```

加载完模型后，我们打印了需要准备的输入的名称以及大小，然后随机生成了相应的数据，最终通过runtime进行推理，在这中间不需要关心任何模型的结构问题。

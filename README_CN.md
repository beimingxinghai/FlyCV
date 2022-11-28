<img width=100% src="docs/assets/flycv.png"/>

**[English](README_EN.md)** | **简体中文**

## 🔥 FlyCV —— 高性能计算机图像系统

![license](https://img.shields.io/hexpm/l/plug?color=%2300CD66&logo=apache&style=flat)
![GitHub star](https://img.shields.io/github/stars/paddlepaddle/flycv?color=%23FFB90F&logo=github)
![GitHub forks](https://img.shields.io/github/forks/paddlepaddle/flycv?color=pink&logo=github)
![GitHub issues](https://img.shields.io/github/issues/paddlepaddle/flycv?color=%2377BFFF&logo=github)
![main language](https://img.shields.io/github/languages/top/paddlepaddle/flycv?color=%23EEE685&logo=c%2B%2B)
![GitHub contributors](https://img.shields.io/github/contributors/paddlepaddle/flycv?color=orange&logo=github)
![Pull Request](https://img.shields.io/github/issues-pr/paddlepaddle/flycv?color=FFDAB9&logo=github)
![Commit activity](https://img.shields.io/github/commit-activity/m/paddlepaddle/flycv?color=F0FFFF&logo=github)
![Platforms](https://img.shields.io/static/v1?label=platforms&message=android|armlinux|windows|linux|ios&color=BBFFFF&logo=gnometerminal)

**FlyCV** 是一个**高性能**的计算机图像处理库。我们目前在 **ARM** 架构下做了很多优化，相比其他图像处理库性能更为出色。同时也在开展其他CPU和异构平台的性能优化工作，例如x86，RISC-V，GPU，DSP等。除了速度更快之外，FlyCV提供了更加细粒度的编译选项控制，这使得我们在库体积上**非常轻量**，可以**按需编译** 。另外我们在编译阶段还提供了**自定义命名空间**的选项支持，可以方便快速地解决相同依赖库冲突的问题。

接下来通过几个例子简单体验一下FlyCV和OpenCV的性能和效果对比。

<img width="32%" src="https://user-images.githubusercontent.com/10016040/203230955-e99ab754-b430-4a38-ba5b-cbcddda70bbc.gif">&nbsp;&nbsp;&nbsp;<img width="32%" src="https://user-images.githubusercontent.com/10016040/203231718-baf03baa-89f8-4fd8-b779-6e04c7ea0234.gif">&nbsp;&nbsp;&nbsp;<img width="32%" src="https://user-images.githubusercontent.com/10016040/203208952-26560bb6-9ab2-4243-acf1-2c4f8f5482c4.gif">

<img width="32%" src="https://user-images.githubusercontent.com/10016040/203209148-b00ed499-2695-4463-8c70-2d0fa642491a.gif">&nbsp;&nbsp;&nbsp;<img width="32%" src="https://user-images.githubusercontent.com/10016040/203209020-c7fafd21-b8e2-498e-8082-a4fa3d0241e9.gif">&nbsp;&nbsp;&nbsp;<img width="32%" src="https://user-images.githubusercontent.com/10016040/203208799-91196ae5-841d-4d27-b0cf-d1c51595c1f4.gif">

可以看到执行效果基本一致，FlyCV速度明显快于OpenCV。

<br>

## :notebook: API 文档

为了便于大家集成使用，我们的接口设计和OpenCV非常接近，可以比较方便的进行迁移。但也存在一些使用上的差别，可以通过下面的接口文档了解更多的细节：[📖 API 文档](https://github.com/PaddlePaddle/FlyCV/wiki/API-Documents)

<br>

## :notebook_with_decorative_cover: 编译文档

我们支持大多数主流的操作系统，包括android、armlinux、macos（x86 & arm）、windows，以及ios。针对不同平台都提供了相关的编译脚本，可以很方便的在本机进行源码编译。相比OpenCV，为了保证**极致轻量化**，我们还提供粗、细粒度两种功能模块编译选项，能够真正支持按需编译，在体积要求非常严格的应用中可以带来大幅优化。

根据下面的文档可以从源码进行编译：[:book: 编译手册](https://github.com/PaddlePaddle/FlyCV/wiki/Compile-manual)

我们也提供了不同平台预编译完成的通用库，你也可以直接进行下载：[预编译库]()

**注：** 本地环境可能差别较大，如果无法运行可以尝试从源码编译。

<br>

## :pushpin: 升级日志

可以通过我们的升级日志查看每个版本的变化：[升级日志](https://github.com/PaddlePaddle/FlyCV/wiki/ChangeLog)

<br>

## :two_men_holding_hands: 技术交流

扫描下面的二维码加入我们的技术交流群，一起探讨计算机视觉领域的图像处理方法和优化工作。

<div>
<img width=20% src="https://user-images.githubusercontent.com/10016040/204269666-b3064337-c405-4f99-818f-a00de1b0ce54.png"/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img width=20% src="https://user-images.githubusercontent.com/10016040/204269943-d6fa9069-cf57-4cf3-9fdc-9c85baf08723.png"/>
</div>

<br>



## :pencil2: 贡献代码

非常欢迎大家一起来贡献代码，我们提供了详细的文档以便于大家可以进行快速集成，具体细节可以查阅 [CONTRIBUTING.md](./CONTRIBUTING.md) ，在这里感谢每一个人的努力付出，共建一个优秀的视觉图像处理系统。

<a href="https://github.com/paddlepaddle/flycv/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=paddlepaddle/flycv" />
</a>

<br>

## :copyright: License
[Apache-2.0](./LICENSE)
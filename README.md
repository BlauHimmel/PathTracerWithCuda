# PathTracerWithCuda

A gpu accelerated path tracer based on CUDA

The compute capability of your GPU should be greater than 6.x(For unified memory has been widely use in my program)

The GRAPHICS CARD'S DRIVER crashed occasionally. I am working on it. :( 

* Interactive camera(Depth of field supported!)
* Interactive UI supported for changing almost any stuff dynamic.
* Scene and config define using json sytax. Easy to use.
* Could change render scene dynamic.
* Diffuse texture supported.
* Obj mesh supported(Material per group supported).
* Naive subsurface scattering supported.
* Bvh accelerating structure.
* Skybox supported.
* Microsurface for reflection.

![](https://github.com/BlauHimmel/PathTracerWithCuda/blob/bvh-cpu/Result/sample29.png)

![](https://github.com/BlauHimmel/PathTracerWithCuda/blob/bvh-cpu/Result/sample30.png)

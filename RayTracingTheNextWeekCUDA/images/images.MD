- 未优化：

![avatar](/render_512x512_10bounces_32spp.png)
512 x 512
Bounces 10
SPP 32
11498.692 ms

![avatar](/render_512x512_15bounces_32spp.png)
512 x 512
Bounces 15
SPP 32
16557.516 ms

- 使用AABB优化射线-三角形相交

![avatar](/render_512x512_10bounces_32spp_AABB.png)
512 x 512
Bounces 10
SPP 32
5956.037 ms

![avatar](/render_512x512_15bounces_32spp_AABB.png)
512 x 512
Bounces 15
SPP 32
7713.623 ms

- 使用AABB优化射线-三角形相交
- 使用texture memory代替global memory

![avatar](/render_512x512_10bounces_32spp_AABB_texture_memory.png)
512 x 512
Bounces 15
SPP 32
3170.691 ms

![avatar](/render_512x512_15bounces_32spp_AABB_texture_memory.png)
512 x 512
Bounces 15
SPP 32
4066.207 ms

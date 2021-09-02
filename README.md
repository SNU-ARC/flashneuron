# FlashNeuron

FlashNeuron is the DNN training system using an NVMe SSD as a backing store. FlashNeuron introduces an offloading scheduler, which selectively offloads a set of intermediate data to the SSD in a compressed format without increasing DNN evaluation time. FlashNeuron causes minimal interference to CPU processes as the GPU and the SSD directly communicate for data transfers. FlashNeuron can increase the batch size over the maximum allowable batch size. By employing a larger batch size, FlashNeuron also improves the training throughput over the baseline using GPU memory only, while minimally disturbing applications running on CPU.

This repository contains the implementation of FlashNeuron in the PyTorch.
We currently update the version of PyTorch. Please stay tuned.

Please cite the following paper if you use FlashNeuron:

**FlashNeuron: SSD-Enabled Large-Batch Training of Very Deep Neural Networks.** Jonghyun Bae, Jongsung Lee, Yunho Jin, Sam Son, Shine Kim, Hakbeom Jang, Tae Jun Ham, and Jae W. Lee. _Proceedings of the 19th USENIX Conference on File and Storage Technologies (FAST 21)_.

~~~
@inproceedings {264816,  
  author = {Jonghyun Bae and Jongsung Lee and Yunho Jin and Sam Son and Shine Kim and Hakbeom Jang and Tae Jun Ham and Jae W. Lee},  
  title = {FlashNeuron: SSD-Enabled Large-Batch Training of Very Deep Neural Networks},  
  booktitle = {19th {USENIX} Conference on File and Storage Technologies ({FAST} 21)},  
  year = {2021},  
  isbn = {978-1-939133-20-5},  
  pages = {387--401},  
  url = {https://www.usenix.org/conference/fast21/presentation/bae},  
  publisher = {{USENIX} Association},  
  month = feb,  
}
~~~

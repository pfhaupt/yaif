use crate::cl_kernel;

use opencl3::memory::Buffer;

pub struct ClNet {
    layers: Vec<Buffer<f32>>,
}
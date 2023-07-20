use opencl3::memory::Buffer;

use crate::cl_kernel::ClStruct;


pub struct ClBuffer {
    buffer: Buffer<f32>,
    rows: usize,
    cols: usize,
    buffer_size: usize,
}

impl ClBuffer {
    pub fn new(cl_struct: &ClStruct, rows: usize, cols: usize) -> Self {
        let buffer = cl_struct.create_buffer(rows, cols);
        let buffer = match buffer {
            Some(b) => b,
            None => panic!("Attempted to get uninitialized buffer.\nPlease bind the buffer using .bind() to create it.")
        };
        Self { rows, cols, buffer_size: rows * cols, buffer }
    }

    pub fn get_buffer(&self) -> &Buffer<f32> {
        &self.buffer
    }

    #[inline]
    pub fn get_dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn get_buffer_size(&self) -> usize {
        self.buffer_size
    }
}

impl Default for ClBuffer {
    fn default() -> Self {
        let mut c = ClStruct::new().unwrap();
        c.load_kernels();
        ClBuffer { buffer: c.create_buffer(0, 0).unwrap(), rows: 0, cols: 0, buffer_size: 0 }
     }
}
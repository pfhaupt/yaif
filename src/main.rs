use yaif::matrix::Matrix;
use std::time::Instant;

use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_float, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
use std::ptr;

fn init_mat(size: usize) -> (Matrix, Matrix) {
    let mut m1 = Matrix::new(size, size);
    let mut m2 = Matrix::new(size, size);
    m1.gaussian_fill(0.0, 1.0);
    m2.gaussian_fill(0.0, 1.0);
    (m1, m2)
}

fn test_solo(tests: usize, size: usize) {
    for _ in 0..tests {
        let (m1, m2) = init_mat(size);
        let _r1 = m1.add(&m2).unwrap();
    }
}

const PROGRAM_SOURCE: &str = r#"
kernel void matrix_add(global float* z,
    global float const* x,
    global float const* y)
{
    const size_t local_dispatch_size = get_local_size(0); // same on y axis
    const size_t global_idx_x = get_global_id(0);
    const size_t global_idx_y = get_global_id(1);
    
    const array_idx = global_idx_x + global_idx_y * local_dispatch_size;
    z[array_idx] = x[array_idx] + y[array_idx];
}"#;

const KERNEL_NAME: &str = "matrix_add";

fn test_opencl(kernel: &Kernel, context: &Context, queue: &CommandQueue, tests: usize, size: usize) -> Result<u64> {
    let array_size = size * size;
    let mut x = Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, array_size, ptr::null_mut())?;
    let mut y = Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, array_size, ptr::null_mut())?;
    let z = Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, array_size, ptr::null_mut())?;
    let mut dur = 0;
    for _ in 0..tests {
        let (m1, m2) = init_mat(size);
        
        let _x_write_event = queue.enqueue_write_buffer(&mut x, CL_BLOCKING, 0, &m1.get_all(), &[])?;
        let y_write_event = queue.enqueue_write_buffer(&mut y, CL_NON_BLOCKING, 0, &m2.get_all(), &[])?;
        
        let kernel_event = 
            ExecuteKernel::new(&kernel)
                .set_arg(&z)
                .set_arg(&x)
                .set_arg(&y)
                .set_global_work_sizes(&[(array_size/64) + 1, 1, 1])
                .set_local_work_sizes(&[16, 1, 1])
                .set_wait_event(&y_write_event)
                .enqueue_nd_range(&queue)?;
        
        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        let mut r1 = vec![0.0; array_size];
        
        let read_event = queue.enqueue_read_buffer(&z, CL_NON_BLOCKING, 0, &mut r1, &events)?;
        read_event.wait()?;
        let mut r = Matrix::new(size, size);
        r.fill_vec(&r1);
        let start_time = kernel_event.profiling_command_start()?;
        let end_time = kernel_event.profiling_command_end()?;
        let duration = end_time - start_time;
        dur += duration;
    }
    Ok(dur)
}

fn time_init(tests: usize, size: usize) {
    for _ in 0..tests {
        let (_m1, _m2) = init_mat(size);
    }
}

fn main() -> Result<()> {
    let size = 1_000;

    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);
    let context = Context::from_device(&device).expect("Context::from_device failed");
    let queue = CommandQueue::create_with_properties(&context, device_id, CL_QUEUE_PROFILING_ENABLE, 0)
        .expect("CommandQueue::create_default failed");
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    const TESTS: usize = 100;
    let now = Instant::now();
    time_init(TESTS, size);
    let elapsed = now.elapsed();
    println!("Init Matrix takes {:?}", elapsed);
    println!("All benchmarks below account for the initialization time.");
    let dur = match test_opencl(&kernel, &context, &queue, TESTS, size) {
        Ok(d) => { d },
        Err(e) => panic!("{}", e),
    };
    println!("OpenCL matrix add ({}x{}): {:16?}ns", size, size, dur);
    let now = Instant::now();
    test_solo(TESTS, size);
    println!("Basic matrix add ({}x{}):  {:16?}ns", size, size, (now.elapsed() - elapsed).as_nanos());
    Ok(())
}

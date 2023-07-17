#![allow(dead_code, unused, non_snake_case)]

#[feature(mmul_v2)]

use yaif::matrix::Matrix;
use core::panic;
use std::process::exit;
use std::time::Instant;

use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_float, cl_uint, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
use std::ptr;

use rand::Rng;

fn init_single_mat(size: usize) -> Matrix {
    let mut m1 = Matrix::new(size, size);
    m1.fill(1.0);
    m1
}

fn init_mat(size: usize) -> (Matrix, Matrix) {
    (init_single_mat(size), init_single_mat(size))
}

fn time_basic(tests: usize, size: usize, which: &str) {
    for _ in 0..tests {
        match which {
            "add" => {
                let (m1, m2) = init_mat(size);
                let _r = m1.add(&m2).unwrap();
            },
            "smul" => {
                let (mut m1, _) = init_mat(size);
                let scalar = rand::thread_rng().gen_range(0.0..10.0);
                m1.multiply_scalar(scalar);
            },
            "mmul" => {
                let (m1, m2) = init_mat(size);
                let _r = m1.multiply(&m2).unwrap();
            },
            _ => {
                unimplemented!("{}", which);
            }
        }
    }
}

fn test_add(kernel: &Kernel, context: &Context, queue: &CommandQueue, tests: usize, size: usize) -> Result<u64> {
    let worst_case = (if size % 32 != 0 { (size / 32 + 1) * 32 } else { size }).pow(2);

    let mut a = Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, worst_case, ptr::null_mut())?;
    let mut b = Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, worst_case, ptr::null_mut())?;
    let c = Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, worst_case, ptr::null_mut())?;
    
    let mut dur = 0;

    for _ in 0..tests {
        let m = rand::thread_rng().gen_range(10..size);
        let n = rand::thread_rng().gen_range(10..size);

        let mut m1 = Matrix::new(m, n);
        m1.fill(2.0);
        let mut m2 = Matrix::new(m, n);
        m2.fill(3.0);

        let check_m1 = m1.clone();
        let check_m2 = m2.clone();

        let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
        let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };

        m1.pad(new_m - m, new_n - n);
        m2.pad(new_m - m, new_n - n);
        
        let _x_write_event = queue.enqueue_write_buffer(&mut a, CL_BLOCKING, 0, &m1.get_all(), &[])?;
        let _y_write_event = queue.enqueue_write_buffer(&mut b, CL_BLOCKING, 0, &m2.get_all(), &[])?;
        
        let kernel_event = yaif::kernel::get_madd_kernel_event(kernel, queue, new_m, new_n, &a, &b, &c)?;
        
        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        let mut r1 = vec![0.0; worst_case];
        
        let read_event = queue.enqueue_read_buffer(&c, CL_BLOCKING, 0, &mut r1, &events)?;
        read_event.wait()?;

        let mut r = Matrix::new(m, n);
        r.fill_fit(&r1, new_n);

        assert_eq!(r, check_m1.add(&check_m2).unwrap(), "OpenCL Matrix Matrix Addition is not working properly!");

        let start_time = kernel_event.profiling_command_start()?;
        let end_time = kernel_event.profiling_command_end()?;
        let duration = end_time - start_time;
        dur += duration;
    }
    Ok(dur)
}

fn test_smul(kernel: &Kernel, context: &Context, queue: &CommandQueue, tests: usize, size: usize) -> Result<u64> {
    let worst_case = (if size % 32 != 0 { (size / 32 + 1) * 32 } else { size }).pow(2);

    let mut x = Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, worst_case, ptr::null_mut())?;

    let mut dur = 0;

    for _ in 0..tests {
        let m = rand::thread_rng().gen_range(10..size);
        let n = rand::thread_rng().gen_range(10..size);

        let mut m1 = Matrix::new(m, n);
        m1.fill(2.0);
        
        let mut check_m1 = m1.clone();
        
        let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
        let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
        
        m1.pad(new_m - m, new_n - n);

        let scalar: cl_float = rand::thread_rng().gen_range(0..10) as f32;
        
        let _x_write_event = queue.enqueue_write_buffer(&mut x, CL_BLOCKING, 0, &m1.get_all(), &[])?;
        
        let kernel_event = 
            ExecuteKernel::new(&kernel)
                .set_arg(&x)
                .set_arg(&scalar)
                .set_global_work_sizes(&[worst_case, 1, 1])
                .set_local_work_sizes(&[64, 1, 1])
                .enqueue_nd_range(&queue)?;
        
        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        let mut r1 = vec![0.0; worst_case];
        
        let read_event = queue.enqueue_read_buffer(&x, CL_BLOCKING, 0, &mut r1, &events)?;
        read_event.wait()?;
        let mut r = Matrix::new(m, n);
        r.fill_fit(&r1, new_n);

        check_m1.multiply_scalar(scalar);
        assert_eq!(r, check_m1, "OpenCL Matrix Scalar Multiplication is not working properly!");

        let start_time = kernel_event.profiling_command_start()?;
        let end_time = kernel_event.profiling_command_end()?;
        let duration = end_time - start_time;
        dur += duration;
    }
    Ok(dur)
}

fn test_mmul(kernel: &Kernel, context: &Context, queue: &CommandQueue, tests: usize, size: usize) -> Result<u64> {
    let worst_case = (if size % 32 != 0 { (size / 32 + 1) * 32 } else { size }).pow(2);

    let mut A = Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, worst_case, ptr::null_mut())?;
    let mut B = Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, worst_case, ptr::null_mut())?;
    let C = Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, worst_case, ptr::null_mut())?;
    
    let mut dur = 0;

    for _ in 0..tests {
        // let m = rand::thread_rng().gen_range(10..size);
        // let n = rand::thread_rng().gen_range(10..size);
        // let k = rand::thread_rng().gen_range(10..size);
        let m = 17;
        let k = 13;
        let n = 9;
        let mut m1 = Matrix::new(m, k);
        let mut m2 = Matrix::new(k, n);
        println!("m1: {:?}", m1.get_dim());
        println!("m2: {:?}", m2.get_dim());
        println!("r:  ({}, {})", m, n);
        m1.fill(2.0);
        m2.fill(3.0);
        let check_m1 = m1.clone();
        let check_m2 = m2.clone();
        let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
        let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
        let new_k = if k % 32 != 0 { (k / 32 + 1) * 32 } else { k };
        m1.pad(new_m - m, new_k - k);
        m2.pad(new_k - k, new_n - n);

        let _a_write_event = queue.enqueue_write_buffer(&mut A, CL_BLOCKING, 0, &m1.get_all(), &[])?;
        let _x_write_event = queue.enqueue_write_buffer(&mut B, CL_BLOCKING, 0, &m2.get_all(), &[])?;
        
        let kernel_event = yaif::kernel::get_mmul_kernel_event(kernel, queue, new_m, new_n, new_k, &A, &B, &C)?;
        
        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        let mut r1 = vec![0.0; worst_case];
        
        let read_event = queue.enqueue_read_buffer(&C, CL_BLOCKING, 0, &mut r1, &events)?;
        read_event.wait()?;
        let mut r = Matrix::new(m, n);
        r.fill_fit(&r1, new_m);

        assert_eq!(r, check_m1.multiply(&check_m2).unwrap(), "OpenCL Matrix Matrix Multiplication is not working properly!");
        
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

fn time_cl(kernel: &Kernel, context: &Context, queue: &CommandQueue, tests: usize, size: usize, which: &str) -> u64 {
    let f = match which {
        "add"=> test_add,
        "smul" => test_smul,
        "mmul" => test_mmul,
        _ => unimplemented!("{}", which)
    };
    match f(&kernel, &context, &queue, tests, size) {
        Ok(d) => { d },
        Err(e) => panic!("{}", e),
    }
}

fn main() -> Result<()> {
    const TESTS: usize = 1;
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
    .first()
    .expect("no device found in platform");
    let device = Device::new(device_id);
    let context = Context::from_device(&device).expect("Context::from_device failed");
    let queue = CommandQueue::create_with_properties(&context, device_id, CL_QUEUE_PROFILING_ENABLE, 0)
        .expect("CommandQueue::create_default failed");

    let add_program = Program::create_and_build_from_source(&context, yaif::kernel::ADD_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let add_kernel = Kernel::create(&add_program, yaif::kernel::ADD_NAME).expect("Kernel::create failed");

    let mul_scalar_program = Program::create_and_build_from_source(&context, yaif::kernel::MUL_SCALAR_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let mul_scalar_kernel = Kernel::create(&mul_scalar_program, yaif::kernel::MUL_SCALAR_NAME).expect("Kernel::create failed");

    let mul_matrix_program = Program::create_and_build_from_source(&context, yaif::kernel::MUL_MATRIX_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let mul_matrix_kernel = Kernel::create(&mul_matrix_program, yaif::kernel::MUL_MATRIX_NAME).expect("Kernel::create failed");

    let mut actual_size = 16.0;

    while actual_size < (1 << 15) as f64 {
        let size = actual_size as usize;
        let now = Instant::now();
        time_init(TESTS, size);
        let elapsed = now.elapsed();
        // println!("Init Matrix ({:4}x{:4}) {:?}", size, size, elapsed);
        let cl_dur = time_cl(&add_kernel, &context, &queue, TESTS, size, "add");
        println!(" -- OpenCL matrix matrix addition       ({:4}x{:4}): {:16?}ns", size, size, cl_dur);
        let now = Instant::now();
        time_basic(TESTS, size, "add");
        let basic_dur = now.elapsed().as_nanos() - elapsed.as_nanos();
        println!(" -- Basic matrix matrix addition        ({:4}x{:4}): {:16?}ns", size, size, basic_dur);

        let cl_dur = time_cl(&mul_scalar_kernel, &context, &queue, TESTS, size, "smul");
        println!(" -- OpenCL matrix scalar multiplication ({:4}x{:4}): {:16?}ns", size, size, cl_dur);
        let now = Instant::now();
        time_basic(TESTS, size, "smul");
        let basic_dur = now.elapsed().as_nanos() - elapsed.as_nanos();
        println!(" -- Basic matrix scalar multiplication  ({:4}x{:4}): {:16?}ns", size, size, basic_dur);
        
        // let cl_dur = time_cl(&mul_matrix_kernel, &context, &queue, TESTS, size, "mmul");
        // println!(" -- OpenCL matrix matrix multiplication ({:5}x{:5}): {:16?}ns", size, size, cl_dur);
        // let now = Instant::now();
        // time_basic(TESTS, size, "mmul");
        // let basic_dur = now.elapsed().as_nanos() - elapsed.as_nanos();
        // println!(" -- Basic matrix matrix multiplication  ({:4}x{:4}): {:16?}ns", size, size, basic_dur);

        actual_size *= 2.0;
    }
    Ok(())
}

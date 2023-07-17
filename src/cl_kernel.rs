#![allow(dead_code, unreachable_code)]

use opencl3::error_codes::ClError;
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_float, CL_BLOCKING};
use opencl3::Result;
use std::ptr;

use crate::net::cl_net::ClNet;


const MMUL_VERSION: usize = 3;

const TS: usize = 32;

const WPT: usize = 4;
const RTS: usize = TS / WPT;

const TSM: usize = 8;
const TSN: usize = TSM;
const TSK: usize = 8;
const WPTM: usize = 4;
const WPTN: usize = WPTM;
const RTSM: usize = TSM / WPTM;
const RTSN: usize = TSN / WPTN;
const LPTA: usize = (TSK * TSM) / (RTSM * RTSN);
const LPTB: usize = (TSK * TSN) / (RTSM * RTSN);


const ADD_MATRIX_NAME: &str = "matrix_add";
const ADD_MATRIX_SOURCE: &str = r#"
kernel void matrix_add(
    const int M,
    const int N,
    global float const* A,
    global float const* B,
    global float* C)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalCol * M + globalRow;
    C[i] = A[i] + B[i];
}"#;


const MUL_SCALAR_NAME: &str = "matrix_mul_scalar";
const MUL_SCALAR_SOURCE: &str = r#"
kernel void matrix_mul_scalar(
    const int M,
    const int N,
    global float* x,
    float y)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalCol * M + globalRow;
    x[i] = x[i] * y;
}"#;

// Thanks a lot to https://cnugteren.github.io/tutorial/pages/page1.html for the awesome tutorial.
const MUL_MATRIX_NAME: &str = "matrix_mul_matrix";
const MUL_MATRIX_SOURCE: &str =
if MMUL_VERSION == 1 {
r#"
kernel void matrix_mul_matrix(
    const int M,
    const int N,
    const int K,
    global float const* A,
    global float const* B,
    global float* C)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[k * M + globalRow] * B[globalCol * K + k];
    }
    C[globalCol * M + globalRow] = acc;
}
"#
} else if MMUL_VERSION == 2 { // Version 2 (working)
r#"
#define TS 32
kernel void matrix_mul_matrix(
    const int M,
    const int N,
    const int K,
    global float const* A,
    global float const* B,
    global float* C,
    local float* Asub,
    local float* Bsub)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col;

    float acc = 0.0f;
    
    const int numTiles = K/TS;
    for (int t = 0; t < numTiles; t++) {
        const int tiledRow = TS * t + row;
        const int tiledCol = TS * t + col;
        Asub[col * TS + row] = A[tiledCol * M + globalRow];
        Bsub[col * TS + row] = B[globalCol * K + tiledRow];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0; k<TS; k++) {
            acc += Asub[k * TS + row] * Bsub[col * TS + k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[globalCol * M + globalRow] = acc;
}
"#
} else if MMUL_VERSION == 3 { // Version 3 (working)
r#"
#define TS 32
#define WPT 4
#define RTS (TS / WPT)
kernel void matrix_mul_matrix(
    const int M,
    const int N,
    const int K,
    global float const* A,
    global float const* B,
    global float* C,
    local float* Asub,
    local float* Bsub)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col;

    float acc[WPT];
    for (int w = 0; w < WPT; w++) {
        acc[w] = 0.0f;
    }

    const int numTiles = K / TS;
    for (int t = 0; t < numTiles; t++) {
        for (int w = 0; w < WPT; w++) {
            const int tiledRow = TS * t + row;
            const int tiledCol = TS * t + col;
            Asub[(col + w * RTS) * TS + row] = A[(tiledCol + w * RTS) * M + globalRow];
            Bsub[(col + w * RTS) * TS + row] = B[(globalCol + w * RTS) * K + tiledRow];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT; w++) {
                acc[w] += Asub[k * TS + row] * Bsub[(col + w * RTS) * TS + k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WPT; w++) {
        C[(globalCol + w * RTS) * M + globalRow] = acc[w];
    }
}
"#
} else if MMUL_VERSION == 4 { // Version 4 (not working anymore :( )
r#"
#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))
#define TS 32
#define WPT 4
#define RTS (TS / WPT)
#define TSM 8
#define TSN TSM
#define TSK 8
#define WPTM 4
#define WPTN WPTM
#define RTSM (TSM / WPTM)
#define RTSN (TSN / WPTN)
#define LPTA ((TSK * TSM)/(RTSM * RTSN))
#define LPTB ((TSK * TSN)/(RTSM * RTSN))
kernel void matrix_mul_matrix(
    const int M,
    const int N,
    const int K,
    global float const* A,
    global float const* B,
    global float* C,
    local float* Asub,
    local float* Bsub)
{
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);
    const int offsetM = TSM * get_group_id(0);
    const int offsetN = TSN * get_group_id(1);

    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    #pragma unroll
    for (int wm=0; wm < WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn < WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    const int numTiles = K/TSK;
    int t = 0;
    do {
        #pragma unroll
        for (int la = 0; la < LPTA; la++) {
            int tid = tidn * RTSM + tidm;
            volatile int id = la * RTSN * RTSM + tid;
            int row = MOD2(id, TSM);
            int col = DIV2(id, TSM);
            int tiledIndex = TSK * t + col;
            Asub[col * TSK + row] = A[tiledIndex * M + offsetM + row];
            Bsub[row * TSN + col] = B[tiledIndex * N + offsetN + row];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TSK; k++) {
            #pragma unroll
            for (int wn = 0; wn < WPTN; wn++) {
                int col = tidn + wn * RTSN;
                Breg[wn] = Bsub[col * TSN + k];
            }
            #pragma unroll
            for (int wm = 0; wm < WPTM; wm++) {
                int row = tidm + wm * RTSM;
                Areg = Asub[k * TSK + row];
                #pragma unroll
                for (int wn = 0; wn < WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        t++;
    } while (t < numTiles);

    #pragma unroll
    for (int wm = 0; wm < WPTM; wm++) {
        int globalRow = offsetM + tidm + wm * RTSM;
        #pragma unroll
        for (int wn = 0; wn < WPTN; wn++) {
            int globalCol = offsetN + tidn + wn * RTSN;
            C[globalCol * M + globalRow] = acc[wm][wn];
        }
    }
}
"#
} else {
    ""
};


fn get_mmul_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, k: usize, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>) -> Result<Event> {
    let mut exec_kernel = ExecuteKernel::new(&kernel);
    let event = exec_kernel
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(&(k as u32))
        .set_arg(a)
        .set_arg(b)
        .set_arg(c);
    let result = match MMUL_VERSION {
        1 => {
            event.set_global_work_sizes(&[m, n, 1])
                .set_local_work_sizes(&[TS, TS, 1])
                .enqueue_nd_range(&queue)?
        },
        2 => {
            event.set_arg_local_buffer(TS * TS * 4)
                .set_arg_local_buffer(TS * TS * 4)
                .set_global_work_sizes(&[m, n, 1])
                .set_local_work_sizes(&[TS, TS, 1])
                .enqueue_nd_range(&queue)?
        },
        3 => {
            event.set_arg_local_buffer(TS * TS * 4)
                .set_arg_local_buffer(TS * TS * 4)
                .set_global_work_sizes(&[m, n / WPT, 1])
                .set_local_work_sizes(&[TS, TS / WPT, 1])
                .enqueue_nd_range(&queue)?
        },
        4 => {
            todo!("Find out what broke Version 4 of MMM.");
            event.set_arg_local_buffer(TSK * TSM * 4)
                .set_arg_local_buffer(TSN * (TSK + 2) * 4)
                .set_global_work_sizes(&[m / WPTM, n / WPTN, 1])
                .set_local_work_sizes(&[RTSM, RTSN, 1])
                .enqueue_nd_range(&queue)?
        },
        _ => {
            panic!("Invalid MMUL_VERSION selected! {}", MMUL_VERSION);
        }
    };
    Ok(result)
}

fn get_madd_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(a)
        .set_arg(b)
        .set_arg(c)
        .set_global_work_sizes(&[m, n, 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_smul_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, a: &Buffer<f32>, scalar: f32) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(a)
        .set_arg(&scalar)
        .set_global_work_sizes(&[m, n, 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

pub struct ClStruct {
    device: Device,
    context: Context,
    queue: CommandQueue
}

impl ClStruct {
    pub fn new() -> Result<Self> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
            .first()
            .expect("no device found in platform");
        let device = Device::new(device_id);
        let context = Context::from_device(&device).expect("Context::from_device failed");
        let queue = CommandQueue::create_with_properties(&context, device_id, CL_QUEUE_PROFILING_ENABLE, 0)
            .expect("CommandQueue::create_default failed");
        Ok( Self { device, context, queue } )
    }

    pub fn load_program(&self, source: &str) -> Program {
        Program::create_and_build_from_source(&self.context, source, "")
            .expect("Program::create_and_build_from_source failed")
    }

    pub fn load_kernel(&self, program: &Program, kernel_name: &str) -> Kernel {
        Kernel::create(program, kernel_name).expect("Kernel::create failed")
    }

    pub fn load_kernels(&self, cl_net: &mut ClNet) {
        todo!()
    }

    pub fn create_buffer(&self, rows: usize, cols: usize) -> Buffer<f32> {
        let b = Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, rows * cols, ptr::null_mut());
        match b {
            Ok(bfr) => bfr,
            Err(e) => panic!("Error when creating OpenCL buffer! {}", e)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use crate::cl_kernel::*;
    
    use rand::Rng;

    const TESTS: usize = 5;
    const SIZE: usize = 1_000;
    const WORST_CASE: usize = SIZE * SIZE;

    fn init(name: &str, source: &str) -> Result<(ClStruct, Program, Kernel)> {
        let cl = ClStruct::new()?;
        
        let program = cl.load_program(source);
        let kernel = cl.load_kernel(&program, name);
        Ok((cl, program, kernel))
    }

    #[test]
    fn test_mat_mat_mul() -> Result<()> {
        let (cl_struct, _, kernel) = init(MUL_MATRIX_NAME, MUL_MATRIX_SOURCE)?;
        
        let mut a = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_ONLY, WORST_CASE, ptr::null_mut())?;
        let mut b = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_ONLY, WORST_CASE, ptr::null_mut())?;
        let c = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_WRITE, WORST_CASE, ptr::null_mut())?;
        
        for _ in 0..TESTS {
            let (m, n, k) = 
                (rand::thread_rng().gen_range(10..SIZE),
                rand::thread_rng().gen_range(10..SIZE),
                rand::thread_rng().gen_range(10..SIZE));
    
            let mut m1 = Matrix::new(m, k);
            m1.fill(1.0);
            let mut m2 = Matrix::new(k, n);
            m2.fill(1.0);
    
            let check_m1 = m1.clone();
            let check_m2 = m2.clone();
    
            let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
            let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
            let new_k = if k % 32 != 0 { (k / 32 + 1) * 32 } else { k };
    
            m1.pad(new_m - m, new_k - k);
            m2.pad(new_k - k, new_n - n);
    
            let _a_write_event = cl_struct.queue.enqueue_write_buffer(&mut a, CL_BLOCKING, 0, &m2.get_all(), &[])?;
            let _b_write_event = cl_struct.queue.enqueue_write_buffer(&mut b, CL_BLOCKING, 0, &m1.get_all(), &[])?;
            
            let kernel_event = get_mmul_kernel_event(&kernel, &cl_struct.queue, new_n, new_m, new_k, &a, &b, &c)?;
            
            let mut events: Vec<cl_event> = Vec::default();
            events.push(kernel_event.get());
    
            let mut r1 = vec![0.0; WORST_CASE];
            
            let read_event = cl_struct.queue.enqueue_read_buffer(&c, CL_BLOCKING, 0, &mut r1, &events)?;
            read_event.wait()?;
            
            let mut r = Matrix::new(m, n);
            r.fill_fit(&r1, new_n);
    
            assert_eq!(r, check_m1.multiply(&check_m2).unwrap(), "OpenCL Matrix Matrix Multiplication is not working properly!");
        }
        Ok(())
    }

    #[test]
    fn test_mat_scalar_mul() -> Result<()> {
        let (cl_struct, _, kernel) = init(MUL_SCALAR_NAME, MUL_SCALAR_SOURCE)?;
        
        let mut x = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_WRITE, WORST_CASE, ptr::null_mut())?;

        for _ in 0..TESTS {
            let (m, n) =
                (rand::thread_rng().gen_range(10..SIZE),
                rand::thread_rng().gen_range(10..SIZE));

            let mut m1 = Matrix::new(m, n);
            m1.fill(2.0);
            
            let mut check_m1 = m1.clone();
            
            let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
            let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
            
            m1.pad(new_m - m, new_n - n);

            let scalar: cl_float = rand::thread_rng().gen_range(0..10) as f32;
            
            let _x_write_event = cl_struct.queue.enqueue_write_buffer(&mut x, CL_BLOCKING, 0, &m1.get_all(), &[])?;
            
            let kernel_event = get_smul_kernel_event(&kernel, &cl_struct.queue, new_m, new_n, &x, scalar)?;
            
            let mut events: Vec<cl_event> = Vec::default();
            events.push(kernel_event.get());

            let mut r1 = vec![0.0; WORST_CASE];
            
            let read_event = cl_struct.queue.enqueue_read_buffer(&x, CL_BLOCKING, 0, &mut r1, &events)?;
            read_event.wait()?;
            let mut r = Matrix::new(m, n);
            r.fill_fit(&r1, new_n);

            check_m1.multiply_scalar(scalar);
            assert_eq!(r, check_m1, "OpenCL Matrix Scalar Multiplication is not working properly!");
        }
        Ok(())
    }

    #[test]
    fn test_mat_mat_add() -> Result<()> {
        let (cl_struct, _, kernel) = init(ADD_MATRIX_NAME, ADD_MATRIX_SOURCE)?;

        let mut a = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_ONLY, WORST_CASE, ptr::null_mut())?;
        let mut b = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_ONLY, WORST_CASE, ptr::null_mut())?;
        let c = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_WRITE_ONLY, WORST_CASE, ptr::null_mut())?;

        for _ in 0..TESTS {
            let (m, n) = 
                (rand::thread_rng().gen_range(10..SIZE),
                rand::thread_rng().gen_range(10..SIZE));

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
            
            let _x_write_event = cl_struct.queue.enqueue_write_buffer(&mut a, CL_BLOCKING, 0, &m1.get_all(), &[])?;
            let _y_write_event = cl_struct.queue.enqueue_write_buffer(&mut b, CL_BLOCKING, 0, &m2.get_all(), &[])?;
            
            let kernel_event = get_madd_kernel_event(&kernel, &cl_struct.queue, new_m, new_n, &a, &b, &c)?;
            
            let mut events: Vec<cl_event> = Vec::default();
            events.push(kernel_event.get());

            let mut r1 = vec![0.0; WORST_CASE];
            
            let read_event = cl_struct.queue.enqueue_read_buffer(&c, CL_BLOCKING, 0, &mut r1, &events)?;
            read_event.wait()?;

            let mut r = Matrix::new(m, n);
            r.fill_fit(&r1, new_n);

            assert_eq!(r, check_m1.add(&check_m2).unwrap(), "OpenCL Matrix Matrix Addition is not working properly!");
        }
        Ok(())
    }
}
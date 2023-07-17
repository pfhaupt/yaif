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
use std::collections::HashMap;
use std::ptr;

use rand_distr::{Normal, Distribution};


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
    if (i < M * N) C[i] = A[i] + B[i];
}"#;

const SUB_MATRIX_NAME: &str = "matrix_sub";
const SUB_MATRIX_SOURCE: &str = r#"
kernel void matrix_sub(
    const int M,
    const int N,
    global float const* A,
    global float const* B,
    global float* C)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalCol * M + globalRow;
    if (i < M * N) C[i] = A[i] - B[i];
}"#;

const ADD_MATRIX_INLINE_NAME: &str = "matrix_add_inline";
const ADD_MATRIX_INLINE_SOURCE: &str = r#"
kernel void matrix_add_inline(
    const int M,
    const int N,
    global float* A,
    global float const* B)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalCol * M + globalRow;
    if (i < M * N) A[i] = A[i] + B[i];
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
    if (i < M * N) x[i] = x[i] * y;
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

const FILL_MATRIX_NAME: &str = "matrix_fill";
const FILL_MATRIX_SOURCE: &str =
r#"
kernel void matrix_fill(
    const int M,
    global float* x,
    float y)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalCol * M + globalRow;
    if (i < M) x[i] = y;
}
"#;

const FILL_MATRIX_VEC_NAME: &str = "matrix_vec_fill";
const FILL_MATRIX_VEC_SOURCE: &str =
r#"
kernel void matrix_vec_fill(
    const int M,
    global float* x,
    global float* y)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalCol * M + globalRow;
    if (i < M) x[i] = y[i];
}
"#;

const SIGMOID_NAME: &str = "sigmoid";
const SIGMOID_SOURCE: &str =
r#"
#define SIGMOID(i) (1.0 / (1.0 + exp(-(i))))
kernel void sigmoid(
    const int M,
    const int N,
    global float const* A,
    global float* B)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalCol * M + globalRow;
    if (i < M * N) B[i] = SIGMOID(A[i]);
}"#;

const DER_SIGMOID_NAME: &str = "der_sigmoid";
const DER_SIGMOID_SOURCE: &str =
r#"
#define SIGMOID(i) (1.0 / (1.0 + exp(-(i))))
kernel void der_sigmoid(
    const int M,
    const int N,
    global float const* A,
    global float* B)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalCol * M + globalRow;
    if (i < M * N) {
        float s = SIGMOID(A[i]);
        B[i] = s * (1.0 - s);
    }
}"#;

const HADAMARD_NAME: &str = "matrix_hadamard";
const HADAMARD_SOURCE: &str = r#"
kernel void matrix_hadamard(
    const int M,
    const int N,
    global float const* A,
    global float const* B,
    global float* C)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalCol * M + globalRow;
    if (i < M * N) C[i] = A[i] * B[i];
}"#;

const TRANSPOSE_NAME: &str = "matrix_transpose";
const TRANSPOSE_SOURCE: &str =
r#"
kernel void matrix_transpose(
    const int M,
    const int N,
    global float* A,
    global float* B)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalRow + M * globalCol;
    const int j = globalCol + N * globalRow;
    B[j] = A[i];
}"#;

const DYADIC_NAME: &str = "matrix_dyadic";
const DYADIC_SOURCE: &str =
r#"
kernel void matrix_dyadic(
    const int M,
    const int N,
    global float const* A,
    global float const* B,
    global float* C)
{
    const int i = get_global_id(0);
    for (int j = 0; j < N; j++) {
        float v = A[i] * B[j];
        C[i * N + j] = v;
    }
}
"#;

const IMPLEMENTED_KERNELS: [&str; 12] = [
    ADD_MATRIX_NAME,
    ADD_MATRIX_INLINE_NAME,
    SUB_MATRIX_NAME,
    MUL_MATRIX_NAME,
    MUL_SCALAR_NAME,
    FILL_MATRIX_NAME,
    FILL_MATRIX_VEC_NAME,
    HADAMARD_NAME,
    DYADIC_NAME,
    TRANSPOSE_NAME,
    SIGMOID_NAME,
    DER_SIGMOID_NAME];

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

fn get_msub_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>) -> Result<Event> {
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

fn get_madd_inline_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, a: &Buffer<f32>, b: &Buffer<f32>) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(a)
        .set_arg(b)
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

fn get_fill_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, size: usize, a: &Buffer<f32>, scalar: f32) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(a)
        .set_arg(&scalar)
        .set_global_work_sizes(&[size, 1, 1])
        .set_local_work_sizes(&[TS, 1, 1])
        .enqueue_nd_range(&queue)
}

fn get_fill_vec_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, size: usize, a: &Buffer<f32>, b: &Buffer<f32>) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(a)
        .set_arg(b)
        .set_global_work_sizes(&[size, 1, 1])
        .set_local_work_sizes(&[TS, 1, 1])
        .enqueue_nd_range(&queue)
}

fn get_hadamard_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>) -> Result<Event> {
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

fn get_dyadic_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>) -> Result<Event> {
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

fn get_transpose_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, a: &Buffer<f32>, b: &Buffer<f32>) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(a)
        .set_arg(b)
        .set_global_work_sizes(&[m, n, 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_sigmoid_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, a: &Buffer<f32>, b: &Buffer<f32>) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(a)
        .set_arg(b)
        .set_global_work_sizes(&[m, n, 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_der_sigmoid_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, a: &Buffer<f32>, b: &Buffer<f32>) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(a)
        .set_arg(b)
        .set_global_work_sizes(&[m, n, 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

pub struct ClStruct {
    device: Device,
    context: Context,
    queue: CommandQueue,
    kernels: HashMap<String, Kernel>
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
        let kernels = HashMap::new();
        Ok( Self { device, context, queue, kernels } )
    }

    pub fn load_program(&self, source: &str) -> Program {
        Program::create_and_build_from_source(&self.context, source, "")
            .expect("Program::create_and_build_from_source failed")
    }

    pub fn load_kernel(&self, program: &Program, kernel_name: &str) -> Kernel {
        Kernel::create(program, kernel_name).expect("Kernel::create failed")
    }

    pub fn load_kernels(&mut self) {
        assert!(IMPLEMENTED_KERNELS.len() == 12, "Can not load all kernels yet");
        let mut add_kernel = |name: &str, source: &str| {
            let p = self.load_program(source);
            let k = self.load_kernel(&p, name);
            self.kernels.insert(String::from(name), k);
        };
        add_kernel(ADD_MATRIX_NAME, ADD_MATRIX_SOURCE);
        add_kernel(ADD_MATRIX_INLINE_NAME, ADD_MATRIX_INLINE_SOURCE);
        add_kernel(SUB_MATRIX_NAME, SUB_MATRIX_SOURCE);
        add_kernel(MUL_SCALAR_NAME, MUL_SCALAR_SOURCE);
        add_kernel(MUL_MATRIX_NAME, MUL_MATRIX_SOURCE);
        add_kernel(FILL_MATRIX_NAME, FILL_MATRIX_SOURCE);
        add_kernel(FILL_MATRIX_VEC_NAME, FILL_MATRIX_VEC_SOURCE);
        add_kernel(HADAMARD_NAME, HADAMARD_SOURCE);
        add_kernel(TRANSPOSE_NAME, TRANSPOSE_SOURCE);
        add_kernel(DYADIC_NAME, DYADIC_SOURCE);
        add_kernel(SIGMOID_NAME, SIGMOID_SOURCE);
        add_kernel(DER_SIGMOID_NAME, DER_SIGMOID_SOURCE);

    }

    pub fn create_buffer(&self, rows: usize, cols: usize) -> Option<Buffer<f32>> {
        let b = Buffer::<cl_float>::create(&self.context, CL_MEM_READ_WRITE, (rows * cols).max(1), ptr::null_mut());
        match b {
            Ok(bfr) => Some(bfr),
            Err(e) => panic!("Error when creating OpenCL buffer! {}", e)
        }
    }

    pub fn read_buffer(&self, buffer: &Option<Buffer<f32>>, size: usize) -> Result<Vec<f32>> {
        match buffer {
            None => {
                Err(ClError(-61))
            },
            Some(bfr) => {
                let mut r1 = vec![0.0; size];
                let read_event = self.queue.enqueue_read_buffer(bfr, CL_BLOCKING, 0, &mut r1, &vec![])?;
                read_event.wait()?;
                Ok(r1)
            }
        }
    }

    pub fn fill_vec(&self, buffer: &Option<Buffer<f32>>, size: usize, values: Vec<f32>) -> Result<()> {
        match self.kernels.get(&String::from(FILL_MATRIX_VEC_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(k) => {
                match buffer {
                    None => Err(ClError(-61)),
                    Some(bfr) => {
                        let new_size = if size % 32 != 0 { (size / 32 + 1) * 32 } else { size };
                        
                        let mut val_bfr = self.create_buffer(new_size, 1).ok_or(ClError(-61))?;
                        let _x_write_event = self.queue.enqueue_write_buffer(&mut val_bfr, CL_BLOCKING, 0, &values, &[])?;
            
                        let fill_event = get_fill_vec_kernel_event(k,
                            &self.queue, size, new_size, bfr, &val_bfr)?;
                        let mut events: Vec<cl_event> = Vec::default();
                        events.push(fill_event.get());
                        Ok(())
                    }
                }
            }
        }
    }

    pub fn fill_scalar(&self, buffer: &Option<Buffer<f32>>, size: usize, val: f32) -> Result<()> {
        match self.kernels.get(&String::from(FILL_MATRIX_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(k) => {
                match buffer {
                    None => Err(ClError(-61)),
                    Some(bfr) => {
                        let new_size = if size % 32 != 0 { (size / 32 + 1) * 32 } else { size };
                        let fill_event = get_fill_kernel_event(k,
                            &self.queue,
                            size,
                            new_size,
                        &bfr,
                            val)?;
                            let mut events: Vec<cl_event> = Vec::default();
                            events.push(fill_event.get());
                            Ok(())
                        }
                }
            }
        }
    }

    pub fn fill_gauss(&self, buffer: &Option<Buffer<f32>>, size: usize, mean: f32, variance: f32) -> Result<()> {
        let new_size = if size % 32 != 0 { (size / 32 + 1) * 32 } else { size };
                        
        let mut r = vec![0.0; new_size];
        let normal = Normal::new(mean, variance).unwrap();
        for i in 0..new_size { r[i] = normal.sample(&mut rand::thread_rng()); }
        self.fill_vec(buffer, size, r)
    }

    pub fn matrix_mult(&self, a: &mut Option<Buffer<f32>>, b: &mut Option<Buffer<f32>>, c: &mut Option<Buffer<f32>>, m: usize, n: usize, k: usize) -> Result<()> {
        match self.kernels.get(&String::from(MUL_MATRIX_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
                let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
                let new_k = if k % 32 != 0 { (k / 32 + 1) * 32 } else { k };

                let a = a.as_mut().unwrap();
                let b = b.as_mut().unwrap();
                let c = c.as_mut().unwrap();

                let kernel_event = get_mmul_kernel_event(&kernel, &self.queue, new_m, new_n, new_k, a, b, c)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())
            }
        }
    }

    pub fn matrix_add(&self, a: &mut Option<Buffer<f32>>, b: &mut Option<Buffer<f32>>, c: &mut Option<Buffer<f32>>, m: usize, n: usize) -> Result<()> {
        match self.kernels.get(&String::from(ADD_MATRIX_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
                let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
                
                let a = a.as_mut().unwrap();
                let b = b.as_mut().unwrap();
                let c = c.as_mut().unwrap();

                let kernel_event = get_madd_kernel_event(&kernel, &self.queue, new_m, new_n, a, b, c)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())

            }
        }
    }

    pub fn matrix_sub(&self, a: &mut Option<Buffer<f32>>, b: &mut Option<Buffer<f32>>, c: &mut Option<Buffer<f32>>, m: usize, n: usize) -> Result<()> {
        match self.kernels.get(&String::from(SUB_MATRIX_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
                let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
                
                let a = a.as_mut().unwrap();
                let b = b.as_mut().unwrap();
                let c = c.as_mut().unwrap();

                let kernel_event = get_msub_kernel_event(&kernel, &self.queue, new_m, new_n, a, b, c)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())

            }
        }
    }

    pub fn matrix_add_inline(&self, a: &mut Option<Buffer<f32>>, b: &mut Option<Buffer<f32>>, m: usize, n: usize) -> Result<()> {
        match self.kernels.get(&String::from(ADD_MATRIX_INLINE_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
                let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
                
                let a = a.as_mut().unwrap();
                let b = b.as_mut().unwrap();

                let kernel_event = get_madd_inline_kernel_event(&kernel, &self.queue, new_m, new_n, a, b)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())

            }
        }
    }

    pub fn matrix_hadamard(&self, a: &mut Option<Buffer<f32>>, b: &mut Option<Buffer<f32>>, c: &mut Option<Buffer<f32>>, m: usize, n: usize) -> Result<()> {
        match self.kernels.get(&String::from(HADAMARD_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
                let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
                
                let a = a.as_mut().unwrap();
                let b = b.as_mut().unwrap();
                let c = c.as_mut().unwrap();

                let kernel_event = get_hadamard_kernel_event(&kernel, &self.queue, new_m, new_n, a, b, c)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())
            }
        }
    }
    
    pub fn matrix_transpose(&self, a: &mut Option<Buffer<f32>>, b: &mut Option<Buffer<f32>>, m: usize, n: usize) -> Result<()> {
        match self.kernels.get(&String::from(TRANSPOSE_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
                let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
                
                let a = a.as_mut().unwrap();
                let b = b.as_mut().unwrap();

                let kernel_event = get_transpose_kernel_event(&kernel, &self.queue, new_m, new_n, a, b)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())

            }
        }
    }

    pub fn matrix_dyadic(&self, a: &mut Option<Buffer<f32>>, b: &mut Option<Buffer<f32>>, c: &mut Option<Buffer<f32>>, m: usize, n: usize) -> Result<()> {
        match self.kernels.get(&String::from(DYADIC_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
                let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
                
                let a = a.as_mut().unwrap();
                let b = b.as_mut().unwrap();
                let c = c.as_mut().unwrap();

                let kernel_event = get_dyadic_kernel_event(&kernel, &self.queue, new_m, new_n, a, b, c)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())
            }
        }
    }

    pub fn sigmoid(&self, a: &mut Option<Buffer<f32>>, b: &mut Option<Buffer<f32>>, m: usize, n: usize) -> Result<()> {
        match self.kernels.get(&String::from(SIGMOID_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
                let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
                
                let a = a.as_mut().unwrap();
                let b = b.as_mut().unwrap();

                let kernel_event = get_sigmoid_kernel_event(&kernel, &self.queue, new_n, new_m, a, b)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())
            }
        }
    }

    pub fn der_sigmoid(&self, a: &mut Option<Buffer<f32>>, b: &mut Option<Buffer<f32>>, m: usize, n: usize) -> Result<()> {
        match self.kernels.get(&String::from(DER_SIGMOID_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
                let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
                
                let a = a.as_mut().unwrap();
                let b = b.as_mut().unwrap();

                let kernel_event = get_der_sigmoid_kernel_event(&kernel, &self.queue, new_n, new_m, a, b)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())
            }
        }
    }

    pub fn copy_buffer(&self, a: &mut Option<Buffer<f32>>, b: &mut Option<Buffer<f32>>, m: usize, n: usize) -> Result<()> {
        match self.kernels.get(&String::from(FILL_MATRIX_VEC_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(k) => {
                let size = m * n;
                let new_size = if size % 32 != 0 { (size / 32 + 1) * 32 } else { size };
                
                let a = a.as_mut().unwrap();
                let b = b.as_mut().unwrap();

                let fill_event = get_fill_vec_kernel_event(k,
                    &self.queue, size, new_size, a, b)?;
                let mut events: Vec<cl_event> = Vec::default();
                events.push(fill_event.get());
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cl_kernel::*;

    const TESTS: usize = 5;
    const SIZE: usize = 1_000;

    fn initialize() -> ClStruct {
        let mut c = ClStruct::new().unwrap();
        c.load_kernels();
        c
    }

    #[test]
    fn test_create_buffer() -> Result<()> {
        let cl_struct = initialize();
        for _ in 0..TESTS {
            let bfr = cl_struct.create_buffer(SIZE, SIZE);
            assert!(bfr.is_some(), "create_buffer() should be able to create a {}x{} buffer.", SIZE, SIZE);
        }
        Ok(())
    }

    #[test]
    fn test_fill_buffer_scalar() -> Result<()> {
        let cl_struct = initialize();
        for i in 0..TESTS {
            let bfr = cl_struct.create_buffer(SIZE, SIZE);
            assert!(bfr.is_some(), "create_buffer() should be able to create a {}x{} buffer.", SIZE, SIZE);

            let e = cl_struct.fill_scalar(&bfr, SIZE * SIZE, i as f32);
            assert!(e.is_ok(), "fill_scalar() did not work properly: {:?}", e.err());

            let r = cl_struct.read_buffer(&bfr, SIZE * SIZE);
            assert!(r.is_ok(), "Could not read from the buffer: {:?}", r.err());

            let mut s = 0.0f32;
            for e in r.unwrap() { s += e }
            assert_eq!(s, i as f32 * (SIZE * SIZE) as f32, "fill_scalar() did not fill the whole buffer.");
        }
        Ok(())
    }

    #[test]
    fn test_fill_buffer_vec_full() -> Result<()> {
        let cl_struct = initialize();
        for i in 0..TESTS {
            let bfr = cl_struct.create_buffer(SIZE, SIZE);
            assert!(bfr.is_some(), "create_buffer() should be able to create a {}x{} buffer.", SIZE, SIZE);

            let v = vec![i as f32; SIZE * SIZE];
            let e = cl_struct.fill_vec(&bfr, SIZE * SIZE, v.clone());
            assert!(e.is_ok(), "fill_vec() did not work properly: {:?}", e.err());

            let r = cl_struct.read_buffer(&bfr, SIZE * SIZE);
            assert!(r.is_ok(), "Could not read from the buffer: {:?}", r.err());
            assert_eq!(r.unwrap(), v, "fill_vec() was supposed to fill the whole buffer.");
        }
        Ok(())
    }

    #[test]
    fn test_fill_buffer_vec_part() -> Result<()> {
        let cl_struct= initialize();
        for i in 0..TESTS {
            let bfr = cl_struct.create_buffer(SIZE, SIZE);
            assert!(bfr.is_some(), "create_buffer() should be able to create a {}x{} buffer.", SIZE, SIZE);

            let v = vec![10.0 * (i + 1) as f32; SIZE];
            let e = cl_struct.fill_vec(&bfr, SIZE, v.clone());
            assert!(e.is_ok(), "fill_vec() did not work properly: {:?}", e.err());

            let r = cl_struct.read_buffer(&bfr, SIZE * SIZE);
            assert!(r.is_ok(), "Could not read from the buffer: {:?}", r.err());            
            let r = r.unwrap();

            assert_eq!(r[0..SIZE], v, "First {} elements of the buffer are supposed to contain {}", SIZE, v[0]);
            assert_eq!(r[SIZE..].iter().sum::<f32>(), 0.0, "Last {} elements of the buffer are supposed to only contain 0s.", SIZE * SIZE - SIZE);
        }
        Ok(())
    }

    // #[test]
    // fn test_mat_mat_mul() -> Result<()> {
    //     let (cl_struct, _, kernel) = init(MUL_MATRIX_NAME, MUL_MATRIX_SOURCE)?;
        
    //     let mut a = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_ONLY, WORST_CASE, ptr::null_mut())?;
    //     let mut b = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_ONLY, WORST_CASE, ptr::null_mut())?;
    //     let c = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_WRITE, WORST_CASE, ptr::null_mut())?;
        
    //     for _ in 0..TESTS {
    //         let (m, n, k) = 
    //             (rand::thread_rng().gen_range(10..SIZE),
    //             rand::thread_rng().gen_range(10..SIZE),
    //             rand::thread_rng().gen_range(10..SIZE));
    
    //         let mut m1 = Matrix::new(m, k);
    //         m1.fill(1.0);
    //         let mut m2 = Matrix::new(k, n);
    //         m2.fill(1.0);
    
    //         let check_m1 = m1.clone();
    //         let check_m2 = m2.clone();
    
    //         let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
    //         let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
    //         let new_k = if k % 32 != 0 { (k / 32 + 1) * 32 } else { k };
    
    //         m1.pad(new_m - m, new_k - k);
    //         m2.pad(new_k - k, new_n - n);
    
    //         let _a_write_event = cl_struct.queue.enqueue_write_buffer(&mut a, CL_BLOCKING, 0, &m2.get_all(), &[])?;
    //         let _b_write_event = cl_struct.queue.enqueue_write_buffer(&mut b, CL_BLOCKING, 0, &m1.get_all(), &[])?;
            
    //         let kernel_event = get_mmul_kernel_event(&kernel, &cl_struct.queue, new_n, new_m, new_k, &a, &b, &c)?;
            
    //         let mut events: Vec<cl_event> = Vec::default();
    //         events.push(kernel_event.get());
    
    //         let mut r1 = vec![0.0; WORST_CASE];
            
    //         let read_event = cl_struct.queue.enqueue_read_buffer(&c, CL_BLOCKING, 0, &mut r1, &events)?;
    //         read_event.wait()?;
            
    //         let mut r = Matrix::new(m, n);
    //         r.fill_fit(&r1, new_n);
    
    //         assert_eq!(r, check_m1.multiply(&check_m2).unwrap(), "OpenCL Matrix Matrix Multiplication is not working properly!");
    //     }
    //     Ok(())
    // }

    // #[test]
    // fn test_mat_scalar_mul() -> Result<()> {
    //     let (cl_struct, _, kernel) = init(MUL_SCALAR_NAME, MUL_SCALAR_SOURCE)?;
        
    //     let mut x = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_WRITE, WORST_CASE, ptr::null_mut())?;

    //     for _ in 0..TESTS {
    //         let (m, n) =
    //             (rand::thread_rng().gen_range(10..SIZE),
    //             rand::thread_rng().gen_range(10..SIZE));

    //         let mut m1 = Matrix::new(m, n);
    //         m1.fill(2.0);
            
    //         let mut check_m1 = m1.clone();
            
    //         let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
    //         let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };
            
    //         m1.pad(new_m - m, new_n - n);

    //         let scalar: cl_float = rand::thread_rng().gen_range(0..10) as f32;
            
    //         let _x_write_event = cl_struct.queue.enqueue_write_buffer(&mut x, CL_BLOCKING, 0, &m1.get_all(), &[])?;
            
    //         let kernel_event = get_smul_kernel_event(&kernel, &cl_struct.queue, new_m, new_n, &x, scalar)?;
            
    //         let mut events: Vec<cl_event> = Vec::default();
    //         events.push(kernel_event.get());

    //         let mut r1 = vec![0.0; WORST_CASE];
            
    //         let read_event = cl_struct.queue.enqueue_read_buffer(&x, CL_BLOCKING, 0, &mut r1, &events)?;
    //         read_event.wait()?;
    //         let mut r = Matrix::new(m, n);
    //         r.fill_fit(&r1, new_n);

    //         check_m1.multiply_scalar(scalar);
    //         assert_eq!(r, check_m1, "OpenCL Matrix Scalar Multiplication is not working properly!");
    //     }
    //     Ok(())
    // }

    // #[test]
    // fn test_mat_mat_add() -> Result<()> {
    //     let (cl_struct, _, kernel) = init(ADD_MATRIX_NAME, ADD_MATRIX_SOURCE)?;

    //     let mut a = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_ONLY, WORST_CASE, ptr::null_mut())?;
    //     let mut b = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_READ_ONLY, WORST_CASE, ptr::null_mut())?;
    //     let c = Buffer::<cl_float>::create(&cl_struct.context, CL_MEM_WRITE_ONLY, WORST_CASE, ptr::null_mut())?;

    //     for _ in 0..TESTS {
    //         let (m, n) = 
    //             (rand::thread_rng().gen_range(10..SIZE),
    //             rand::thread_rng().gen_range(10..SIZE));

    //         let mut m1 = Matrix::new(m, n);
    //         m1.fill(2.0);
    //         let mut m2 = Matrix::new(m, n);
    //         m2.fill(3.0);

    //         let check_m1 = m1.clone();
    //         let check_m2 = m2.clone();

    //         let new_m = if m % 32 != 0 { (m / 32 + 1) * 32 } else { m };
    //         let new_n = if n % 32 != 0 { (n / 32 + 1) * 32 } else { n };

    //         m1.pad(new_m - m, new_n - n);
    //         m2.pad(new_m - m, new_n - n);
            
    //         let _x_write_event = cl_struct.queue.enqueue_write_buffer(&mut a, CL_BLOCKING, 0, &m1.get_all(), &[])?;
    //         let _y_write_event = cl_struct.queue.enqueue_write_buffer(&mut b, CL_BLOCKING, 0, &m2.get_all(), &[])?;
            
    //         let kernel_event = get_madd_kernel_event(&kernel, &cl_struct.queue, new_m, new_n, &a, &b, &c)?;
            
    //         let mut events: Vec<cl_event> = Vec::default();
    //         events.push(kernel_event.get());

    //         let mut r1 = vec![0.0; WORST_CASE];
            
    //         let read_event = cl_struct.queue.enqueue_read_buffer(&c, CL_BLOCKING, 0, &mut r1, &events)?;
    //         read_event.wait()?;

    //         let mut r = Matrix::new(m, n);
    //         r.fill_fit(&r1, new_n);

    //         assert_eq!(r, check_m1.add(&check_m2).unwrap(), "OpenCL Matrix Matrix Addition is not working properly!");
    //     }
    //     Ok(())
    // }
}
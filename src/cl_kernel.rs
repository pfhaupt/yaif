#![allow(dead_code, unreachable_code)]

use opencl3::error_codes::ClError;
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_float, CL_BLOCKING};
use opencl3::Result;
use std::collections::HashMap;
use std::ptr;

use crate::cl_buffer::ClBuffer;

use rand_distr::{Normal, Distribution};

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
    global float const* First,
    global float const* Second,
    global float* Result)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    const int i = globalCol * M + globalRow;
    Result[i] = First[i] + Second[i];
}"#;

const SUB_MATRIX_NAME: &str = "matrix_sub";
const SUB_MATRIX_SOURCE: &str = r#"
kernel void matrix_sub(
    const int M,
    const int N,
    global float const* First,
    global float const* Second,
    global float* Result)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    const int i = globalCol * M + globalRow;
    Result[i] = First[i] - Second[i];
}"#;

const ADD_MATRIX_INLINE_NAME: &str = "matrix_add_inline";
const ADD_MATRIX_INLINE_SOURCE: &str = r#"
kernel void matrix_add_inline(
    const int M,
    const int N,
    global float* Target,
    global float const* Other)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    const int i = globalCol * M + globalRow;
    Target[i] = Target[i] + Other[i];
}"#;

const SUB_MATRIX_INLINE_NAME: &str = "matrix_sub_inline";
const SUB_MATRIX_INLINE_SOURCE: &str = r#"
kernel void matrix_sub_inline(
    const int M,
    const int N,
    global float* Target,
    global float const* Other)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    const int i = globalCol * M + globalRow;
    Target[i] = Target[i] - Other[i];
}"#;

const MUL_SCALAR_NAME: &str = "matrix_mul_scalar";
const MUL_SCALAR_SOURCE: &str = r#"
kernel void matrix_mul_scalar(
    const int M,
    const int N,
    global float* buffer,
    float y)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    const int i = globalCol * M + globalRow;
    buffer[i] = buffer[i] * y;
}"#;

const MMUL_VERSION: usize = 1;

// Thanks a lot to https://cnugteren.github.io/tutorial/pages/page1.html for the awesome tutorial.
const MUL_MATRIX_NAME: &str = "matrix_mul_matrix";
const MUL_MATRIX_SOURCE: &str =
if MMUL_VERSION == 1 { // Version 1 (naive implementation, works for any matrices)
r#"
kernel void matrix_mul_matrix(
    const int M,
    const int N,
    const int K,
    global float const* First,
    global float const* Second,
    global float* Third)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += First[k * M + globalRow] * Second[globalCol * K + k];
    }
    Third[globalCol * M + globalRow] = acc;
}
"#
} else if MMUL_VERSION == 2 { // Version 2 (only works properly for Dim(2^n))
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
    if (globalRow >= M || globalCol >= N) return;

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
} else if MMUL_VERSION == 3 { // Version 3 (only works properly for Dim(2^n))
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
} else if MMUL_VERSION == 4 { // Version 4 (only works properly for Dim(2^n))
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
    if (offsetM >= M || offsetN >= N) return;

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
    const int N,
    global float* buffer,
    float y)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    const int i = globalCol * M + globalRow;
    buffer[i] = y;
}
"#;

const FILL_MATRIX_VEC_NAME: &str = "matrix_vec_fill";
const FILL_MATRIX_VEC_SOURCE: &str =
r#"
kernel void matrix_vec_fill(
    const int M,
    const int N,
    global float* from,
    global float* to)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    const int i = globalCol * M + globalRow;
    to[i] = from[i];
}
"#;

const SIGMOID_NAME: &str = "sigmoid";
const SIGMOID_SOURCE: &str =
r#"
#define SIGMOID(i) (1.0 / (1.0 + exp(-(i))))
kernel void sigmoid(
    const int M,
    const int N,
    global float const* from,
    global float* to)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    const int i = globalCol * M + globalRow;
    to[i] = SIGMOID(from[i]);
}"#;

const DER_SIGMOID_NAME: &str = "der_sigmoid";
const DER_SIGMOID_SOURCE: &str =
r#"
#define SIGMOID(i) (1.0 / (1.0 + exp(-(i))))
kernel void der_sigmoid(
    const int M,
    const int N,
    global float const* from,
    global float* to)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    const int i = globalCol * M + globalRow;
    float s = SIGMOID(from[i]);
    to[i] = s * (1.0 - s);
}"#;

const HADAMARD_NAME: &str = "matrix_hadamard";
const HADAMARD_SOURCE: &str = r#"
kernel void matrix_hadamard(
    const int M,
    const int N,
    global float const* First,
    global float const* Second,
    global float* Result)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    const int i = globalCol * M + globalRow;
    Result[i] = First[i] * Second[i];
}"#;

const TRANSPOSE_NAME: &str = "matrix_transpose";
const TRANSPOSE_SOURCE: &str =
r#"
kernel void matrix_transpose(
    const int M,
    const int N,
    global float* from,
    global float* to)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    const int i = globalRow + M * globalCol;
    const int j = globalCol + N * globalRow;
    to[i] = from[j];
}"#;

const DYADIC_NAME: &str = "matrix_dyadic";
const DYADIC_SOURCE: &str =
r#"
kernel void matrix_dyadic(
    const int M,
    const int N,
    global float const* First,
    global float const* Second,
    global float* Result)
{
    const int i = get_global_id(0);
    if (i >= M) return;
    for (int j = 0; j < N; j++) {
        Result[i * N + j] = First[i] * Second[j];;
    }
}
"#;

const IMPLEMENTED_KERNELS: [&str; 13] = [
    ADD_MATRIX_NAME,
    ADD_MATRIX_INLINE_NAME,
    SUB_MATRIX_NAME,
    SUB_MATRIX_INLINE_NAME,
    MUL_MATRIX_NAME,
    MUL_SCALAR_NAME,
    FILL_MATRIX_NAME,
    FILL_MATRIX_VEC_NAME,
    HADAMARD_NAME,
    DYADIC_NAME,
    TRANSPOSE_NAME,
    SIGMOID_NAME,
    DER_SIGMOID_NAME];

const fn pad(val: usize, multiple: usize) -> usize { if val % multiple == 0 { val } else { (val / multiple + 1) * multiple } }

fn get_mmul_kernel_event(kernel: &Kernel, queue: &CommandQueue, first: &Buffer<f32>, second: &Buffer<f32>, result: &Buffer<f32>, m: usize, n: usize, k: usize) -> Result<Event> {
    let mut exec_kernel = ExecuteKernel::new(&kernel);
    let event = exec_kernel
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(&(k as u32))
        .set_arg(first)
        .set_arg(second)
        .set_arg(result);
    let result = match MMUL_VERSION {
        1 => {
            event.set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
                .set_local_work_sizes(&[TS, TS, 1])
                .enqueue_nd_range(&queue)?
        },
        2 => {
            event.set_arg_local_buffer(TS * TS * 4)
                .set_arg_local_buffer(TS * TS * 4)
                .set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
                .set_local_work_sizes(&[TS, TS, 1])
                .enqueue_nd_range(&queue)?
        },
        3 => {
            event.set_arg_local_buffer(TS * TS * 4)
                .set_arg_local_buffer(TS * TS * 4)
                .set_global_work_sizes(&[pad(m, TS), pad(n, TS) / WPT, 1])
                .set_local_work_sizes(&[TS, TS / WPT, 1])
                .enqueue_nd_range(&queue)?
        },
        4 => {
            event.set_arg_local_buffer(TSK * TSM * 4)
                .set_arg_local_buffer(TSN * (TSK + 2) * 4)
                .set_global_work_sizes(&[pad(m, TS) / WPTM, pad(n, TS) / WPTN, 1])
                .set_local_work_sizes(&[RTSM, RTSN, 1])
                .enqueue_nd_range(&queue)?
        },
        _ => {
            panic!("Invalid MMUL_VERSION selected! {}", MMUL_VERSION);
        }
    };
    Ok(result)
}

fn get_madd_kernel_event(kernel: &Kernel, queue: &CommandQueue, first: &Buffer<f32>, second: &Buffer<f32>, result: &Buffer<f32>, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(first)
        .set_arg(second)
        .set_arg(result)
        .set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_msub_kernel_event(kernel: &Kernel, queue: &CommandQueue, first: &Buffer<f32>, second: &Buffer<f32>, result: &Buffer<f32>, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(first)
        .set_arg(second)
        .set_arg(result)
        .set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_madd_inline_kernel_event(kernel: &Kernel, queue: &CommandQueue, target: &Buffer<f32>, other: &Buffer<f32>, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(target)
        .set_arg(other)
        .set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_msub_inline_kernel_event(kernel: &Kernel, queue: &CommandQueue, target: &Buffer<f32>, other: &Buffer<f32>, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(target)
        .set_arg(other)
        .set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_smul_kernel_event(kernel: &Kernel, queue: &CommandQueue, buffer: &Buffer<f32>, scalar: f32, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(buffer)
        .set_arg(&scalar)
        .set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_fill_kernel_event(kernel: &Kernel, queue: &CommandQueue, buffer: &Buffer<f32>, scalar: f32, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&((m * n) as u32))
        .set_arg(&(2 as u32))
        .set_arg(buffer)
        .set_arg(&scalar)
        .set_global_work_sizes(&[pad(m * n, TS), 1, 1])
        .set_local_work_sizes(&[TS, 1, 1])
        .enqueue_nd_range(&queue)
}

fn get_fill_vec_kernel_event(kernel: &Kernel, queue: &CommandQueue, from: &Buffer<f32>, to: &Buffer<f32>, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&((m * n) as u32))
        .set_arg(&(2 as u32))
        .set_arg(from)
        .set_arg(to)
        .set_global_work_sizes(&[pad(m * n, TS), 1, 1])
        .set_local_work_sizes(&[TS, 1, 1])
        .enqueue_nd_range(&queue)
}

fn get_hadamard_kernel_event(kernel: &Kernel, queue: &CommandQueue, first: &Buffer<f32>, second: &Buffer<f32>, result: &Buffer<f32>, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(first)
        .set_arg(second)
        .set_arg(result)
        .set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_dyadic_kernel_event(kernel: &Kernel, queue: &CommandQueue, first: &Buffer<f32>, second: &Buffer<f32>, result: &Buffer<f32>, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(first)
        .set_arg(second)
        .set_arg(result)
        .set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_transpose_kernel_event(kernel: &Kernel, queue: &CommandQueue, from: &Buffer<f32>, to: &Buffer<f32>, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(from)
        .set_arg(to)
        .set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_sigmoid_kernel_event(kernel: &Kernel, queue: &CommandQueue, from: &Buffer<f32>, to: &Buffer<f32>, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(from)
        .set_arg(to)
        .set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

fn get_der_sigmoid_kernel_event(kernel: &Kernel, queue: &CommandQueue, from: &Buffer<f32>, to: &Buffer<f32>, m: usize, n: usize) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(from)
        .set_arg(to)
        .set_global_work_sizes(&[pad(m, TS), pad(n, TS), 1])
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

    pub fn finish(&self) {
        self.queue.finish().unwrap();
    }

    pub fn load_program(&self, source: &str) -> Program {
        Program::create_and_build_from_source(&self.context, source, "")
            .expect("Program::create_and_build_from_source failed")
    }

    pub fn load_kernel(&self, program: &Program, kernel_name: &str) -> Kernel {
        Kernel::create(program, kernel_name).expect("Kernel::create failed")
    }

    pub fn load_kernels(&mut self) {
        assert!(IMPLEMENTED_KERNELS.len() == 13, "Can not load all kernels yet");
        if MMUL_VERSION != 1 {
            eprintln!("\x1b[93m[WARNING] You've selected a MMUL_VERSION that only properly works for Matrices with Dim(A)=2^n!\x1b[0m");
        }
        let mut add_kernel = |name: &str, source: &str| {
            let p = self.load_program(source);
            let k = self.load_kernel(&p, name);
            self.kernels.insert(String::from(name), k);
        };
        add_kernel(ADD_MATRIX_NAME, ADD_MATRIX_SOURCE);
        add_kernel(ADD_MATRIX_INLINE_NAME, ADD_MATRIX_INLINE_SOURCE);
        add_kernel(SUB_MATRIX_NAME, SUB_MATRIX_SOURCE);
        add_kernel(SUB_MATRIX_INLINE_NAME, SUB_MATRIX_INLINE_SOURCE);
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
            Err(_) => None
        }
    }

    pub fn read_buffer(&self, buffer: &ClBuffer) -> Result<Vec<f32>> {
        let mut r1 = vec![0.0; buffer.get_buffer_size()];
        let read_event = self.queue.enqueue_read_buffer(&buffer.get_buffer(), CL_BLOCKING, 0, &mut r1, &vec![])?;
        read_event.wait()?;
        Ok(r1)
    }

    pub fn fill_vec(&self, buffer: &ClBuffer, values: Vec<f32>) -> Result<()> {
        match self.kernels.get(&String::from(FILL_MATRIX_VEC_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(k) => {
                let (m, n) = buffer.get_dims();
                if values.len() > m * n {
                    panic!("Unrecoverable error when filling a buffer!\nAttempted to fill the buffer with too many values! Can't fill a {}x{} Matrix with {} values.", m, n, values.len());
                }
                let mut val_bfr = self.create_buffer(m, n).ok_or(ClError(-61))?;
                let _x_write_event = self.queue.enqueue_write_buffer(&mut val_bfr, CL_BLOCKING, 0, &values, &[])?;
    
                let fill_event = get_fill_vec_kernel_event(k,
                    &self.queue, &val_bfr, &buffer.get_buffer(), m, n)?;
                fill_event.wait()?;
                let mut events: Vec<cl_event> = Vec::default();
                events.push(fill_event.get());
                Ok(())
            }
        }
    }

    pub fn fill_scalar(&self, buffer: &ClBuffer, val: f32) -> Result<()> {
        match self.kernels.get(&String::from(FILL_MATRIX_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(k) => {
                let (m, n) = buffer.get_dims();
                let fill_event = get_fill_kernel_event(k,
                    &self.queue, &buffer.get_buffer(), val, m, n)?;
                fill_event.wait()?;
                let mut events: Vec<cl_event> = Vec::default();
                events.push(fill_event.get());
                Ok(())
            }
        }
    }

    pub fn fill_gauss(&self, buffer: &ClBuffer, mean: f32, variance: f32) -> Result<()> {
        let (m, n) = buffer.get_dims();
        let mut r = vec![0.0; m * n];
        let normal = Normal::new(mean, variance).unwrap();
        for i in 0..(m * n) { r[i] = normal.sample(&mut rand::thread_rng()); }
        self.fill_vec(buffer, r)
    }

    pub fn matrix_mult(&self, first: &mut ClBuffer, second: &mut ClBuffer, result: &mut ClBuffer) -> Result<()> {
        match self.kernels.get(&String::from(MUL_MATRIX_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let a = &first.get_buffer();
                let b = &second.get_buffer();
                let c = &result.get_buffer();

                let (m1, n1) = first.get_dims();
                let (n2, k1) = second.get_dims();
                let (m2, k2) = result.get_dims();
                if n1 != n2 {
                    panic!("Unrecoverable error when attempting to multiply two matrices.\nAttempted to multiply a {}x{}-Matrix with a {}x{}-Matrix.", m1, n1, n2, k1);
                }
                if m1 != m2 || k1 != k2 {
                    panic!("Unrecoverable error when attempting to multiply two matrices.\nResult dimensions do not match inputs: Input would yield a {}x{} matrix, but got {}x{} buffer.", m1, k1, m2, k2);
                }
                let m = m1;
                let n = n1;
                let k = k1;
                let kernel_event = get_mmul_kernel_event(&kernel, &self.queue, a, b, c, m, k, n)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())
            }
        }
    }

    pub fn matrix_add(&self, first: &mut ClBuffer, second: &mut ClBuffer, result: &mut ClBuffer) -> Result<()> {
        match self.kernels.get(&String::from(ADD_MATRIX_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let a = &first.get_buffer();
                let b = &second.get_buffer();
                let c = &result.get_buffer();

                let (m1, n1) = first.get_dims();
                let (m2, n2) = second.get_dims();
                let (m3, n3) = result.get_dims();
                if (m1 != m2 || m2 != m3) || (n1 != n2 || n2 != n3) {
                    panic!("Unrecoverable error when attempting to add two matrices.\nAttempted to add a {}x{}-Matrix with a {}x{}-Matrix.", m1, n1, m2, n2);
                }
                let m = m1;
                let n = n1;

                let kernel_event = get_madd_kernel_event(&kernel, &self.queue, a, b, c, m, n)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())

            }
        }
    }

    pub fn matrix_sub(&self, first: &mut ClBuffer, second: &mut ClBuffer, result: &mut ClBuffer) -> Result<()> {
        match self.kernels.get(&String::from(SUB_MATRIX_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let a = &first.get_buffer();
                let b = &second.get_buffer();
                let c = &result.get_buffer();

                let (m1, n1) = first.get_dims();
                let (m2, n2) = second.get_dims();
                let (m3, n3) = result.get_dims();
                if (m1 != m2 || m2 != m3) || (n1 != n2 || n2 != n3) {
                    panic!("Unrecoverable error when attempting to subtract two matrices.\nAttempted to subtract a {}x{}-Matrix with a {}x{}-Matrix.", m1, n1, m2, n2);
                }
                let m = m1;
                let n = n1;

                let kernel_event = get_msub_kernel_event(&kernel, &self.queue, a, b, c, m, n)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())

            }
        }
    }

    pub fn matrix_add_inline(&self, target: &mut ClBuffer, other: &mut ClBuffer) -> Result<()> {
        match self.kernels.get(&String::from(ADD_MATRIX_INLINE_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let a = &target.get_buffer();
                let b = &other.get_buffer();

                let (m1, n1) = target.get_dims();
                let (m2, n2) = other.get_dims();
                if m1 != m2 || n1 != n2 {
                    panic!("Unrecoverable error when attempting to add two matrices.\nAttempted to add a {}x{}-Matrix with a {}x{}-Matrix.", m1, n1, m2, n2);
                }
                let m = m1;
                let n = n1;

                let kernel_event = get_madd_inline_kernel_event(&kernel, &self.queue, a, b, m, n)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())

            }
        }
    }

    pub fn matrix_sub_inline(&self, target: &mut ClBuffer, other: &mut ClBuffer) -> Result<()> {
        match self.kernels.get(&String::from(SUB_MATRIX_INLINE_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let a = &target.get_buffer();
                let b = &other.get_buffer();

                let (m1, n1) = target.get_dims();
                let (m2, n2) = other.get_dims();
                if m1 != m2 || n1 != n2 {
                    panic!("Unrecoverable error when attempting to subtract two matrices.\nAttempted to subtract a {}x{}-Matrix with a {}x{}-Matrix.", m1, n1, m2, n2);
                }
                let m = m1;
                let n = n1;

                let kernel_event = get_msub_inline_kernel_event(&kernel, &self.queue, a, b, m, n)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())

            }
        }
    }

    pub fn matrix_scalar_mult(&self, buffer: &mut ClBuffer, scalar: f32) -> Result<()> {
        match self.kernels.get(&String::from(MUL_SCALAR_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let a = &buffer.get_buffer();
                
                let (m1, n1) = buffer.get_dims();
                let m = m1;
                let n = n1;

                let kernel_event = get_smul_kernel_event(&kernel, &self.queue, a, scalar, m, n)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())

            }
        }
    }

    pub fn matrix_hadamard(&self, first: &mut ClBuffer, second: &mut ClBuffer, result: &mut ClBuffer) -> Result<()> {
        match self.kernels.get(&String::from(HADAMARD_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let a = &first.get_buffer();
                let b = &second.get_buffer();
                let c = &result.get_buffer();
                
                let (m1, n1) = first.get_dims();
                let (m2, n2) = second.get_dims();
                let (m3, n3) = result.get_dims();
                if (m1 != m2 || m2 != m3) || (n1 != n2 || n2 != n3) {
                    panic!("Unrecoverable error when attempting to get the hadamard product of two matrices.\nAttempted to calculate product of a {}x{}-Matrix with a {}x{}-Matrix.", m1, n1, m2, n2);
                }
                let m = m1;
                let n = n1;

                let kernel_event = get_hadamard_kernel_event(&kernel, &self.queue, a, b, c, m, n)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())
            }
        }
    }
    
    pub fn matrix_transpose(&self, from: &mut ClBuffer, to: &mut ClBuffer) -> Result<()> {
        match self.kernels.get(&String::from(TRANSPOSE_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let a = &from.get_buffer();
                let b = &to.get_buffer();
                
                let (m1, n1) = from.get_dims();
                let (m2, n2) = to.get_dims();
                if m1 != n2 || n1 != m2 {
                    panic!("Unrecoverable error when attempting to transpose a Matrix.\nAttempted to transpose a {}x{}-Matrix into a {}x{}-Matrix.", m1, n1, m2, n2);
                }

                let m = m1;
                let n = n1;

                let kernel_event = get_transpose_kernel_event(&kernel, &self.queue, a, b, m, n)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())

            }
        }
    }

    pub fn matrix_dyadic(&self, first: &mut ClBuffer, second: &mut ClBuffer, result: &mut ClBuffer) -> Result<()> {
        match self.kernels.get(&String::from(DYADIC_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let a = &first.get_buffer();
                let b = &second.get_buffer();
                let c = &result.get_buffer();

                let (m1, n1) = first.get_dims();
                let (m2, n2) = second.get_dims();
                let (m3, n3) = result.get_dims();
                if n1 != 1 {
                    panic!("Unrecoverable error when attempting to calculate dyadic product of two matrices.\nExpected Column Vector as first argument, got a {}x{}-Matrix.", m1, n1);
                } else if m2 != 1 {
                    panic!("Unrecoverable error when attempting to calculate dyadic product of two matrices.\nExpected Row Vector as second argument, got a {}x{}-Matrix.", m2, n2);
                } else if m1 != m3 || n2 != n3 {
                    panic!("Unrecoverable error when attempting to calculate dyadic product of two matrices.\nResult dimensions are incorrect. Expected {}x{}-Matrix, got {}x{}.", m1, n2, m3, n3);
                }
                let m = m3;
                let n = n3;

                let kernel_event = get_dyadic_kernel_event(&kernel, &self.queue, a, b, c, m, n)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())
            }
        }
    }

    pub fn sigmoid(&self, from: &mut ClBuffer, to: &mut ClBuffer) -> Result<()> {
        match self.kernels.get(&String::from(SIGMOID_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let a = &from.get_buffer();
                let b = &to.get_buffer();

                let (m1, n1) = from.get_dims();
                let (m2, n2) = to.get_dims();
                if m1 != m2 || n1 != n2 {
                    panic!("Unrecoverable error when attempting to apply Sigmoid to Matrix.\nAttempted to write the results from a {}x{}-Matrix to a {}x{}-Matrix.", m1, n1, m2, n2);
                }
                let m = m1;
                let n = n1;

                let kernel_event = get_sigmoid_kernel_event(&kernel, &self.queue, a, b, m, n)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())
            }
        }
    }

    pub fn der_sigmoid(&self, from: &mut ClBuffer, to: &mut ClBuffer) -> Result<()> {
        match self.kernels.get(&String::from(DER_SIGMOID_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(kernel) => {
                let a = &from.get_buffer();
                let b = &to.get_buffer();

                let (m1, n1) = from.get_dims();
                let (m2, n2) = to.get_dims();
                if m1 != m2 || n1 != n2 {
                    panic!("Unrecoverable error when attempting to apply Derivative of Sigmoid to Matrix.\nAttempted to write the results from a {}x{}-Matrix to a {}x{}-Matrix.", m1, n1, m2, n2);
                }
                let m = m1;
                let n = n1;

                let kernel_event = get_der_sigmoid_kernel_event(&kernel, &self.queue, a, b, m, n)?;
                    
                let mut events: Vec<cl_event> = Vec::default();
                events.push(kernel_event.get());

                Ok(())
            }
        }
    }

    pub fn copy_buffer(&self, from: &mut ClBuffer, to: &mut ClBuffer) -> Result<()> {
        match self.kernels.get(&String::from(FILL_MATRIX_VEC_NAME)) {
            None => panic!("Could not find kernel in HashMap!"),
            Some(k) => {
                let a = &from.get_buffer();
                let b = &to.get_buffer();

                let (m1, n1) = from.get_dims();
                let (m2, n2) = to.get_dims();
                if m1 != m2 || n1 != n2 {
                    panic!("Unrecoverable error when attempting to copy a Matrix.\nAttempted to write the results from a {}x{}-Matrix to a {}x{}-Matrix.", m1, n1, m2, n2);
                }
                let m = m1;
                let n = n1;

                let fill_event = get_fill_vec_kernel_event(k,&self.queue, a, b, m, n)?;
                let mut events: Vec<cl_event> = Vec::default();
                events.push(fill_event.get());
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::{cl_kernel::*, matrix::Matrix};

    const SIZE: usize = 317;
    const BUFFER_SIZE: usize = SIZE * SIZE;

    const V1: f32 = 4.0;
    const V2: f32 = 3.0;

    fn initialize() -> ClStruct {
        let mut c = ClStruct::new().unwrap();
        c.load_kernels();
        c
    }

    #[test]
    fn test_create_buffer() {
        let cl_struct = initialize();

        let bfr = cl_struct.create_buffer(SIZE, SIZE);
        assert!(bfr.is_some(), "create_buffer() should be able to create a {}x{} buffer.", SIZE, SIZE);
    }

    #[test]
    fn test_create_and_fill_buffer_too_big() {
        let cl_struct = initialize();
        const S: usize = 1 << 24;

        let bfr = ClBuffer::new(&cl_struct, S, S);

        let e = cl_struct.fill_scalar(&bfr, 1.0);
        assert!(e.is_err(), "fill_buffer() should fail unless you have {}GB of VRAM.", (S * S * 4) / (1024 * 1024 * 1024));
    }

    #[test]
    fn test_fill_buffer_scalar() {
        let cl_struct = initialize();

        let bfr = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let e = cl_struct.fill_scalar(&bfr, V1);
        assert!(e.is_ok(), "fill_scalar() did not work properly: {:?}", e.err().unwrap());

        let r = cl_struct.read_buffer(&bfr);
        assert!(r.is_ok(), "Could not read from the buffer: {:?}", r.err().unwrap());

        let r2 = vec![V1; BUFFER_SIZE];
        assert_eq!(r.unwrap(), r2, "fill_scalar() did not fill the whole buffer.");
    }

    #[test]
    fn test_fill_buffer_vec_full() {
        let cl_struct = initialize();

        let bfr = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let v = vec![V2; BUFFER_SIZE];
        let e = cl_struct.fill_vec(&bfr, v.clone());
        assert!(e.is_ok(), "fill_vec() did not work properly: {:?}", e.err().unwrap());

        let r = cl_struct.read_buffer(&bfr);
        assert!(r.is_ok(), "Could not read from the buffer: {:?}", r.err().unwrap());

        assert_eq!(r.unwrap(), v, "fill_vec() was supposed to fill the whole buffer.");
    }

    #[test]
    fn test_fill_buffer_vec_part() {
        let cl_struct= initialize();

        let bfr = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let v = vec![V1 * V2; SIZE];
        let e = cl_struct.fill_vec(&bfr, v.clone());
        assert!(e.is_ok(), "fill_vec() did not work properly: {:?}", e.err().unwrap());

        let r = cl_struct.read_buffer(&bfr);
        assert!(r.is_ok(), "Could not read from the buffer: {:?}", r.err().unwrap());            
        let r = r.unwrap();

        assert_eq!(r[0..SIZE], v, "First {} elements of the buffer are supposed to contain {}", SIZE, v[0]);
        assert!(r[SIZE..].iter().all(|v| *v == 0.0), "Last {} elements of the buffer are supposed to only contain 0s.", SIZE);
    }
    
    #[test]
    fn test_matrix_matrix_addition_square() {
        let cl_struct = initialize();

        let mut bfr1 = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let mut bfr2 = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let mut bfr3 = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_add(&mut bfr1, &mut bfr2, &mut bfr3);
        assert!(r.is_ok(), "matrix_add() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr3);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(SIZE, SIZE);
        m1.fill(V1);
        let mut m2 = Matrix::new(SIZE, SIZE);
        m2.fill(V2);
        let a = m1.add(&m2).unwrap().get_all();
        assert_eq!(c, a, "matrix_add() did not work properly");
    }

    #[test]
    fn test_matrix_matrix_addition_arbitrary() {
        let cl_struct = initialize();

        let m = 12908;
        let n = 1290;
        
        let mut bfr1 = ClBuffer::new(&cl_struct, m, n);
        let mut bfr2 = ClBuffer::new(&cl_struct, m, n);
        let mut bfr3 = ClBuffer::new(&cl_struct, m, n);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_add(&mut bfr1, &mut bfr2, &mut bfr3);
        assert!(r.is_ok(), "matrix_add() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr3);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();
        
        let mut m1 = Matrix::new(m, n);
        m1.fill(V1);
        let mut m2 = Matrix::new(m, n);
        m2.fill(V2);
        let a = m1.add(&m2).unwrap().get_all();
        assert_eq!(c, a, "matrix_add() did not work properly");
    }
    
    #[test]
    fn test_matrix_matrix_subtraction_square() {
        let cl_struct = initialize();

        let mut bfr1 = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut bfr2 = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut bfr3 = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_sub(&mut bfr1, &mut bfr2, &mut bfr3);
        assert!(r.is_ok(), "matrix_sub() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr3);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();
        
        let mut m1 = Matrix::new(SIZE, SIZE);
        m1.fill(V1);
        let mut m2 = Matrix::new(SIZE, SIZE);
        m2.fill(V2);
        let a = m1.sub(&m2).unwrap().get_all();
        assert_eq!(c, a, "matrix_sub() did not work properly");
    }

    #[test]
    fn test_matrix_matrix_subtraction_arbitrary() {
        let cl_struct = initialize();

        let m = 12908;
        let n = 1290;

        let mut bfr1 = ClBuffer::new(&cl_struct, m, n);
        let mut bfr2 = ClBuffer::new(&cl_struct, m, n);
        let mut bfr3 = ClBuffer::new(&cl_struct, m, n);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_sub(&mut bfr1, &mut bfr2, &mut bfr3);
        assert!(r.is_ok(), "matrix_sub() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr3);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();
        

        let mut m1 = Matrix::new(m, n);
        m1.fill(V1);
        let mut m2 = Matrix::new(m, n);
        m2.fill(V2);
        let a = m1.sub(&m2).unwrap().get_all();
        assert_eq!(c, a, "matrix_sub() did not work properly");
    }

    #[test]
    fn test_matrix_matrix_multiplication_square() {
        let cl_struct = initialize();

        let mut bfr1 = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut bfr2 = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut bfr3 = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_mult(&mut bfr1, &mut bfr2, &mut bfr3);
        assert!(r.is_ok(), "matrix_mult() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr3);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(SIZE, SIZE);
        m1.fill(V1);
        let mut m2 = Matrix::new(SIZE, SIZE);
        m2.fill(V2);
        let a = m1.multiply(&m2).unwrap().get_all();
        assert_eq!(c, a, "matrix_mult() did not work properly");
    }

    #[test]
    fn test_matrix_matrix_multiplication_arbitrary() {
        let cl_struct = initialize();

        let m = 191;
        let n = 381;
        let k = 396;
        
        let mut bfr1 = ClBuffer::new(&cl_struct, m, k);
        let mut bfr2 = ClBuffer::new(&cl_struct, k, n);
        let mut bfr3 = ClBuffer::new(&cl_struct, m, n);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_mult(&mut bfr1, &mut bfr2, &mut bfr3);
        assert!(r.is_ok(), "matrix_mult() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr3);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(m, k);
        m1.fill(V1);
        let mut m2 = Matrix::new(k, n);
        m2.fill(V2);
        let r1 = m1.multiply(&m2).unwrap();
        assert_eq!(c, r1.get_all(), "matrix_mult() did not work properly");
    }
    
    #[test]
    fn test_matrix_scalar_multiplication_square() {
        let cl_struct = initialize();

        let mut bfr1 = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let r = cl_struct.matrix_scalar_mult(&mut bfr1, V2);
        assert!(r.is_ok(), "matrix_scalar_mult() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr1);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(SIZE, SIZE);
        m1.fill(V1);
        m1.multiply_scalar(V2);
        assert_eq!(c, m1.get_all(), "matrix_scalar_mult() did not work properly");
    }

    #[test]
    fn test_matrix_scalar_multiplication_arbitrary() {
        let cl_struct = initialize();

        let m = 1290;
        let n = 819;

        let mut bfr1 = ClBuffer::new(&cl_struct, m, n);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let r = cl_struct.matrix_scalar_mult(&mut bfr1, V2);
        assert!(r.is_ok(), "matrix_scalar_mult() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr1);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(m, n);
        m1.fill(V1);
        m1.multiply_scalar(V2);
        assert_eq!(c, m1.get_all(), "matrix_scalar_mult() did not work properly");
    }
    
    #[test]
    fn test_matrix_matrix_addition_inline_square() {
        let cl_struct = initialize();

        let mut bfr1 = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut bfr2 = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_add_inline(&mut bfr1, &mut bfr2);
        assert!(r.is_ok(), "matrix_add() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr1);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(SIZE, SIZE);
        m1.fill(V1);
        let mut m2 = Matrix::new(SIZE, SIZE);
        m2.fill(V2);
        m1 = m1.add(&m2).unwrap();
        assert_eq!(c, m1.get_all(), "matrix_add() did not work properly");
    }

    #[test]
    fn test_matrix_matrix_addition_inline_arbitrary() {
        let cl_struct = initialize();

        let m = 1290;
        let n = 8192;

        let mut bfr1 = ClBuffer::new(&cl_struct, m, n);
        let mut bfr2 = ClBuffer::new(&cl_struct, m, n);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_add_inline(&mut bfr1, &mut bfr2);
        assert!(r.is_ok(), "matrix_add_inline() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr1);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(m, n);
        m1.fill(V1);
        let mut m2 = Matrix::new(m, n);
        m2.fill(V2);
        m1 = m1.add(&m2).unwrap();
        assert_eq!(c, m1.get_all(), "matrix_add() did not work properly");
    }
     
    #[test]
    fn test_matrix_matrix_subtraction_inline_square() {
        let cl_struct = initialize();

        let mut bfr1 = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut bfr2 = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_sub_inline(&mut bfr1, &mut bfr2);
        assert!(r.is_ok(), "matrix_sub_inline() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr1);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(SIZE, SIZE);
        m1.fill(V1);
        let mut m2 = Matrix::new(SIZE, SIZE);
        m2.fill(V2);
        m1 = m1.sub(&m2).unwrap();
        assert_eq!(c, m1.get_all(), "matrix_sub() did not work properly");
    }

    #[test]
    fn test_matrix_matrix_subtraction_inline_arbitrary() {
        let cl_struct = initialize();

        let m = 12098;
        let n = 2198;

        let mut bfr1 = ClBuffer::new(&cl_struct, m, n);
        let mut bfr2 = ClBuffer::new(&cl_struct, m, n);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_sub_inline(&mut bfr1, &mut bfr2);
        assert!(r.is_ok(), "matrix_sub_inline() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr1);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(m, n);
        m1.fill(V1);
        let mut m2 = Matrix::new(m, n);
        m2.fill(V2);
        m1 = m1.sub(&m2).unwrap();
        assert_eq!(c, m1.get_all(), "matrix_sub() did not work properly");
    }
   
    #[test]
    fn test_matrix_matrix_hadamard_square() {
        let cl_struct = initialize();

        let mut bfr1 = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut bfr2 = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut bfr3 = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_hadamard(&mut bfr1, &mut bfr2, &mut bfr3);
        assert!(r.is_ok(), "matrix_hadamard() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr3);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(SIZE, SIZE);
        m1.fill(V1);
        let mut m2 = Matrix::new(SIZE, SIZE);
        m2.fill(V2);
        let a = m1.hadamard_product(&m2).unwrap().get_all();
        assert_eq!(c, a, "matrix_hadamard() did not work properly");
    }

    #[test]
    fn test_matrix_matrix_hadamard_arbitrary() {
        let cl_struct = initialize();

        let m = 1201;
        let n = 120;

        let mut bfr1 = ClBuffer::new(&cl_struct, m, n);
        let mut bfr2 = ClBuffer::new(&cl_struct, m, n);
        let mut bfr3 = ClBuffer::new(&cl_struct, m, n);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_scalar() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_scalar() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_hadamard(&mut bfr1, &mut bfr2, &mut bfr3);
        assert!(r.is_ok(), "matrix_hadamard() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr3);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(m, n);
        m1.fill(V1);
        let mut m2 = Matrix::new(m, n);
        m2.fill(V2);
        let a = m1.hadamard_product(&m2).unwrap().get_all();
        assert_eq!(c, a, "matrix_hadamard() did not work properly");
    }
    
    #[test]
    fn test_matrix_transpose_square() {
        let cl_struct = initialize();

        let mut r1 = vec![0.0; BUFFER_SIZE];
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..BUFFER_SIZE { r1[i] = normal.sample(&mut rand::thread_rng()); }

        let mut bfr1 = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut bfr2 = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let f1 = cl_struct.fill_vec(&bfr1, r1.clone());
        assert!(f1.is_ok(), "fill_vec() did not work properly: {:?}", f1.err().unwrap());

        let r = cl_struct.matrix_transpose(&mut bfr1, &mut bfr2);
        assert!(r.is_ok(), "matrix_transpose() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr2);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m = Matrix::new(SIZE, SIZE);
        m.fill_vec(&r1);
        let m = m.transpose();
        assert_eq!(c, m.get_all(), "matrix_transpose() did not work properly!");
    }

    #[test]
    fn test_matrix_transpose_arbitrary() {
        let cl_struct = initialize();
        let m = 2910;
        let n = 574;

        let mut r1 = vec![0.0; m * n];
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..(m * n) { r1[i] = normal.sample(&mut rand::thread_rng()); }

        let mut bfr1 = ClBuffer::new(&cl_struct, m, n);
        let mut bfr2 = ClBuffer::new(&cl_struct, n, m);

        let f1 = cl_struct.fill_vec(&bfr1, r1.clone());
        assert!(f1.is_ok(), "fill_vec() did not work properly: {:?}", f1.err().unwrap());

        let r = cl_struct.matrix_transpose(&mut bfr1, &mut bfr2);
        assert!(r.is_ok(), "matrix_transpose() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr2);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m = Matrix::new(m, n);
        m.fill_vec(&r1);
        let m = m.transpose();
        assert_eq!(c, m.get_all(), "matrix_transpose() did not work properly!");
    }

    #[test]
    fn test_matrix_matrix_dyadic_square() {
        let cl_struct = initialize();
        
        let mut bfr1 = ClBuffer::new(&cl_struct, SIZE, 1);
        let mut bfr2 = ClBuffer::new(&cl_struct, 1, SIZE);
        let mut bfr3 = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_vec() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_vec() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_dyadic(&mut bfr1, &mut bfr2, &mut bfr3);
        assert!(r.is_ok(), "matrix_dyadic() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr3);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(SIZE, 1);
        m1.fill(V1);
        let mut m2 = Matrix::new(1, SIZE);
        m2.fill(V2);
        let m3 = m1.dyadic_product(&m2).unwrap();
        assert_eq!(c, m3.get_all(), "matrix_dyadic() did not work properly!");
    }

    #[test]
    fn test_matrix_matrix_dyadic_arbitrary() {
        let cl_struct = initialize();

        let m = 1028;
        let n = 102;
        
        let mut bfr1 = ClBuffer::new(&cl_struct, m, 1);
        let mut bfr2 = ClBuffer::new(&cl_struct, 1, n);
        let mut bfr3 = ClBuffer::new(&cl_struct, m, n);

        let f1 = cl_struct.fill_scalar(&bfr1, V1);
        assert!(f1.is_ok(), "fill_vec() did not work properly: {:?}", f1.err().unwrap());

        let f2 = cl_struct.fill_scalar(&bfr2, V2);
        assert!(f2.is_ok(), "fill_vec() did not work properly: {:?}", f2.err().unwrap());

        let r = cl_struct.matrix_dyadic(&mut bfr1, &mut bfr2, &mut bfr3);
        assert!(r.is_ok(), "matrix_dyadic() did not work properly: {:?}", r.err().unwrap());

        let c = cl_struct.read_buffer(&bfr3);
        assert!(c.is_ok(), "Could not read from the buffer: {:?}", c.err().unwrap());
        let c = c.unwrap();

        let mut m1 = Matrix::new(m, 1);
        m1.fill(V1);
        let mut m2 = Matrix::new(1, n);
        m2.fill(V2);
        let m3 = m1.dyadic_product(&m2).unwrap();
        assert_eq!(c, m3.get_all(), "matrix_dyadic() did not work properly!");
    }
    
    #[test]
    fn test_matrix_sigmoid_square() {
        let cl_struct = initialize();

        let sigmoid = |val: f32| -> f32 {
            1.0 /  (1.0 + (-val).exp())
        };
        
        let mut from = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut to = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let mut r1 = vec![0.0; BUFFER_SIZE];
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..BUFFER_SIZE { r1[i] = normal.sample(&mut rand::thread_rng()); }

        let f1 = cl_struct.fill_vec(&from, r1.clone());
        assert!(f1.is_ok(), "fill_vec() did not work properly: {:?}", f1.err().unwrap());

        let e = cl_struct.sigmoid(&mut from, &mut to);
        assert!(e.is_ok(), "sigmoid() did not work properly: {:?}", e.err().unwrap());

        let r2 = cl_struct.read_buffer(&to);
        assert!(r2.is_ok(), "read_buffer() did not work properly: {:?}", r2.err().unwrap());
        let r2 = r2.unwrap();

        let r1: Vec<f32> = r1.iter().map(|f| sigmoid(*f)).collect();
        
        let e = r1.iter().zip(r2.iter()).all(|(f1, f2)|(f1 - f2).abs() <= f32::EPSILON);
        assert!(e, "sigmoid() did not work properly!");
    }

    #[test]
    fn test_matrix_sigmoid_arbitrary() {
        let cl_struct = initialize();

        let m = 1290;
        let n = 51;

        let sigmoid = |val: f32| -> f32 {
            1.0 /  (1.0 + (-val).exp())
        };

        let mut from = ClBuffer::new(&cl_struct, m, n);
        let mut to = ClBuffer::new(&cl_struct, m, n);

        let mut r1 = vec![0.0; m * n];
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..(m * n) { r1[i] = normal.sample(&mut rand::thread_rng()); }

        let f1 = cl_struct.fill_vec(&from, r1.clone());
        assert!(f1.is_ok(), "fill_vec() did not work properly: {:?}", f1.err().unwrap());

        let e = cl_struct.sigmoid(&mut from, &mut to);
        assert!(e.is_ok(), "sigmoid() did not work properly: {:?}", e.err().unwrap());

        let r2 = cl_struct.read_buffer(&to);
        assert!(r2.is_ok(), "read_buffer() did not work properly: {:?}", r2.err().unwrap());
        let r2 = r2.unwrap();

        let r1: Vec<f32> = r1.iter().map(|f| sigmoid(*f)).collect();
        
        let e = r1.iter().zip(r2.iter()).all(|(f1, f2)|(f1 - f2).abs() <= f32::EPSILON);
        assert!(e, "sigmoid() did not work properly!");
    }
    
    #[test]
    fn test_matrix_derivative_sigmoid_square() {
        let cl_struct = initialize();

        let sigmoid = |val: f32| -> f32 {
            1.0 /  (1.0 + (-val).exp())
        };

        let der_sigmoid = |val: f32| -> f32 {
            let t = sigmoid(val);
            t * (1.0 - t)
        };

        let mut from = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut to = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let mut r1 = vec![0.0; BUFFER_SIZE];
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..BUFFER_SIZE { r1[i] = normal.sample(&mut rand::thread_rng()); }

        let f1 = cl_struct.fill_vec(&from, r1.clone());
        assert!(f1.is_ok(), "fill_vec() did not work properly: {:?}", f1.err().unwrap());

        let e = cl_struct.der_sigmoid(&mut from, &mut to);
        assert!(e.is_ok(), "der_sigmoid() did not work properly: {:?}", e.err().unwrap());

        let r2 = cl_struct.read_buffer(&to);
        assert!(r2.is_ok(), "read_buffer() did not work properly: {:?}", r2.err().unwrap());
        let r2 = r2.unwrap();

        let r1: Vec<f32> = r1.iter().map(|f| der_sigmoid(*f)).collect();
        
        let e = r1.iter().zip(r2.iter()).all(|(f1, f2)|(f1 - f2).abs() <= f32::EPSILON);
        assert!(e, "der_sigmoid() did not work properly!");
    }

    #[test]
    fn test_matrix_derivative_sigmoid_arbitrary() {
        let cl_struct = initialize();

        let m = 1290;
        let n = 51;


        let sigmoid = |val: f32| -> f32 {
            1.0 /  (1.0 + (-val).exp())
        };

        let der_sigmoid = |val: f32| -> f32 {
            let t = sigmoid(val);
            t * (1.0 - t)
        };

        let mut from = ClBuffer::new(&cl_struct, m, n);
        let mut to = ClBuffer::new(&cl_struct, m, n);

        let mut r1 = vec![0.0; m * n];
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..(m * n) { r1[i] = normal.sample(&mut rand::thread_rng()); }

        let f1 = cl_struct.fill_vec(&from, r1.clone());
        assert!(f1.is_ok(), "fill_vec() did not work properly: {:?}", f1.err().unwrap());

        let e = cl_struct.der_sigmoid(&mut from, &mut to);
        assert!(e.is_ok(), "der_sigmoid() did not work properly: {:?}", e.err().unwrap());

        let r2 = cl_struct.read_buffer(&to);
        assert!(r2.is_ok(), "read_buffer() did not work properly: {:?}", r2.err().unwrap());
        let r2 = r2.unwrap();

        let r1: Vec<f32> = r1.iter().map(|f| der_sigmoid(*f)).collect();
        
        let e = r1.iter().zip(r2.iter()).all(|(f1, f2)|(f1 - f2).abs() <= f32::EPSILON);
        assert!(e, "der_sigmoid() did not work properly!");
    }
    
    #[test]
    fn test_matrix_copy_buffer_square() {
        let cl_struct = initialize();

        let mut from = ClBuffer::new(&cl_struct, SIZE, SIZE);
        let mut to = ClBuffer::new(&cl_struct, SIZE, SIZE);

        let mut r1 = vec![0.0; BUFFER_SIZE];
        for i in 0..BUFFER_SIZE { r1[i] = rand::thread_rng().gen_range(0.0..1.0); }

        let f1 = cl_struct.fill_vec(&from, r1.clone());
        assert!(f1.is_ok(), "fill_vec() did not work properly: {:?}", f1.err().unwrap());

        let e = cl_struct.copy_buffer(&mut from, &mut to);
        assert!(e.is_ok(), "copy_buffer() did not work properly: {:?}", e.err().unwrap());

        let r1 = cl_struct.read_buffer(&from);
        assert!(r1.is_ok(), "read_buffer() did not work properly: {:?}", r1.err().unwrap());
        let r1 = r1.unwrap();
        let r2 = cl_struct.read_buffer(&to);
        assert!(r2.is_ok(), "read_buffer() did not work properly: {:?}", r2.err().unwrap());
        let r2 = r2.unwrap();
        
        let e = r1.iter().zip(r2.iter()).all(|(f1, f2)|(f1 - f2).abs() <= f32::EPSILON);
        assert!(e, "copy_buffer() did not work properly!");
    }

    #[test]
    fn test_matrix_copy_buffer_arbitrary() {
        let cl_struct = initialize();

        let m = 1290;
        let n = 51;

        let mut from = ClBuffer::new(&cl_struct, m, n);
        let mut to = ClBuffer::new(&cl_struct, m, n);

        let mut r1 = vec![0.0; m * n];
        for i in 0..(m * n) { r1[i] = rand::thread_rng().gen_range(0.0..1.0); }

        let f1 = cl_struct.fill_vec(&from, r1.clone());
        assert!(f1.is_ok(), "fill_vec() did not work properly: {:?}", f1.err().unwrap());

        let e = cl_struct.copy_buffer(&mut from, &mut to);
        assert!(e.is_ok(), "copy_buffer() did not work properly: {:?}", e.err().unwrap());

        let r1 = cl_struct.read_buffer(&from);
        assert!(r1.is_ok(), "read_buffer() did not work properly: {:?}", r1.err().unwrap());
        let r1 = r1.unwrap();
        let r2 = cl_struct.read_buffer(&to);
        assert!(r2.is_ok(), "read_buffer() did not work properly: {:?}", r2.err().unwrap());
        let r2 = r2.unwrap();
        
        let e = r1.iter().zip(r2.iter()).all(|(f1, f2)|(f1 - f2).abs() <= f32::EPSILON);
        assert!(e, "copy_buffer() did not work properly!");
    }
    
    /*
    pub fn copy_buffer */
    
}